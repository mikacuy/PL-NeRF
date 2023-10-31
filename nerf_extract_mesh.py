'''
Use a different learning rate for the coarse network
Use constant aggregation for the first few iterations
'''
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm, trange

from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity
from lpips import LPIPS
import configargparse
import datetime
import math
import cv2
import shutil

from run_nerf_helpers import *

from load_llff import load_llff_data
# from load_dtu import load_dtu, load_dtu2
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, load_scene_blender_fixed_dist_new, load_scene_blender2
from load_LINEMOD import load_LINEMOD_data

from natsort import natsorted 
from argparse import Namespace
import trimesh
from tqdm import tqdm, trange
import mcubes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
DEBUG = False

def build_json_for_dtu(splits, intrinsics, poses, near, far):
    out_dict = {}
    out_dict = {"near": near,
                "far": far}
    i_train, i_test = splits
    train_dicts = []
    test_dicts = []
    for i in i_train:
        train_dict = {}
        train_dict["extrinsic"] = poses[i].tolist()
        train_dict["intrinsic"] = intrinsics[i].tolist()
        train_dict["pose_id"] = int(i) 
        train_dicts.append(train_dict)
    for i in i_test:
        test_dict = {}
        test_dict["extrinsic"] = poses[i].tolist()
        test_dict["intrinsic"] = intrinsics[i].tolist()
        test_dict["pose_id"] = int(i) 
        test_dicts.append(test_dict)
    out_dict["train_frames"] = train_dicts
    out_dict["test_frames"] = test_dicts
    return out_dict


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        # input_dirs = viewdirs[:,None].expand(inputs.shape)
        # input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        input_dirs_flat = viewdirs
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    coarse_grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars = list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    optimizer_coarse = torch.optim.Adam(params=coarse_grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(args.ckpt_dir, args.expname, f) for f in sorted(os.listdir(os.path.join(args.ckpt_dir, args.expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'mode' : args.mode,
        'color_mode': args.color_mode,
        'farcolorfix': args.farcolorfix        
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}

    ### set to True for linear
    # render_kwargs_test['perturb'] = False
    render_kwargs_test['perturb'] = True

    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_coarse

def compute_weights(raw, z_vals, rays_d, noise=0.):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10, device=device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    return weights

### Our reformulation to piecewise linear
def compute_weights_piecewise_linear(raw, z_vals, near, far, rays_d, noise=0., return_tau=False):
    raw2expr = lambda raw, dists: torch.exp(-raw*dists)

    ### Concat
    z_vals = torch.cat([near, z_vals, far], -1)


    dists = z_vals[...,1:] - z_vals[...,:-1]

    ### Original code
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    tau = torch.cat([torch.ones((raw.shape[0], 1), device=device)*1e-10, raw[...,3] + noise, torch.ones((raw.shape[0], 1), device=device)*1e10], -1) ### tau(near) = 0, tau(far) = very big (will hit an opaque surface)

    tau = F.relu(tau) ## Make positive from proof of DS-NeRF

    interval_ave_tau = 0.5 * (tau[...,1:] + tau[...,:-1])

    '''
    Evaluating exp(-0.5 (tau_{i+1}+tau_i) (s_{i+1}-s_i) )
    '''
    expr = raw2expr(interval_ave_tau, dists)  # [N_rays, N_samples+1]

    ### Transmittance until s_n
    T = torch.cumprod(torch.cat([torch.ones((expr.shape[0], 1), device=device), expr], -1), -1) # [N_rays, N_samples+2], T(near)=1, starts off at 1

    ### Factor to multiply transmittance with
    factor = (1 - expr)

    weights = factor * T[:, :-1] # [N_rays, N_samples+1]

    if return_tau:
        return weights, tau, T
    else:
        return weights    


def raw2outputs(raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std=0, pytest=False, white_bkgd=False, farcolorfix=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    if mode == "linear":
        weights, tau, T = compute_weights_piecewise_linear(raw, z_vals, near, far, rays_d, noise, return_tau=True)

        if color_mode == "midpoint":

            if farcolorfix:
                rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb, torch.zeros((rgb[:, -1].shape), device=device).unsqueeze(1)], 1)

            else:
                rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb, rgb[: ,-1, :].unsqueeze(1)], 1)

            rgb_mid = .5 * (rgb_concat[:, 1:, :] + rgb_concat[:, :-1, :])

            rgb_map = torch.sum(weights[...,None] * rgb_mid, -2)  # [N_rays, 3]

        elif color_mode == "left":

            rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb], 1)
            rgb_map = torch.sum(weights[...,None] * rgb_concat, -2)

        else:
            print("ERROR: Color mode unimplemented, please select left or midpoint.")

        ### Piecewise linear means take the midpoint
        z_vals = torch.cat([near, z_vals, far], -1)
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        depth_map = torch.sum(weights * z_vals_mid, -1)

    elif mode == "constant":
        weights = compute_weights(raw, z_vals, rays_d, noise)
        
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)    
        
        tau = None
        T = None    

    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)


    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, tau, T


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                mode,
                color_mode,                
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,                
                quad_solution_v2=False,
                zero_tol = 1e-4,
                epsilon = 1e-3,
                farcolorfix = False,
                constant_init = False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    ### If constant init then overwrite mode for coarse model to constant first
    if constant_init:
        # coarse_mode = "constant"
        mode = "constant"
    # else:
    #     coarse_mode = mode

    # print(mode)

#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, tau, T = raw2outputs(raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std, pytest=pytest, white_bkgd=white_bkgd, farcolorfix=farcolorfix)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, z_vals_0, weights_0 = rgb_map, disp_map, acc_map, depth_map, z_vals, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if mode == "linear":
            z_samples, _, _, _ = sample_pdf_reformulation(z_vals, weights, tau, T, near, far, N_importance, det=(perturb==0.), pytest=pytest, quad_solution_v2=quad_solution_v2, zero_threshold = zero_tol, epsilon_=epsilon)
        elif mode == "constant":
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)

        z_samples = z_samples.detach()

        ######## Clamping in quad solution should have fixed this
        z_samples = torch.clamp(z_samples, near, far)
        ########

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, tau, T = raw2outputs(raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std, pytest=pytest, white_bkgd=white_bkgd, farcolorfix=farcolorfix)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['depth0'] = depth_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

#### For mesh extraction ####
def extract_fields(bound_min, bound_max, resolution, query_func, model):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(tqdm(X)):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)

                    viewdirs = torch.zeros_like(pts, device=pts.device)

                    # print(pts.shape)
                    # print(viewdirs.shape)
                    # print(query_func(pts, viewdirs, model).shape)

                    val = query_func(pts, viewdirs, model).reshape(len(xs), len(ys), len(zs), -1)

                    # print(val.shape)

                    taus = F.relu(val[...,3])

                    # print(taus.shape)
                    # print(torch.nonzero(taus))            
                    # exit()

                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = taus.detach().cpu().numpy()
    return u

def extract_iso_level(density, threshold=25):
    # Density boundaries
    min_a, max_a, std_a = density.min(), density.max(), density.std()

    # Adaptive iso level
    iso_value = min(max(threshold, min_a + std_a), max_a - std_a)
    print(f"Min density {min_a}, Max density: {max_a}, Mean density {density.mean()}")
    print(f"Querying based on iso level: {iso_value}")

    return iso_value


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, model, adaptive=False):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func, model)

    if not adaptive:
        vertices, triangles = mcubes.marching_cubes(u, threshold)
    else:
        vertices, triangles = mcubes.marching_cubes(u, extract_iso_level(u, threshold))
        
    try:
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()
    except:
        b_max_np = bound_max
        b_min_np = bound_min        

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles
#############################

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--task', default="train", type=str, help='one out of: "train", "test", "video"')


    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--ckpt_dir", type=str, default="",
                        help='checkpoint directory')

    parser.add_argument("--scene_id", type=str, default="lego",
                        help='scene identifier')
    parser.add_argument("--data_dir", type=str, default="../nerf_synthetic",
                        help='directory containing the scenes')
    parser.add_argument("--dataset", type=str, default="blender", 
                        help='dataset used -- selects which dataloader"')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--coarse_lrate", type=float, default=1e-4, 
                        help='learning rate')    
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--testskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    # parser.add_argument('--white_bkgd', default= False, type=bool)
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--num_iterations", type=int, default=200000, 
                        help='number of iterations for training')

    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=600000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=500000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=500000, 
                        help='frequency of render_poses video saving')

    ### For PWL ###
    parser.add_argument("--mode", type=str, default="constant", 
                        help='rendering opacity aggregation mode -- whether to use piecewise constant (vanilla) or piecewise linear (reformulation)."')
    parser.add_argument("--color_mode", type=str, default="midpoint", 
                        help='rendering color aggregation mode -- whether to use left bin or midpoint."')

    parser.add_argument('--quad_solution_v2', default= True, type=bool)

    ### Epsilon and zero tol in quadratic solution
    parser.add_argument("--zero_tol", type=float, default=1e-4, 
                        help='zero tol to revert to piecewise constant assumption')    
    parser.add_argument("--epsilon", type=float, default=1e-3, 
                        help='epsilon value in the increasing and decreasing cases or max(x,epsilon)')

    parser.add_argument('--set_near_plane', default= 2., type=float)
    parser.add_argument('--farcolorfix', default= False, type=bool)

    parser.add_argument("--constant_init",   type=int, default=1000, 
                        help='number of iterations to use constant aggregation')    

    parser.add_argument("--coarse_weight", type=float, default=1.0, 
                        help='zero tol to revert to piecewise constant assumption') 

    parser.add_argument('--test_dist', default= 1.0, type=float)

    parser.add_argument("--eval_scene_id", type=str, default="chair_rgba_fixdist_nv100_dist0.25-1.0-4_depth_sfn",
                        help='scene identifier for eval')
    parser.add_argument("--eval_data_dir", type=str, default="../nerf_synthetic/fixed_dist_new-rgba/",
                        help='directory containing the scenes for eval')
    
    ### DTU flags 
    parser.add_argument("--dtu_scene_id", type=int, default=21, 
                        help='scan id for DTU dataset to render')
    parser.add_argument("--num_train", type=int, default=40, 
                        help='number of training views to use (1 - 49)')
    parser.add_argument("--dtu_split", type=str, default=None, 
                        help='number of training views to use (1 - 49)')

    ##################

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    print(args.white_bkgd)


    tmp_task = args.task
    tmp_data_dir = args.data_dir
    tmp_scene_id = args.scene_id
    tmp_dataset = args.dataset
    tmp_test_dist = args.test_dist
    tmp_ckpt_dir = args.ckpt_dir
    tmp_set_near_plane = args.set_near_plane

    tmp_white_bkgd = args.white_bkgd
    tmp_eval_scene_id = args.eval_scene_id
    tmp_eval_data_dir = args.eval_data_dir
    # tmp_white_bkgd = False
    tmp_test_skip = args.testskip

    # tmp_mode = args.mode
    # tmp_N_samples = args.N_samples
    # tmp_N_importance = args.N_importance

    # load nerf parameters from training
    args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
    with open(args_file, 'r') as af:
        args_dict = json.load(af)
    args = Namespace(**args_dict)
    # task and paths are not overwritten
    args.task = tmp_task
    args.data_dir = tmp_data_dir
    args.ckpt_dir = tmp_ckpt_dir
    # args.mode = tmp_mode
    args.train_jsonfile = 'transforms_train.json'
    args.set_near_plane = tmp_set_near_plane
    # args.N_samples = tmp_N_samples
    # args.N_importance = tmp_N_importance
    args.dataset = tmp_dataset
    args.test_dist = tmp_test_dist
    args.scene_id = tmp_scene_id
    args.white_bkgd = tmp_white_bkgd 
    args.eval_scene_id = tmp_eval_scene_id 
    args.eval_data_dir = tmp_eval_data_dir
    args.testskip = tmp_test_skip


    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    scene_data_dir = os.path.join(args.data_dir, args.scene_id)
    K = None
    if args.dataset == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(scene_data_dir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, scene_data_dir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(scene_data_dir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, scene_data_dir)
        i_train, i_val, i_test = i_split

        # near = 2.
        near = args.set_near_plane
        print("Set near plane to: " + str(near))
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset == "blender2":
        images, poses, render_poses, hwf, i_split = load_scene_blender2(scene_data_dir, half_res=args.half_res)
        print('Loaded blender2', images.shape, render_poses.shape, hwf, scene_data_dir)
        i_train, i_val, i_test = i_split

        # near = 2.
        near = args.set_near_plane
        print("Set near plane to: " + str(near))
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]  


    elif args.dataset == "blender_fixeddist":
        images, poses, render_poses, hwf, i_split = load_scene_blender_fixed_dist_new(scene_data_dir, half_res=args.half_res, train_dist=1.0, test_dist=args.test_dist)

        print('Loaded blender fixed dist', images.shape, hwf, scene_data_dir)
        i_train, i_val, i_test = i_split

        near = args.set_near_plane
        print("Set near plane to: " + str(near))
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3] 

    elif args.dataset == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(scene_data_dir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    elif args.dataset == 'DTU':
        
        # use the existing split
        if args.dtu_split is not None:
            with open(args.dtu_split, 'r') as ff:
                train_split = json.load(ff)
        else:
            train_split = None
        images, Ks, poses, render_poses, hwf, i_split, near, far, splits = load_dtu(args.data_dir, args.dtu_scene_id, num_train=args.num_train, half_res=args.half_res, train_split=train_split)
        K = Ks[0]
        print(f'Loaded DTU, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_test = i_split
        i_val = i_test
        save_json = build_json_for_dtu(splits, Ks, poses, near, far)
        save_split_file = os.path.join(args.ckpt_dir, args.expname, 'split.json')
        with open(save_split_file, 'w') as f:
            json.dump(save_json, f, indent=4)
        
        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    elif args.dataset == 'DTU2':
        
        # use the existing split
        if args.dtu_split is not None:
            with open(args.dtu_split, 'r') as ff:
                train_split = json.load(ff)
        else:
            train_split = None
        images, K, poses, render_poses, hwf, i_split, near, far, splits = load_dtu2(args.data_dir, args.dtu_scene_id, num_train=args.num_train, half_res=args.half_res, train_split=train_split)
        print(f'Loaded DTU, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_test = i_split
        i_val = i_test
        save_json = build_json_for_dtu(splits, [K]*poses.shape[0], poses, near, far)
        save_split_file = os.path.join(args.ckpt_dir, args.expname, 'split.json')
        with open(save_split_file, 'w') as f:
            json.dump(save_json, f, indent=4)
        
        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    elif args.dataset == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=scene_data_dir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, scene_data_dir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # ##### Load blender meshes #####
    # def as_mesh(scene_or_mesh):
    #     """
    #     Convert a possible scene to a mesh.

    #     If conversion occurs, the returned mesh has only vertex and face data.
    #     """
    #     if isinstance(scene_or_mesh, trimesh.Scene):
    #         if len(scene_or_mesh.geometry) == 0:
    #             mesh = None  # empty scene
    #         else:
    #             # we lose texture information here

    #             for g in scene_or_mesh.geometry.values():
    #                 print(g)

    #             mesh = trimesh.util.concatenate(
    #                 tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
    #                     for g in scene_or_mesh.geometry.values() if g.faces.shape[-1] ==3 ))
    #     else:
    #         assert(isinstance(mesh, trimesh.Trimesh))
    #         mesh = scene_or_mesh
    #     return mesh

    # source_mesh_file = os.path.join(args.data_dir, "NeRF-Meshes", args.scene_id+".obj")
    # gt_mesh = trimesh.load_mesh(source_mesh_file, force='mesh')
    # print("Trimesh load")
    # print(gt_mesh)

    # gt_mesh = as_mesh(gt_mesh)

    # # try:
    # #     gt_mesh = as_mesh(gt_mesh)
    # # except:
    # #     gt_mesh = gt_mesh

    # print(gt_mesh)
    # print(gt_mesh.vertices.shape)
    # print(gt_mesh.faces.shape)
    # print(gt_mesh.is_watertight)

    # gt_mesh.fill_holes()
    # print(gt_mesh.is_watertight)

    # # T2 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    # T1 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    # # mesh = trimesh.Trimesh(gt_mesh.vertices@T1@T2, gt_mesh.faces)
    # mesh = trimesh.Trimesh(gt_mesh.vertices@T1, gt_mesh.faces)
    # mesh.export(os.path.join(args.data_dir, "nerf_meshes_reoriented", args.scene_id+".obj"))

    # exit()
    # ###############################


    ## Load GT mesh
    source_mesh_file = os.path.join(args.data_dir, "nerf_meshes_reoriented", args.scene_id+".obj")
    gt_mesh = trimesh.load_mesh(source_mesh_file, force='mesh')

    try:
        gt_mesh = as_mesh(gt_mesh)
    except:
        gt_mesh = gt_mesh

    ## Define box based on GT mesh
    vertices = gt_mesh.vertices

    # print(vertices.shape)
    # print(vertices)

    max_xyz = np.max(vertices, axis=0) + 0.25
    min_xyz = np.min(vertices, axis=0) - 0.25

    print("GT max min")
    print("Max xyz")
    print(max_xyz)
    print("Min xyz")    
    print(min_xyz)

    ## Get scene bounds to define grid for marching cube extraction
    # ###Compute boundaries of 3D space
    # old_max_xyz = torch.full((3,), -1e6, device=device)
    # old_min_xyz = torch.full((3,), 1e6, device=device)
    # for idx_train in i_train:
    #     rays_o, rays_d = get_rays(H, W, torch.Tensor(K).to(device), torch.Tensor(poses[idx_train]).to(device)) # (H, W, 3), (H, W, 3)
    #     points_3D = rays_o + rays_d * far # [H, W, 3]
    #     old_max_xyz = torch.max(points_3D.view(-1, 3).amax(0), max_xyz)
    #     old_min_xyz = torch.min(points_3D.view(-1, 3).amin(0), min_xyz)

    # print("NeRF max min")
    # print("Max xyz")
    # print(old_max_xyz)
    # print("Min xyz")    
    # print(old_min_xyz)

    # exit()
   

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_coarse = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    ### Extract the mesh
    resolution = 512
    threshold = 25

    vertices, triangles = extract_geometry(min_xyz, max_xyz, resolution, threshold, render_kwargs_test["network_query_fn"], render_kwargs_test["network_fine"])
    # vertices, triangles = extract_geometry(min_xyz, max_xyz, resolution, threshold, render_kwargs_test["network_query_fn"], render_kwargs_test["network_fine"], adaptive=True)
    mesh = trimesh.Trimesh(vertices, triangles)

    # print(mesh)

    ### This code cleans up floaters
    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=10000)

    # print(cc)

    mask = np.zeros(len(mesh.faces), dtype=np.bool)
    mask[np.concatenate(cc)] = True
    mesh.update_faces(mask)

    mesh_outdir = os.path.join('extracted_meshes')
    os.makedirs(os.path.join(mesh_outdir), exist_ok=True)

    fname = "{}_{}_res{}_thresh{}_cleaned.ply".format(args.scene_id, args.mode, resolution, threshold)
    mesh.export(os.path.join(mesh_outdir, fname))

    # fname = "mesh_out_{}_res{}_thresh{}_cleaned.ply".format(args.mode, resolution, threshold)
    # mesh.export(fname)


    print(vertices.shape)
    print(triangles.shape)
    print("Done outputing "+fname)
    exit() 


    lpips_alex = LPIPS()

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(args.ckpt_dir, args.expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return



    elif args.task == "test_fixed_dist":

        # ### Also eval the full test set
        # images = torch.Tensor(images[i_test]).to(device)
        # poses = torch.Tensor(poses[i_test]).to(device)
        # i_test = i_test - i_test[0]            

        # mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, None, None, poses, H, W, K, lpips_alex, args, \
        #     render_kwargs_test, with_test_time_optimization=False)

        # write_images_with_metrics(images_test, mean_metrics_test, far, args, with_test_time_optimization=False)   

        ###### Eval fixed dist ######
        all_test_dist = [0.25, 0.5, 0.75, 1.0]

        ### This is for the blender hemisphere experiments
        near_planes = [1e-4, 0.5, 1.0, 2.0]

        for i in range(len(all_test_dist)):
            test_dist = all_test_dist[i]
            curr_near = near_planes[i]
            print("Eval " + str(test_dist))

            bds_dict = {
                'near' : curr_near,
                'far' : far,
            }
            render_kwargs_test.update(bds_dict)

            ### After training, eval with fixed dist data
            torch.cuda.empty_cache()
            scene_data_dir = os.path.join(args.eval_data_dir, args.eval_scene_id)

            images, poses, render_poses, hwf, i_split = load_scene_blender_fixed_dist_new(scene_data_dir, half_res=args.half_res, train_dist=1.0, test_dist=test_dist)

            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]


            print('Loaded blender fixed dist', images.shape, hwf, scene_data_dir)
            i_train, i_val, i_test = i_split

            # Cast intrinsics to right types
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]

            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

            with_test_time_optimization = False

            images = torch.Tensor(images[i_test]).to(device)
            poses = torch.Tensor(poses[i_test]).to(device)
            i_test = i_test - i_test[0]

            mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, None, None, poses, H, W, K, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=False)
            write_images_with_metrics_testdist(images_test, mean_metrics_test, far, args, test_dist, with_test_time_optimization=with_test_time_optimization)        


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
