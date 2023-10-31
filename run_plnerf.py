'''
Mikaela Uy
mikacuy@cs.stanford.edu

PL-NeRF: novel view synthesis experiments
A piecewise linear formulation to volume rendering
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
from load_dtu import load_dtu, load_dtu2
from load_blender import load_blender_data, load_scene_blender_fixed_dist_new, load_scene_blender2

from natsort import natsorted 
from argparse import Namespace

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
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
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


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def test_images_samples(count, indices, images, depths, valid_depths, poses, H, W, K, lpips_alex, args, render_kwargs_test, \
    embedcam_fn=None, with_test_time_optimization=False):
    far = render_kwargs_test['far']

    if count is None:
        # take all images in order
        count = len(indices)
        img_i = indices
    else:
        # take random images
        if count > len(indices):
            count = len(indices)
        img_i = np.random.choice(indices, size=count, replace=False)

    rgbs_res = torch.empty(count, 3, H, W)
    rgbs0_res = torch.empty(count, 3, H, W)
    target_rgbs_res = torch.empty(count, 3, H, W)
    depths_res = torch.empty(count, 1, H, W)
    depths0_res = torch.empty(count, 1, H, W)
    target_depths_res = torch.empty(count, 1, H, W)
    target_valid_depths_res = torch.empty(count, 1, H, W, dtype=bool)
    
    mean_metrics = MeanTracker()
    mean_depth_metrics = MeanTracker() # track separately since they are not always available
    for n, img_idx in enumerate(img_i):
        print("Render image {}/{}".format(n + 1, count))

        target = images[img_idx]

        target_depth = torch.zeros((target.shape[0], target.shape[1], 1)).to(device)
        target_valid_depth = torch.zeros((target.shape[0], target.shape[1]), dtype=bool).to(device)

        pose = poses[img_idx, :3,:4]
        intrinsic = K
        
        with torch.no_grad():
            # rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 2), c2w=pose, **render_kwargs_test)
            # print(render_kwargs_test)
            rgb, _, _, extras = render(H, W, intrinsic, chunk=args.chunk, c2w=pose, **render_kwargs_test)         
            ### 

            target_hypothesis_repeated = extras['depth_map'].unsqueeze(-1).repeat(1, 1, extras["pred_hyp"].shape[-1])
            dists = torch.norm(extras["pred_hyp"].unsqueeze(-1) - target_hypothesis_repeated.unsqueeze(-1), p=2, dim=-1)

            mask = extras['depth_map'] < 4.0

            dist_masked = dists[mask, ...]

            depth_rmse = torch.mean(dists)

            if not torch.isnan(depth_rmse):
                depth_metrics = {"importance_sampling_error" : depth_rmse.item()}
                mean_depth_metrics.add(depth_metrics)


    mean_metrics = mean_depth_metrics

    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_samples_error" + "_" + str(args.N_importance))

    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'metrics_expecteddepth.txt'), 'w') as f:
        mean_metrics.print(f)


    return mean_metrics

def render_images_with_metrics(count, indices, images, depths, valid_depths, poses, H, W, K, lpips_alex, args, render_kwargs_test, \
    embedcam_fn=None, with_test_time_optimization=False):
    far = render_kwargs_test['far']

    if count is None:
        # take all images in order
        count = len(indices)
        img_i = indices
    else:
        # take random images
        if count > len(indices):
            count = len(indices)
        img_i = np.random.choice(indices, size=count, replace=False)

    rgbs_res = torch.empty(count, 3, H, W)
    rgbs0_res = torch.empty(count, 3, H, W)
    target_rgbs_res = torch.empty(count, 3, H, W)
    depths_res = torch.empty(count, 1, H, W)
    depths0_res = torch.empty(count, 1, H, W)
    target_depths_res = torch.empty(count, 1, H, W)
    target_valid_depths_res = torch.empty(count, 1, H, W, dtype=bool)
    
    mean_metrics = MeanTracker()
    mean_depth_metrics = MeanTracker() # track separately since they are not always available
    for n, img_idx in enumerate(img_i):
        print("Render image {}/{}".format(n + 1, count), end="")
        target = images[img_idx]

        if args.dataset == "scannet":
            target_depth = depths[img_idx]
            target_valid_depth = valid_depths[img_idx]
        else:
            target_depth = torch.zeros((target.shape[0], target.shape[1], 1)).to(device)
            target_valid_depth = torch.zeros((target.shape[0], target.shape[1]), dtype=bool).to(device)

        pose = poses[img_idx, :3,:4]
        intrinsic = K
        
        with torch.no_grad():
            # rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 2), c2w=pose, **render_kwargs_test)
            # print(render_kwargs_test)
            rgb, _, _, extras = render(H, W, intrinsic, chunk=args.chunk, c2w=pose, **render_kwargs_test)
            
            # compute depth rmse
            depth_rmse = compute_rmse(extras['depth_map'][target_valid_depth], target_depth[:, :, 0][target_valid_depth])
            if not torch.isnan(depth_rmse):
                depth_metrics = {"depth_rmse" : depth_rmse.item()}
                mean_depth_metrics.add(depth_metrics)
            
            # compute color metrics
            target = torch.tensor(target).to(rgb.device)
            img_loss = img2mse(rgb, target)
            psnr = mse2psnr(img_loss)
            print("PSNR: {}".format(psnr))
            rgb = torch.clamp(rgb, 0, 1)
            ssim = structural_similarity(rgb.cpu().numpy(), target.cpu().numpy(), data_range=1., channel_axis=-1)
            lpips = lpips_alex(rgb.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]
            
            # store result
            rgbs_res[n] = rgb.clamp(0., 1.).permute(2, 0, 1).cpu()
            target_rgbs_res[n] = target.permute(2, 0, 1).cpu()
            depths_res[n] = (extras['depth_map'] / far).unsqueeze(0).cpu()
            target_depths_res[n] = (target_depth[:, :, 0] / far).unsqueeze(0).cpu()
            target_valid_depths_res[n] = target_valid_depth.unsqueeze(0).cpu()
            metrics = {"img_loss" : img_loss.item(), "psnr" : psnr.item(), "ssim" : ssim, "lpips" : lpips[0, 0, 0],}
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target)
                psnr0 = mse2psnr(img_loss0)
                depths0_res[n] = (extras['depth0'] / far).unsqueeze(0).cpu()
                rgbs0_res[n] = torch.clamp(extras['rgb0'], 0, 1).permute(2, 0, 1).cpu()
                metrics.update({"img_loss0" : img_loss0.item(), "psnr0" : psnr0.item()})
            mean_metrics.add(metrics)
    
    res = { "rgbs" :  rgbs_res, "target_rgbs" : target_rgbs_res, "depths" : depths_res, "target_depths" : target_depths_res, \
        "target_valid_depths" : target_valid_depths_res}
    if 'rgb0' in extras:
        res.update({"rgbs0" : rgbs0_res, "depths0" : depths0_res,})
    all_mean_metrics = MeanTracker()
    all_mean_metrics.add({**mean_metrics.as_dict(), **mean_depth_metrics.as_dict()})
    return all_mean_metrics, res

def write_images_with_metrics(images, mean_metrics, far, args, with_test_time_optimization=False, test_samples=False):
    
    if not test_samples:
        result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_" + str(args.mode)+ "_" + str(args.N_samples) + "_" + str(args.N_importance) + ("with_optimization_" if with_test_time_optimization else "") + args.scene_id)
    else:
        result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_samples" + str(args.mode)+ "_" + str(args.N_samples) + "_" + str(args.N_importance) + ("with_optimization_" if with_test_time_optimization else "") + str(args.N_samples) + "_" + str(args.N_importance) + args.scene_id)

    os.makedirs(result_dir, exist_ok=True)
    for n, (rgb, depth, gt_rgb) in enumerate(zip(images["rgbs"].permute(0, 2, 3, 1).cpu().numpy(), \
            images["depths"].permute(0, 2, 3, 1).cpu().numpy(), images["target_rgbs"].permute(0, 2, 3, 1).cpu().numpy())):

        # write rgb
        cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".png"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))

        cv2.imwrite(os.path.join(result_dir, str(n) + "_gt" + ".png"), cv2.cvtColor(to8b(gt_rgb), cv2.COLOR_RGB2BGR))

        # write depth
        cv2.imwrite(os.path.join(result_dir, str(n) + "_d" + ".png"), to16b(depth))

    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        mean_metrics.print(f)
    mean_metrics.print()

def write_images_with_metrics_testdist(images, mean_metrics, far, args, test_dist, with_test_time_optimization=False, test_samples=False):
    
    if not test_samples:
        result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_dist" + str(test_dist) + "_" + ("with_optimization_" if with_test_time_optimization else "") + args.scene_id)
    else:
        result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_samples_dist"  + str(test_dist) + "_" + ("with_optimization_" if with_test_time_optimization else "") + str(args.N_samples) + "_" + str(args.N_importance) + args.scene_id)

    # if not test_samples:
    #     result_dir = os.path.join(args.ckpt_dir, args.expname, "train_images_" + ("with_optimization_" if with_test_time_optimization else "") + args.scene_id)
    # else:
    #     result_dir = os.path.join(args.ckpt_dir, args.expname, "train_images_samples" + ("with_optimization_" if with_test_time_optimization else "") + str(args.N_samples) + "_" + str(args.N_importance) + args.scene_id)

    os.makedirs(result_dir, exist_ok=True)
    for n, (rgb, depth, gt_rgb) in enumerate(zip(images["rgbs"].permute(0, 2, 3, 1).cpu().numpy(), \
            images["depths"].permute(0, 2, 3, 1).cpu().numpy(), images["target_rgbs"].permute(0, 2, 3, 1).cpu().numpy())):

        # write rgb
        # cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".jpg"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".png"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))

        cv2.imwrite(os.path.join(result_dir, str(n) + "_gt" + ".png"), cv2.cvtColor(to8b(gt_rgb), cv2.COLOR_RGB2BGR))

        # write depth
        cv2.imwrite(os.path.join(result_dir, str(n) + "_d" + ".png"), to16b(depth))

    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        mean_metrics.print(f)
    mean_metrics.print()

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
    optimizer_coarse = torch.optim.Adam(params=coarse_grad_vars, lr=args.coarse_lrate, betas=(0.9, 0.999))

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
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

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
        'color_mode': args.color_mode     
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}

    render_kwargs_test['perturb'] = True

    render_kwargs_test['raw_noise_std'] = 0.

    # return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_coarse
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
        mode = "constant"

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

    # ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'pred_hyp' : pred_depth_hyp}
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
    parser.add_argument("--coarse_lrate", type=float, default=5e-4, 
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
    parser.add_argument("--num_iterations", type=int, default=500000, 
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

    parser.add_argument("--constant_init",   type=int, default=1000, 
                        help='number of iterations to use constant aggregation')    

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
    
    if args.task == "train":
        if args.expname is None:
            args.expname = "{}_{}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'), args.scene_id)
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
        os.makedirs(os.path.join(args.ckpt_dir, args.expname), exist_ok=True)
        with open(args_file, 'w') as af:
            json.dump(vars(args), af, indent=4)

    else:
        if args.expname is None:
            print("Error: Specify experiment name for test or video")
            exit()
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

    # Create log dir and copy the config file
    if args.config is not None:
        f = os.path.join(args.ckpt_dir, args.expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_coarse = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)


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

    if args.task == "train":
        print("Begin training.")
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        tb = SummaryWriter(log_dir=os.path.join("runs", args.ckpt_dir, args.expname))

        # Prepare raybatch tensor if batching random rays
        N_rand = args.N_rand
        use_batching = not args.no_batching
        if use_batching:
            # For random ray batching
            print('get rays')
            rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
            print('done, concats')
            rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
            rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)
            print('shuffle rays')
            np.random.shuffle(rays_rgb)

            print('done')
            i_batch = 0

        # Move training data to GPU
        if use_batching:
            images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)
        if use_batching:
            rays_rgb = torch.Tensor(rays_rgb).to(device)


        N_iters = args.num_iterations + 1
        print('Begin')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        # Summary writers
        # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
        
        start = start + 1
        time0 = time.time()
        for i in trange(start, N_iters):

            # Sample random ray batch
            if use_batching:
                # Random over all images
                batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
                batch = torch.transpose(batch, 0, 1)
                batch_rays, target_s = batch[:2], batch[2]

                i_batch += N_rand
                if i_batch >= rays_rgb.shape[0]:
                    print("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0])
                    rays_rgb = rays_rgb[rand_idx]
                    i_batch = 0

            else:
                # Random from one image
                img_i = np.random.choice(i_train)
                target = images[img_i]
                target = torch.Tensor(target).to(device)
                pose = poses[img_i, :3,:4]

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                    if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        if i == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True, constant_init = i < args.constant_init,
                                                    **render_kwargs_train)

            optimizer.zero_grad()
            optimizer_coarse.zero_grad()

            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()

            optimizer.step()
            optimizer_coarse.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            new_lrate_coarse = args.coarse_lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer_coarse.param_groups:
                param_group['lr'] = new_lrate
            ################################                


            dt = time.time()-time0
            # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
            #####           end            #####

            # Rest is logging
            if i%args.i_weights==0:
                path = os.path.join(args.ckpt_dir, args.expname, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

            
            if i%args.i_img==0:
                # visualize 2 train images
                _, images_train = render_images_with_metrics(2, i_train, images, None, None, \
                    poses, H, W, K, lpips_alex, args, render_kwargs_test, embedcam_fn=None)
                tb.add_image('train_image',  torch.cat((
                    torchvision.utils.make_grid(images_train["rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_train["target_rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_train["depths"], nrow=1), \
                    torchvision.utils.make_grid(images_train["target_depths"], nrow=1)), 2), i)
                # compute validation metrics and visualize 8 validation images
                # mean_metrics_val, images_val = render_images_with_metrics(None, i_test, images, None, None, poses, H, W, K, lpips_alex, args, render_kwargs_test, with_test_time_optimization=False)
                mean_metrics_val, images_val = render_images_with_metrics(20, i_test, images, None, None, poses, H, W, K, lpips_alex, args, render_kwargs_test, with_test_time_optimization=False)
                tb.add_scalars('mse', {'val': mean_metrics_val.get("img_loss")}, i)
                tb.add_scalars('psnr', {'val': mean_metrics_val.get("psnr")}, i)
                tb.add_scalar('ssim', mean_metrics_val.get("ssim"), i)
                tb.add_scalar('lpips', mean_metrics_val.get("lpips"), i)
                if mean_metrics_val.has("depth_rmse"):
                    tb.add_scalar('depth_rmse', mean_metrics_val.get("depth_rmse"), i)
                if 'rgbs0' in images_val:
                    tb.add_scalars('mse0', {'val': mean_metrics_val.get("img_loss0")}, i)
                    tb.add_scalars('psnr0', {'val': mean_metrics_val.get("psnr0")}, i)
                if 'rgbs0' in images_val:
                    tb.add_image('val_image',  torch.cat((
                        torchvision.utils.make_grid(images_val["rgbs"], nrow=1), \
                        torchvision.utils.make_grid(images_val["rgbs0"], nrow=1), \
                        torchvision.utils.make_grid(images_val["target_rgbs"], nrow=1), \
                        torchvision.utils.make_grid(images_val["depths"], nrow=1), \
                        torchvision.utils.make_grid(images_val["depths0"], nrow=1), \
                        torchvision.utils.make_grid(images_val["target_depths"], nrow=1)), 2), i)
                else:
                    tb.add_image('val_image',  torch.cat((
                        torchvision.utils.make_grid(images_val["rgbs"], nrow=1), \
                        torchvision.utils.make_grid(images_val["target_rgbs"], nrow=1), \
                        torchvision.utils.make_grid(images_val["depths"], nrow=1), \
                        torchvision.utils.make_grid(images_val["target_depths"], nrow=1)), 2), i)


            # if i%args.i_video==0 and i > 0:
            #     # Turn on testing mode
            #     with torch.no_grad():
            #         rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            #     print('Done, saving', rgbs.shape, disps.shape)
            #     moviebase = os.path.join(args.ckpt_dir, args.expname, '{}_spiral_{:06d}_'.format(args.expname, i))
            #     imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            #     imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)


            # if i%args.i_testset==0 and i > 0:
            #     testsavedir = os.path.join(args.ckpt_dir, args.expname, 'testset_{:06d}'.format(i))
            #     os.makedirs(testsavedir, exist_ok=True)
            #     print('test poses shape', poses[i_test].shape)
            #     with torch.no_grad():
            #         render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            #     print('Saved test set')

        
            if i%args.i_print==0:
                tb.add_scalars('mse', {'train': img_loss.item()}, i)
                tb.add_scalars('psnr', {'train': psnr.item()}, i)
                if 'rgb0' in extras:
                    tb.add_scalars('mse0', {'train': img_loss0.item()}, i)
                    tb.add_scalars('psnr0', {'train': psnr0.item()}, i)
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")


            global_step += 1


        dt = time.time()-time0
        print(f"Total time: {dt} seconds.")
        exit()

        ### Test after training
        if args.dataset == "llff":
            images = torch.Tensor(images).to(device)
            poses = torch.Tensor(poses).to(device)
            i_test = i_test 

        else:
            images = torch.Tensor(images[i_test]).to(device)
            poses = torch.Tensor(poses[i_test]).to(device)
            i_test = i_test - i_test[0]  

        mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, None, None, poses, H, W, K, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=False)

        write_images_with_metrics(images_test, mean_metrics_test, far, args, with_test_time_optimization=False)


        ###### Eval fixed dist ######
        all_test_dist = [0.25, 0.5, 0.75, 1.0]
        near_planes = [1e-4, 0.5, 0.5, 0.5]

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


    elif args.task == "test":

        if args.dataset == "llff":
            images = torch.Tensor(images).to(device)
            poses = torch.Tensor(poses).to(device)
            i_test = i_test 

        else:
            images = torch.Tensor(images[i_test]).to(device)
            poses = torch.Tensor(poses[i_test]).to(device)
            i_test = i_test - i_test[0]            

        mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, None, None, poses, H, W, K, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=False)

        if args.dataset == "blender_fixeddist":
            write_images_with_metrics_testdist(images_test, mean_metrics_test, far, args, args.test_dist, with_test_time_optimization=False)

        else:
            write_images_with_metrics(images_test, mean_metrics_test, far, args,  with_test_time_optimization=False)

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

    elif args.task =="test_samples_error":

        with_test_time_optimization = False

        images = torch.Tensor(images[i_test]).to(device)

        poses = torch.Tensor(poses[i_test]).to(device)

        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

        i_test = i_test - i_test[0]
        mean_metrics_test = test_images_samples(None, i_test, images, None, None, poses, H, W, K, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=with_test_time_optimization)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
