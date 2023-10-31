'''
Mikaela Uy
mikacuy@cs.stanford.edu

PL-NeRF: depth supervised experiments
'''
import os
import shutil
import subprocess
import math
import time
import datetime
from argparse import Namespace

import configargparse
from skimage.metrics import structural_similarity
from lpips import LPIPS
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from model import NeRF, get_embedder, get_rays, sample_pdf, img2mse, mse2psnr, to8b, \
    select_coordinates, to16b, sample_pdf_reformulation, compute_space_carving_loss, get_space_carving_idx, \
    sample_pdf_return_u, sample_pdf_joint_return_u, sample_pdf_reformulation_joint, sample_pdf_reformulation_return_u, sample_pdf_reformulation_joint_return_u
from data import load_scene_blender, load_scene_blender_depth, load_scene_blender2_depth
from train_utils import MeanTracker, update_learning_rate
from metric import compute_rmse

import imageio
from natsort import natsorted 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
DEBUG = False

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, embedded_cam, fn, embed_fn, embeddirs_fn, bb_center, bb_scale, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    inputs_flat = (inputs_flat - bb_center) * bb_scale ### --> does this make sense for inward facing cameras?

    embedded = embed_fn(inputs_flat) # samples * rays, multires * 2 * 3 + 3

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs, embedded_cam.unsqueeze(0).expand(embedded_dirs.shape[0], embedded_cam.shape[0])], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, use_viewdirs=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], use_viewdirs, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, intrinsic, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., with_5_9=False, use_viewdirs=False, c2w_staticcam=None, 
                  rays_depth=None, **kwargs):
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
      with_5_9: render with aspect ratio 5.33:9 (one third of 16:9)
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
        rays_o, rays_d = get_rays(H, W, intrinsic, c2w)
        if with_5_9:
            W_before = W
            W = int(H / 9. * 16. / 3.)
            if W % 2 != 0:
                W = W - 1
            start = (W_before - W) // 2
            rays_o = rays_o[:, start:start + W, :]
            rays_d = rays_d[:, start:start + W, :]
    elif rays.shape[0] == 2:
        # use provided ray batch
        rays_o, rays_d = rays
    else:
        rays_o, rays_d, rays_depth = rays
    
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, intrinsic, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    if rays_depth is not None:
        rays_depth = torch.reshape(rays_depth, [-1,3]).float()
        rays = torch.cat([rays, rays_depth], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, use_viewdirs, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_hyp(H, W, intrinsic, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., with_5_9=False, use_viewdirs=False, c2w_staticcam=None, 
                  rays_depth=None, **kwargs):
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
      with_5_9: render with aspect ratio 5.33:9 (one third of 16:9)
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
        rays_o, rays_d = get_rays(H, W, intrinsic, c2w)
        if with_5_9:
            W_before = W
            W = int(H / 9. * 16. / 3.)
            if W % 2 != 0:
                W = W - 1
            start = (W_before - W) // 2
            rays_o = rays_o[:, start:start + W, :]
            rays_d = rays_d[:, start:start + W, :]
    elif rays.shape[0] == 2:
        # use provided ray batch
        rays_o, rays_d = rays
    else:
        rays_o, rays_d, rays_depth = rays
    
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, intrinsic, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # print(rays_o.shape)
    # print(rays_d.shape)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    # print(near)
    # print(far)
    # print(use_viewdirs)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    if rays_depth is not None:
        rays_depth = torch.reshape(rays_depth, [-1,3]).float()
        rays = torch.cat([rays, rays_depth], -1)

    # print(rays.shape)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, use_viewdirs, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

### range from [mean-3*sd, mean+3*sd]
def precompute_depth_sampling(depth):
    depth_min = (depth[:, 0] - 3. * depth[:, 1])
    depth_max = depth[:, 0] + 3. * depth[:, 1]
    return torch.stack((depth[:, 0], depth_min, depth_max), -1)

def render_video(poses, H, W, intrinsics, filename, args, render_kwargs_test, fps=25):
    video_dir = os.path.join(args.ckpt_dir, args.expname, 'video_demo2_' + filename)
    video_depth_dir = os.path.join(args.ckpt_dir, args.expname, 'video_demo2_depth_' + filename)
    video_depth_colored_dir = os.path.join(args.ckpt_dir, args.expname, 'video_demo2_depth_colored' + filename)

    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    if os.path.exists(video_depth_dir):
        shutil.rmtree(video_depth_dir)
    if os.path.exists(video_depth_colored_dir):
        shutil.rmtree(video_depth_colored_dir)        

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(video_depth_dir, exist_ok=True)
    os.makedirs(video_depth_colored_dir, exist_ok=True)

    depth_scale = render_kwargs_test["far"]
    max_depth_in_video = 0

    start_idx = 0
    end_idx = len(poses)

    for img_idx in range(start_idx, end_idx):
        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]
        with torch.no_grad():
            if args.input_ch_cam > 0:
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
            # render video in 16:9 with one third rgb, one third depth and one third depth standard deviation
            rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 2), c2w=pose, with_5_9=False, **render_kwargs_test)
            rgb_cpu_numpy_8b = to8b(rgb.cpu().numpy())
            video_frame = cv2.cvtColor(rgb_cpu_numpy_8b, cv2.COLOR_RGB2BGR)

            max_depth_in_video = max(max_depth_in_video, extras['depth_map'].max())
            depth_colored_frame = cv2.applyColorMap(to8b((extras['depth_map'] / depth_scale).cpu().numpy()), cv2.COLORMAP_TURBO)
            depth = (extras['depth_map']).cpu().numpy()*1000.
            depth = (depth).astype(np.uint16)

            cv2.imwrite(os.path.join(video_dir, str(img_idx) + '.png'), video_frame)        
            cv2.imwrite(os.path.join(video_depth_dir, str(img_idx) + '.png'), depth)
            cv2.imwrite(os.path.join(video_depth_colored_dir, str(img_idx) + '.png'), depth_colored_frame)


    video_file = os.path.join(args.ckpt_dir, args.expname, filename + '.mp4')

    print(video_dir)
    imgs = os.listdir(video_dir)
    imgs = natsorted(imgs)
    print(imgs)

    imageio.mimsave(video_file,
                    [imageio.imread(os.path.join(video_dir, img)) for img in imgs],
                    fps=10, macro_block_size=1)
    print("Done.")


def optimize_camera_embedding(image, pose, H, W, intrinsic, args, render_kwargs_test):
    render_kwargs_test["embedded_cam"] = torch.zeros(args.input_ch_cam, requires_grad=True).to(device)
    optimizer = torch.optim.Adam(params=(render_kwargs_test["embedded_cam"],), lr=5e-1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)
    half_W = W
    print(" - Optimize camera embedding")
    max_psnr = 0
    best_embedded_cam = torch.zeros(args.input_ch_cam).to(device)
    # make batches
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, half_W - 1, half_W), indexing='ij'), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2]).long()
    assert(coords[:, 1].max() < half_W)
    batches = create_random_subsets(range(len(coords)), 2 * args.N_rand, device=device)
    # make rays
    rays_o, rays_d = get_rays(H, half_W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    start_time = time.time()
    for i in range(100):
        sum_img_loss = torch.zeros(1)
        optimizer.zero_grad()
        for b in batches:
            curr_coords = coords[b]
            curr_rays_o = rays_o[curr_coords[:, 0], curr_coords[:, 1]]  # (N_rand, 3)
            curr_rays_d = rays_d[curr_coords[:, 0], curr_coords[:, 1]]  # (N_rand, 3)
            target_s = image[curr_coords[:, 0], curr_coords[:, 1]]
            batch_rays = torch.stack([curr_rays_o, curr_rays_d], 0)
            rgb, _, _, _ = render(H, half_W, None, chunk=args.chunk, rays=batch_rays, verbose=i < 10, **render_kwargs_test)
            img_loss = img2mse(rgb, target_s)
            img_loss.backward()
            sum_img_loss += img_loss
        optimizer.step()
        psnr = mse2psnr(sum_img_loss / len(batches))
        lr_scheduler.step(psnr)
        if psnr > max_psnr:
            max_psnr = psnr
            best_embedded_cam = render_kwargs_test["embedded_cam"].detach().clone()
            print("Step {}: PSNR: {} ({:.2f}min)".format(i, psnr, (time.time() - start_time) / 60))
    render_kwargs_test["embedded_cam"] = best_embedded_cam

def test_images_samples(count, indices, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test, \
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

        target_depth = depths[img_idx]
        target_valid_depth = valid_depths[img_idx]

        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]

        if args.input_ch_cam > 0:
            if embedcam_fn is None:
                # use zero embedding at test time or optimize for the latent code
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
                if with_test_time_optimization:
                    optimize_camera_embedding(target, pose, H, W, intrinsic, args, render_kwargs_test)
                    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_latent_codes_" + args.scene_id)
                    os.makedirs(result_dir, exist_ok=True)
                    np.savetxt(os.path.join(result_dir, str(img_idx) + ".txt"), render_kwargs_test["embedded_cam"].cpu().numpy())
            else:
                render_kwargs_test["embedded_cam"] = embedcam_fn(torch.tensor(img_idx, device=device))
        
        with torch.no_grad():
            rgb, _, _, extras = render(H, W, intrinsic, chunk=args.chunk, c2w=pose, **render_kwargs_test)
            

            target_hypothesis_repeated = extras['depth_map'].unsqueeze(-1).repeat(1, 1, extras["pred_hyp"].shape[-1])

            dists = torch.norm(extras["pred_hyp"].unsqueeze(-1) - target_hypothesis_repeated.unsqueeze(-1), p=2, dim=-1)

            depth_rmse = torch.mean(dists, axis=-1)
            depth_rmse = depth_rmse[target_valid_depth]

            depth_rmse = torch.mean(depth_rmse)

            if not torch.isnan(depth_rmse):
                depth_metrics = {"importance_sampling_error" : depth_rmse.item()}
                mean_depth_metrics.add(depth_metrics)

    mean_metrics = mean_depth_metrics

    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_predicted_samples_error" + "_" + str(args.N_importance))

    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'metrics_depth_samples.txt'), 'w') as f:
        mean_metrics.print(f)

    return mean_metrics


def render_images_with_metrics(count, indices, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test, \
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

        target_depth = depths[img_idx]
        target_valid_depth = valid_depths[img_idx]

        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]

        if args.input_ch_cam > 0:
            if embedcam_fn is None:
                # use zero embedding at test time or optimize for the latent code
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
                if with_test_time_optimization:
                    optimize_camera_embedding(target, pose, H, W, intrinsic, args, render_kwargs_test)
                    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_latent_codes_" + args.scene_id)
                    os.makedirs(result_dir, exist_ok=True)
                    np.savetxt(os.path.join(result_dir, str(img_idx) + ".txt"), render_kwargs_test["embedded_cam"].cpu().numpy())
            else:
                render_kwargs_test["embedded_cam"] = embedcam_fn(torch.tensor(img_idx, device=device))
        
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
        result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_" + ("with_optimization_" if with_test_time_optimization else "") + args.scene_id)
    else:
        result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_samples" + ("with_optimization_" if with_test_time_optimization else "") + str(args.N_samples) + "_" + str(args.N_importance) + args.scene_id)

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

def load_checkpoint(args):
    path = os.path.join(args.ckpt_dir, args.expname)
    ckpts = [os.path.join(path, f) for f in sorted(os.listdir(path)) if '000.tar' in f]
    print('Found ckpts', ckpts)
    ckpt = None
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
    return ckpt

def create_nerf(args, scene_render_params):
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
                 input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)


    model = nn.DataParallel(model).to(device)
    grad_vars = list(model.parameters())

    grad_vars = []
    ch_grad_vars = []

    grad_names = []
    ch_grad_names = []


    for name, param in model.named_parameters():
        grad_vars.append(param)
        grad_names.append(name)


    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)         

        model_fine = nn.DataParallel(model_fine).to(device)
        # grad_vars += list(model_fine.parameters())

        for name, param in model_fine.named_parameters():
            grad_vars.append(param)
            grad_names.append(name)


    network_query_fn = lambda inputs, viewdirs, embedded_cam, network_fn : run_network(inputs, viewdirs, embedded_cam, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                bb_center=args.bb_center,
                                                                bb_scale=args.bb_scale,
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0

    ##########################

    # Load checkpoints
    ckpt = load_checkpoint(args)
    if ckpt is not None:
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################
    embedded_cam = torch.tensor((), device=device)
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'embedded_cam' : embedded_cam,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
        'white_bkgd' : args.white_bkgd,
        'mode' : args.mode,
        'color_mode': args.color_mode
    }
    render_kwargs_train.update(scene_render_params)

    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}

    render_kwargs_test['perturb'] = True

    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def compute_weights(raw, z_vals, rays_d, noise=0.):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10, device=device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    return weights

### Our reformulation to piecewise linear
def compute_weights_piecewise_linear(raw, z_vals, near, far, rays_d, noise=0., return_tau=False):
    raw2expr = lambda raw, dists: torch.exp(-raw*dists)

    ### Concat
    z_vals = torch.cat([near, z_vals, far], -1)

    dists = z_vals[...,1:] - z_vals[...,:-1]
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

    factor = (1 - expr)

    weights = factor * T[:, :-1] # [N_rays, N_samples+1]

    '''
    We will need to return tau and T for backprop later
    '''
    ### Remember to remove the last value of T(far) is not used
    ### tau(far) is also not used

    if return_tau:
        return weights, tau, T
    else:
        return weights    

def raw2depth(raw, z_vals, near, far, rays_d, mode):

    if mode == "linear":
        weights = compute_weights_piecewise_linear(raw, z_vals, near, far, rays_d)
    elif mode == "constant":
        weights = compute_weights(raw, z_vals, rays_d)

    depth = torch.sum(weights * z_vals, -1)
    std = (((z_vals - depth.unsqueeze(-1)).pow(2) * weights).sum(-1)).sqrt()
    return depth, std


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
     
        ### Skip the first bin weights [near, s_0]
        if color_mode == "midpoint":

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

def perturb_z_vals(z_vals, pytest):
    # get intervals between samples
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand_like(z_vals)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        t_rand = np.random.rand(*list(z_vals.shape))
        t_rand = torch.Tensor(t_rand)

    z_vals = lower + (upper - lower) * t_rand
    return z_vals

def render_rays(ray_batch,
                use_viewdirs,
                network_fn,
                network_query_fn,
                N_samples,
                mode,
                color_mode,
                precomputed_z_samples=None,
                embedded_cam=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                white_bkgd=False,                
                is_joint=False,
                cached_u= None,
                scale_sample_gradient=False,
                quad_solution_v2=False,
                zero_tol = 1e-4,
                epsilon = 1e-3,
                farcolorfix = False):
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
    viewdirs = None
    depth_range = None
    if use_viewdirs:
        viewdirs = ray_batch[:,8:11]
        if ray_batch.shape[-1] > 11:
            depth_range = ray_batch[:,11:14]
    else:
        if ray_batch.shape[-1] > 8:
            depth_range = ray_batch[:,8:11]
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    t_vals = torch.linspace(0., 1., steps=N_samples)
    
    # sample and render rays for nerf
    if not lindisp:
        # print("Not lindisp")
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # print("Lindisp")
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    if perturb > 0.:
        # print("Perturb.")
        z_vals = perturb_z_vals(z_vals, pytest)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, tau, T = raw2outputs(raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std, pytest=pytest, white_bkgd=white_bkgd, farcolorfix=farcolorfix)

    if N_importance == 0:
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if not is_joint:
            if mode == "linear":
                z_samples, T_below, tau_below, bin_below, u = sample_pdf_reformulation_return_u(z_vals, weights, tau, T, near, far, N_samples, det=(perturb==0.), pytest=pytest, quad_solution_v2=quad_solution_v2)
            elif mode == "constant":
                z_samples, u = sample_pdf_return_u(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)
        else:
            if mode == "linear":
                z_samples, T_below, tau_below, bin_below, u = sample_pdf_reformulation_joint_return_u(z_vals, weights, tau, T, near, far, N_samples, det=(perturb==0.), pytest=pytest, quad_solution_v2=quad_solution_v2)
            elif mode == "constant":
                z_samples, u = sample_pdf_joint_return_u(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)

        pred_depth_hyp = z_samples

    elif N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, z_vals_0, weights_0 = rgb_map, disp_map, acc_map, depth_map, z_vals, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        
        if mode == "linear":
            z_samples, T_below, tau_below, bin_below = sample_pdf_reformulation(z_vals, weights, tau, T, near, far, N_importance, det=(perturb==0.), pytest=pytest, quad_solution_v2=quad_solution_v2)
        elif mode == "constant":
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)

        z_samples = z_samples.detach()

        ######## Clamping in quad solution should have fixed this
        z_samples = torch.clamp(z_samples, near, far)
        ########

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine

        raw = network_query_fn(pts, viewdirs, embedded_cam, run_fn)
        
        rgb_map, disp_map, acc_map, weights, depth_map, tau, T = raw2outputs(raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std, pytest=pytest, white_bkgd=white_bkgd)

        ### P_depth from fine network
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if not is_joint:
            if mode == "linear":
                z_samples, T_below, tau_below, bin_below, u = sample_pdf_reformulation_return_u(z_vals, weights, tau, T, near, far, N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u, quad_solution_v2=quad_solution_v2)
            elif mode == "constant":
                z_samples, u = sample_pdf_return_u(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u)
        else:
            if mode == "linear":
                z_samples, T_below, tau_below, bin_below, u = sample_pdf_reformulation_joint_return_u(z_vals, weights, tau, T, near, far, N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u, quad_solution_v2=quad_solution_v2)
            elif mode == "constant":
                z_samples, u = sample_pdf_joint_return_u(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u)

        pred_depth_hyp = z_samples


    if mode == "linear":
        weights = weights[..., 1:]

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'z_vals' : z_vals, 'weights' : weights, 'pred_hyp' : pred_depth_hyp,\
    'u':u}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_vals0'] = z_vals_0
        ret['weights0'] = weights_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        # ret['pred_hyp'] = pred_depth_hyp

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def get_ray_batch_from_one_image_hypothesis_idx(H, W, img_i, images, depths, valid_depths, poses, intrinsics, all_hypothesis, args, space_carving_idx=None, cached_u=None, gt_valid_depths=None):
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
    # img_i = np.random.choice(i_train)
    
    target = images[img_i]
    target_depth = depths[img_i]
    target_valid_depth = valid_depths[img_i]
    pose = poses[img_i]
    intrinsic = intrinsics[img_i, :]

    target_hypothesis = all_hypothesis[img_i]

    rays_o, rays_d = get_rays(H, W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    select_coords = select_coordinates(coords, args.N_rand)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_d = target_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1) or (N_rand, 2)
    target_vd = target_valid_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
    target_h = target_hypothesis[:, select_coords[:, 0], select_coords[:, 1]]

    if space_carving_idx is not None:
        target_hypothesis  = target_hypothesis.repeat(1, 1, 1, space_carving_idx.shape[-1])

        curr_space_carving_idx = space_carving_idx[img_i, select_coords[:, 0], select_coords[:, 1]]

        target_h_rays = target_hypothesis[ :, select_coords[:, 0], select_coords[:, 1]]

        target_h = torch.gather(target_h_rays, 1, curr_space_carving_idx.unsqueeze(0).long())


    if cached_u is not None:
        curr_cached_u = cached_u[img_i, select_coords[:, 0], select_coords[:, 1]]
    else:
        curr_cached_u = None

    space_carving_mask = gt_valid_depths[img_i].squeeze()
    space_carving_mask = space_carving_mask[select_coords[:, 0], select_coords[:, 1]]

    batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)

    return batch_rays, target_s, target_d, target_vd, img_i, target_h, space_carving_mask, curr_cached_u

def train_nerf(images, depths, valid_depths, poses, intrinsics, i_split, args, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths, all_depth_hypothesis, is_init_scales=False, scales_init=None, shifts_init=None):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    tb = SummaryWriter(log_dir=os.path.join("runs", args.ckpt_dir, args.expname))
    near, far = scene_sample_params['near'], scene_sample_params['far']
    H, W = images.shape[1:3]
    i_train, i_val, i_test, i_video = i_split
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)
    print('TEST views are', i_test)

    # use ground truth depth for validation and test if available
    if gt_depths is not None:
        depths[i_test] = gt_depths[i_test]
        valid_depths[i_test] = gt_valid_depths[i_test]
        depths[i_val] = gt_depths[i_val]
        valid_depths[i_val] = gt_valid_depths[i_val]
    i_relevant_for_training = np.concatenate((i_train, i_val), 0)
    if len(i_test) == 0:
        print("Error: There is no test set")
        exit()
    if len(i_val) == 0:
        print("Warning: There is no validation set, test set is used instead")
        i_val = i_test
        i_relevant_for_training = np.concatenate((i_relevant_for_training, i_val), 0)

    # keep test data on cpu until needed
    test_images = images[i_test]

    if depths is not None:
        test_depths = depths[i_test]
        test_valid_depths = valid_depths[i_test]
    else:
        test_depths = torch.zeros((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)).to(device)
        test_valid_depths = torch.zeros((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)).to(device)        
    
    test_poses = poses[i_test]
    test_intrinsics = intrinsics[i_test]
    i_test = i_test - i_test[0]

    # move training data to gpu
    images = torch.Tensor(images[i_relevant_for_training]).to(device)
    
    if depths is not None:
        depths = torch.Tensor(depths[i_relevant_for_training]).to(device)
        valid_depths = torch.Tensor(valid_depths[i_relevant_for_training]).bool().to(device)
    else:
        depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], 1)).to(device)
        valid_depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], 1)).to(device)

    poses = torch.Tensor(poses[i_relevant_for_training]).to(device)
    intrinsics = torch.Tensor(intrinsics[i_relevant_for_training]).to(device)
    all_depth_hypothesis = torch.Tensor(all_depth_hypothesis).to(device)

    if gt_depths is not None:
        # complete and check depth
        gt_depths_train = torch.Tensor(gt_depths[i_train]).to(device) # only used to evaluate error of completed depth
        gt_valid_depths_train = torch.Tensor(gt_valid_depths[i_train]).bool().to(device) # only used to evaluate error of completed depth


    ###########################################################################
    #### Use GT depth for space carving --> overriding all_depth_hypothesis ###
    gt_depths_train = gt_depths_train.unsqueeze(1)
    gt_valid_depths_train = gt_valid_depths_train.unsqueeze(1)
    ###########################################################################

    ##### Initialize depth scale and shift
    DEPTH_SCALES = torch.autograd.Variable(torch.ones((images.shape[0], 1), dtype=torch.float, device=images.device)*args.scale_init, requires_grad=True)
    DEPTH_SHIFTS = torch.autograd.Variable(torch.ones((images.shape[0], 1), dtype=torch.float, device=images.device)*args.shift_init, requires_grad=True)      

    print(DEPTH_SCALES)
    print()
    print(DEPTH_SHIFTS)
    print()
    print(DEPTH_SCALES.shape)
    print(DEPTH_SHIFTS.shape)

    optimizer_ss = torch.optim.Adam(params=(DEPTH_SCALES, DEPTH_SHIFTS,), lr=args.scaleshift_lr)
    
    print("Done with scale and shift init.")
    ################################

    # create nerf model
    render_kwargs_train, render_kwargs_test, start, nerf_grad_vars, optimizer = create_nerf(args, scene_sample_params)
    
    # create camera embedding function
    embedcam_fn = None
    if args.input_ch_cam > 0:
        embedcam_fn = torch.nn.Embedding(len(i_train), args.input_ch_cam)

    # optimize nerf
    print('Begin')
    N_iters = args.num_iterations + 1
    global_step = start
    start = start + 1


    for i in trange(start, N_iters):

        # update learning rate
        if i > args.start_decay_lrate and i <= args.end_decay_lrate:
            portion = (i - args.start_decay_lrate) / (args.end_decay_lrate - args.start_decay_lrate)
            decay_rate = 0.1
            new_lrate = args.lrate * (decay_rate ** portion)
            update_learning_rate(optimizer, new_lrate)

        img_i = np.random.choice(i_train)

        curr_scale = DEPTH_SCALES[img_i]
        curr_shift = DEPTH_SHIFTS[img_i]

        ## Scale and shift
        batch_rays, target_s, target_d, target_vd, img_i, target_h, space_carving_mask, curr_cached_u = get_ray_batch_from_one_image_hypothesis_idx(H, W, img_i, images, depths, valid_depths, poses, \
            intrinsics, gt_depths_train, args, None, None, gt_valid_depths_train)

        target_h = target_h*curr_scale + curr_shift        

        if args.input_ch_cam > 0:
            render_kwargs_train['embedded_cam'] = embedcam_fn[img_i]

        target_d = target_d.squeeze(-1)

        # render
        ## Feed cached quantiles into the renderer
        render_kwargs_train["cached_u"] = curr_cached_u

        rgb, _, _, extras = render_hyp(H, W, None, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,  is_joint=args.is_joint, \
            quad_solution_v2=args.quad_solution_v2, **render_kwargs_train)

        # compute loss and optimize
        optimizer.zero_grad()
        optimizer_ss.zero_grad()
        img_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(img_loss)
        
        loss = img_loss

        if args.space_carving_weight>0. and i>args.warm_start_nerf:
            
            space_carving_loss = compute_space_carving_loss(extras["pred_hyp"], target_h, is_joint=args.is_joint, norm_p=args.norm_p, threshold=args.space_carving_threshold, mask=space_carving_mask)
            
            loss = loss + args.space_carving_weight * space_carving_loss
        else:
            space_carving_loss = torch.mean(torch.zeros([target_h.shape[0]]).to(target_h.device))

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            psnr0 = mse2psnr(img_loss0)
            loss = loss + img_loss0

        loss.backward()
        nn.utils.clip_grad_value_(nerf_grad_vars, 0.1)
        optimizer.step()

        ### Don't optimize scale shift for the last 100k epochs, check whether the appearance will crisp
        if i < args.freeze_ss:
            optimizer_ss.step()

        ### Update camera embeddings
        if args.input_ch_cam > 0 and args.opt_ch_cam:
            optimizer_latent.step() 

        # write logs
        if i%args.i_weights==0:
            path = os.path.join(args.ckpt_dir, args.expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),}
            if render_kwargs_train['network_fine'] is not None:
                save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()

            if args.input_ch_cam > 0:
                save_dict['embedded_cam'] = embedcam_fn

            save_dict['depth_shifts'] = DEPTH_SHIFTS
            save_dict['depth_scales'] = DEPTH_SCALES

            torch.save(save_dict, path)
            print('Saved checkpoints at', path)
        
        if i%args.i_print==0:
            tb.add_scalars('mse', {'train': img_loss.item()}, i)

            if args.space_carving_weight > 0.:
                tb.add_scalars('space_carving_loss', {'train': space_carving_loss.item()}, i)

            tb.add_scalars('psnr', {'train': psnr.item()}, i)
            if 'rgb0' in extras:
                tb.add_scalars('mse0', {'train': img_loss0.item()}, i)
                tb.add_scalars('psnr0', {'train': psnr0.item()}, i)

            scale_mean = torch.mean(DEPTH_SCALES[i_train])
            shift_mean = torch.mean(DEPTH_SHIFTS[i_train])
            tb.add_scalars('depth_scale_mean', {'train': scale_mean.item()}, i)
            tb.add_scalars('depth_shift_mean', {'train': shift_mean.item()}, i) 

            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  MSE: {img_loss.item()} Space carving: {space_carving_loss.item()}")
            
        if i%args.i_img==0:
            # visualize 2 train images
            _, images_train = render_images_with_metrics(2, i_train, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test, embedcam_fn=embedcam_fn)
            tb.add_image('train_image',  torch.cat((
                torchvision.utils.make_grid(images_train["rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_train["target_rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_train["depths"], nrow=1), \
                torchvision.utils.make_grid(images_train["target_depths"], nrow=1)), 2), i)
            # compute validation metrics and visualize 8 validation images
            mean_metrics_val, images_val = render_images_with_metrics(8, i_val, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
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

        # test at the last iteration
        if (i + 1) == N_iters:
            torch.cuda.empty_cache()
            images = torch.Tensor(test_images).to(device)
            depths = torch.Tensor(test_depths).to(device)
            valid_depths = torch.Tensor(test_valid_depths).bool().to(device)
            poses = torch.Tensor(test_poses).to(device)
            intrinsics = torch.Tensor(test_intrinsics).to(device)
            mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
            write_images_with_metrics(images_test, mean_metrics_test, far, args)
            tb.flush()

        global_step += 1


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('task', type=str, help='one out of: "train", "test", "test_with_opt", "video"')
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, default=None, 
                        help='specify the experiment, required for "test" and "video", optional for "train"')
    parser.add_argument("--dataset", type=str, default="blender2_depth", 
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
    parser.add_argument("--N_rand", type=int, default=32*32,
                        help='batch size (number of random rays per gradient step)')


    parser.add_argument('--num_iterations', type=int, default=500000, help='Number of epochs')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--start_decay_lrate", type=int, default=400000, 
                        help='start iteration for learning rate decay')
    parser.add_argument("--end_decay_lrate", type=int, default=500000, 
                        help='end iteration for learning rate decay')


    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=1024*64*4, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=256,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=9,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=0,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--lindisp", action='store_true', default=False,
                        help='sampling linearly in disparity rather than depth')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=600000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--ckpt_dir", type=str, default="",
                        help='checkpoint directory')

    # data options
    parser.add_argument("--scene_id", type=str, default="chair",
                        help='scene identifier')
    parser.add_argument("--data_dir", type=str, default="",
                        help='directory containing the scenes')

    ### Train json file --> experimenting making views sparser
    parser.add_argument("--train_jsonfile", type=str, default='transforms_train.json',
                        help='json file containing training images')

    ### For Blender dataset ###
    #### training options --> not implemented yet
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')    
    #########

    parser.add_argument("--random_seed",   type=int, default=0, 
                        help='random seed used')


    ######################
    ### For scade loss ###
    ######################
    parser.add_argument("--cimle_dir", type=str, default="dump_0826_pretrained_dd_scene0710_train/",
                        help='dump_dir name for prior depth hypotheses')
    parser.add_argument("--num_hypothesis", type=int, default=20, 
                        help='number of cimle hypothesis')
    parser.add_argument("--space_carving_weight", type=float, default=0.007,
                        help='weight of the depth loss, values <=0 do not apply depth loss')
    parser.add_argument("--warm_start_nerf", type=int, default=0, 
                        help='number of iterations to train only vanilla nerf without additional losses.')

    parser.add_argument('--scaleshift_lr', default= 0.000001, type=float)
    parser.add_argument('--scale_init', default= 1.0, type=float)
    parser.add_argument('--shift_init', default= 0.0, type=float)
    parser.add_argument("--freeze_ss", type=int, default=0, 
                            help='dont update scale/shift starting epoch X')

    ### u sampling is joint or not
    parser.add_argument('--is_joint', default= False, type=bool)

    ### Norm for space carving loss
    parser.add_argument("--norm_p", type=int, default=2, help='norm for loss')
    parser.add_argument("--space_carving_threshold", type=float, default=0.0,
                        help='threshold to not penalize the space carving loss.')
    parser.add_argument('--mask_corners', default= False, type=bool)

    parser.add_argument("--input_ch_cam", type=int, default=0,
                        help='number of channels for camera index embedding')

    parser.add_argument("--opt_ch_cam", action='store_true', default=False,
                        help='optimize camera embedding')    
    parser.add_argument('--ch_cam_lr', default= 0.0001, type=float)

    ##################
    ##################


    ### For PWL formulation
    parser.add_argument("--mode", type=str, default="constant", 
                        help='rendering aggregation mode -- whether to use piecewise constant (vanilla) or piecewise linear (reformulation)."')
    parser.add_argument("--color_mode", type=str, default="midpoint", 
                        help='rendering color aggregation mode -- whether to use left bin or midpoint."')

    parser.add_argument('--quad_solution_v2', default= True, type=bool)
    parser.add_argument("--zero_tol", type=float, default=1e-4, 
                        help='zero tol to revert to piecewise constant assumption')    
    parser.add_argument("--epsilon", type=float, default=1e-3, 
                        help='epsilon value in the increasing and decreasing cases or max(x,epsilon)')

    parser.add_argument('--set_near_plane', default= 0.5, type=float)

    parser.add_argument('--train_skip', default= 1, type=int, help='Number training images to skip for sparsity')

    return parser

def run_nerf():
    
    parser = config_parser()
    args = parser.parse_args()

    if args.task == "train":
        if args.expname is None:
            args.expname = "{}_{}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'), args.scene_id)
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
        os.makedirs(os.path.join(args.ckpt_dir, args.expname), exist_ok=True)
        with open(args_file, 'w') as af:
            json.dump(vars(args), af, indent=4)

    elif args.task == "test_samples":
        if args.expname is None:
            print("Error: Specify experiment name for test or video")
            exit()
        tmp_task = args.task
        tmp_data_dir = args.data_dir
        tmp_ckpt_dir = args.ckpt_dir
        tmp_N_samples = args.N_samples
        tmp_N_importance = args.N_importance
        tmp_mode = args.mode
        tmp_set_near_plane = args.set_near_plane

        # load nerf parameters from training
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
        with open(args_file, 'r') as af:
            args_dict = json.load(af)
        args_loaded = Namespace(**args_dict)
        # task and paths are not overwritten
        args.task = tmp_task
        args.data_dir = tmp_data_dir
        args.ckpt_dir = tmp_ckpt_dir
        args.train_jsonfile = 'transforms_train.json'

        args.N_samples = tmp_N_samples
        args.N_importance = tmp_N_importance
        args.mode = tmp_mode
        args.set_near_plane = tmp_set_near_plane

    else:
        if args.expname is None:
            print("Error: Specify experiment name for test or video")
            exit()
        tmp_task = args.task
        tmp_data_dir = args.data_dir
        tmp_ckpt_dir = args.ckpt_dir
        tmp_set_near_plane = args.set_near_plane
        tmp_mode = args.mode
        tmp_N_samples = args.N_samples
        tmp_N_importance = args.N_importance

        # load nerf parameters from training
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
        with open(args_file, 'r') as af:
            args_dict = json.load(af)
        args = Namespace(**args_dict)
        # task and paths are not overwritten
        args.task = tmp_task
        args.data_dir = tmp_data_dir
        args.ckpt_dir = tmp_ckpt_dir
        args.mode = tmp_mode
        args.train_jsonfile = 'transforms_train.json'
        args.set_near_plane = tmp_set_near_plane
        args.N_samples = tmp_N_samples
        args.N_importance = tmp_N_importance

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    scene_data_dir = os.path.join(args.data_dir, args.scene_id)

    if args.dataset == "blender_depth":
        images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths = load_scene_blender_depth(scene_data_dir, half_res=args.half_res, train_skip=args.train_skip, near_plane=args.set_near_plane)

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]  

    elif args.dataset == "blender2_depth":
        images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths = load_scene_blender2_depth(scene_data_dir, half_res=args.half_res, train_skip=args.train_skip, near_plane=args.set_near_plane)

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]  
    else:
        print("ERROR: Dataloader not implemented for dataset: "+args.dataset)
        exit()

    if args.dataset == "blender_depth" or args.dataset == "blender2_depth":
        near = args.set_near_plane
        print("Set near plane to: " + str(near))

        ### Make the depth hypothesis same as the laoded depth
        all_depth_hypothesis = depths[:, np.newaxis, :, :]

    i_train, i_val, i_test, i_video = i_split

    ### Make scale 1.
    args.bb_center = 0.0
    args.bb_scale = 1.0

    scene_sample_params = {
        'precomputed_z_samples' : None,
        'near' : near,
        'far' : far,
    }

    lpips_alex = LPIPS()

    if args.task == "train":
        train_nerf(images, depths, valid_depths, poses, intrinsics, i_split, args, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths, all_depth_hypothesis)
        exit()

    # create nerf model for testing
    _, render_kwargs_test, _, nerf_grad_vars, _ = create_nerf(args, scene_sample_params)
    for param in nerf_grad_vars:
        param.requires_grad = False

    # render test set and compute statistics
    if args.task == "test": 
        with_test_time_optimization = False
        if args.task == "test_opt":
            with_test_time_optimization = True
        images = torch.Tensor(images[i_test]).to(device)
        if gt_depths is None:
            depths = torch.Tensor(depths[i_test]).to(device)
            valid_depths = torch.Tensor(valid_depths[i_test]).bool().to(device)
        else:
            depths = torch.Tensor(gt_depths[i_test]).to(device)
            valid_depths = torch.Tensor(gt_valid_depths[i_test]).bool().to(device)
        poses = torch.Tensor(poses[i_test]).to(device)
        intrinsics = torch.Tensor(intrinsics[i_test]).to(device)
        i_test = i_test - i_test[0]
        mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=with_test_time_optimization)

        if "samples" in args.task:
            write_images_with_metrics(images_test, mean_metrics_test, far, args, with_test_time_optimization=with_test_time_optimization, test_samples=True)
        else:
            write_images_with_metrics(images_test, mean_metrics_test, far, args, with_test_time_optimization=with_test_time_optimization)
    elif args.task == "video":
        vposes = torch.Tensor(poses[i_video]).to(device)
        vintrinsics = torch.Tensor(intrinsics[i_video]).to(device)
        render_video(vposes, H, W, vintrinsics, str(0), args, render_kwargs_test)

    elif args.task =="test_samples_error":

        with_test_time_optimization = False

        images = torch.Tensor(images[i_test]).to(device)
        if gt_depths is None:
            depths = torch.Tensor(depths[i_test]).to(device)
            valid_depths = torch.Tensor(valid_depths[i_test]).bool().to(device)
        else:
            depths = torch.Tensor(gt_depths[i_test]).to(device)
            valid_depths = torch.Tensor(gt_valid_depths[i_test]).bool().to(device)
        poses = torch.Tensor(poses[i_test]).to(device)
        intrinsics = torch.Tensor(intrinsics[i_test]).to(device)
        i_test = i_test - i_test[0]
        mean_metrics_test = test_images_samples(None, i_test, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=with_test_time_optimization)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    run_nerf()
