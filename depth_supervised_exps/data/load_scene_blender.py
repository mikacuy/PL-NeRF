import os

import numpy as np
import json
import cv2

import torchvision.transforms as transforms
import imageio
import torch

#### For Pose Corrections ####
BLENDER2OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])).cpu().numpy() @ c2w.cpu().numpy() @ BLENDER2OPENCV
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])).cpu().numpy() @ c2w.cpu().numpy()
    return c2w


def read_files(rgb_file, downsample_scale=None):
    # fname = os.path.join(basedir, rgb_file)
    fname = rgb_file
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

    if downsample_scale is not None:
        img = cv2.resize(img, (int(img.shape[1]/downsample_scale), int(img.shape[0]/downsample_scale)), interpolation=cv2.INTER_LINEAR)

    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available

    return img

# def load_ground_truth_depth(depth_file, depth_scaling_factor, near, far):
#     gt_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float64)[...,0]
#     gt_depth = (gt_depth / depth_scaling_factor).astype(np.float32)

#     return gt_depth

def load_ground_truth_depth(depth_file, depth_scaling_factor, near, far):
    gt_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float64)
    gt_depth = (gt_depth / depth_scaling_factor).astype(np.float32)

    return gt_depth

def load_scene_blender(basedir, train_json = "transforms_train.json", half_res=True):
    splits = ['train', 'val', 'test', 'video']
    # splits = ['test']

    all_imgs = []

    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = 2.
                far = 6.
                camera_angle_x = float(meta['camera_angle_x'])

            imgs = []
            poses = []
            intrinsics = []

            if s=='train':
                skip = 1
            elif s =="test":
                skip = 8
            elif "video" in s:
                skip = 1
            
            for frame in meta['frames'][::skip]:
                if len(frame['file_path']) != 0 :
                    if half_res :
                        downsample = 2
                    else:
                        downsample = 1

                    img = read_files(os.path.join(basedir, frame['file_path']+".png"), downsample_scale=downsample)

                    filenames.append(frame['file_path'])
                    imgs.append(img)

                # poses.append(np.array(frame['transform_matrix'])@ BLENDER2OPENCV)
                poses.append(np.array(frame['transform_matrix']))

                H, W = img.shape[:2]
                focal = .5 * W / np.tan(.5 * camera_angle_x)                            

                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        elif s == "video":
            ### Use spherical poses
            render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

            poses = []
            intrinsics = []
            for i in range(render_poses.shape[0]):
                poses.append(render_poses[i])
                focal = .5 * W / np.tan(.5 * camera_angle_x)
                focal /= downsample                               

                H, W = img.shape[:2]
                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    
    return imgs, None, None, poses, H, W, intrinsics, near, far, i_split, None, None

def load_scene_blender2(basedir, train_json = "transforms_train.json", half_res=True):
    splits = ['train', 'val', 'test', 'video']
    # splits = ['test']

    all_imgs = []

    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, '{}_transforms.json'.format(s))):

            json_fname =  os.path.join(basedir, '{}_transforms.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = 2.
                far = 6.
                camera_angle_x = float(meta['camera_angle_x'])

            imgs = []
            poses = []
            intrinsics = []

            if s=='train':
                skip = 1
            elif s =="test":
                skip = 8
            elif "video" in s:
                skip = 1
            
            for frame in meta['frames'][::skip]:
                if len(frame['file_path']) != 0 :
                    if half_res :
                        downsample = 2
                    else:
                        downsample = 1

                    img = read_files(os.path.join(basedir, frame['file_path']+".png"), downsample_scale=downsample)

                    filenames.append(frame['file_path'])
                    imgs.append(img)

                # poses.append(np.array(frame['transform_matrix'])@ BLENDER2OPENCV)
                poses.append(np.array(frame['transform_matrix']))

                H, W = img.shape[:2]
                focal = .5 * W / np.tan(.5 * camera_angle_x)                            

                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        elif s == "video":
            ### Use spherical poses
            render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

            poses = []
            intrinsics = []
            for i in range(render_poses.shape[0]):
                poses.append(render_poses[i])
                focal = .5 * W / np.tan(.5 * camera_angle_x)
                focal /= downsample                               

                H, W = img.shape[:2]
                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    
    return imgs, None, None, poses, H, W, intrinsics, near, far, i_split, None, None


def load_scene_blender_multidist(basedir, train_json = "transforms_train.json", half_res=True, train_dist=1.0, test_dist=1.0, video_idx=0):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []

    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:

        folder = '{}_{}_nv100_dist0.5-1.5-5'.format(basedir.split("/")[-1].split("_")[0], s)

        if s == "train":
            transforms_file = 'transforms_{}.json'.format(str(train_dist))
        elif s == "val":
            transforms_file = 'transforms_{}.json'.format(str(test_dist))            
        elif s == "test":
            transforms_file = 'transforms_{}.json'.format(str(test_dist))
            # folder = 'lego_{}_nv100_dist0.5-1.5-5'.format(s)
        elif s == 'video':
            transforms_file = 'transforms_video{}.json'.format(str(video_idx))           
        else:
            ## dummy will return not exist
            transforms_file = "blah"

        if os.path.exists(os.path.join(basedir, folder, transforms_file)):

            json_fname =  os.path.join(basedir, folder, transforms_file)

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = 2.
                far = 6.
                camera_angle_x = float(meta['camera_angle_x'])

            imgs = []
            poses = []
            intrinsics = []

            if s=='train':
                skip = 1
            elif s == "val":
                skip = 8
            elif s =="test":
                skip = 1
            elif "video" in s:
                skip = 1
            
            for frame in meta['frames'][::skip]:
                if len(frame['file_path']) != 0 :
                    if half_res :
                        downsample = 2
                    else:
                        downsample = 1

                    img = read_files(os.path.join(basedir, folder, frame['file_path']+".png"), downsample_scale=downsample)

                    filenames.append(frame['file_path'])
                    imgs.append(img)

                # poses.append(np.array(frame['transform_matrix'])@ BLENDER2OPENCV)
                poses.append(np.array(frame['transform_matrix']))

                H, W = img.shape[:2]
                focal = .5 * W / np.tan(.5 * camera_angle_x)                            

                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        ## For zooming in video
        elif os.path.exists(os.path.join(basedir, transforms_file)):

            json_fname =  os.path.join(basedir, transforms_file)

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = 2.
                far = 6.
                camera_angle_x = float(meta['camera_angle_x'])

            imgs = []
            poses = []
            intrinsics = []

            if s=='train':
                skip = 1
            elif s == "val":
                skip = 8
            elif s =="test":
                skip = 1
            elif "video" in s:
                skip = 1
            
            for frame in meta['frames'][::skip]:
                if len(frame['file_path']) != 0 :
                    if half_res :
                        downsample = 2
                    else:
                        downsample = 1

                    img = read_files(os.path.join(basedir, folder, frame['file_path']+".png"), downsample_scale=downsample)

                    filenames.append(frame['file_path'])
                    imgs.append(img)

                # poses.append(np.array(frame['transform_matrix'])@ BLENDER2OPENCV)
                poses.append(np.array(frame['transform_matrix']))

                H, W = img.shape[:2]
                focal = .5 * W / np.tan(.5 * camera_angle_x)                            

                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        elif s == "video_spherical":
            ### Use spherical poses
            render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

            poses = []
            intrinsics = []
            for i in range(render_poses.shape[0]):
                poses.append(render_poses[i])
                focal = .5 * W / np.tan(.5 * camera_angle_x)
                focal /= downsample                               

                H, W = img.shape[:2]
                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    
    return imgs, None, None, poses, H, W, intrinsics, near, far, i_split, None, None


def load_scene_blender_fixed_dist_new(basedir, train_json = "transforms_train.json", half_res=True, train_dist=1.0, test_dist=1.0, val_dist=1.0):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []

    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []

    for s in splits:

        if s == "train":
            folder = 'radius_{}_{}'.format(str(train_dist), s)
            transforms_file = 'transforms_radius{}_{}.json'.format(str(train_dist), s)
        elif s == "val":
            folder = 'radius_{}_{}'.format(str(val_dist), s)
            transforms_file = 'transforms_radius{}_{}.json'.format(str(val_dist), s)            
        elif s == "test":
            folder = 'radius_{}_{}'.format(str(test_dist), s)
            transforms_file = 'transforms_radius{}_{}.json'.format(str(test_dist), s)        
        else:
            ## dummy will return not exist
            transforms_file = "blah"

        if os.path.exists(os.path.join(basedir, transforms_file)):

            json_fname =  os.path.join(basedir, transforms_file)

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            # if 'train' in s:
            near = 2.
            far = 6.
            camera_angle_x = float(meta['camera_angle_x'])

            imgs = []
            poses = []
            intrinsics = []

            if s=='train':
                skip = 1
            elif s == "val":
                skip = 1
            elif s =="test":
                skip = 20
            elif "video" in s:
                skip = 1
            
            for frame in meta['frames'][::skip]:
                if len(frame['file_path']) != 0 :
                    if half_res :
                        downsample = 2
                    else:
                        downsample = 1

                    img = read_files(os.path.join(basedir, frame['file_path']+".png"), downsample_scale=downsample)

                    filenames.append(frame['file_path'])
                    imgs.append(img)

                # poses.append(np.array(frame['transform_matrix'])@ BLENDER2OPENCV)
                poses.append(np.array(frame['transform_matrix']))

                H, W = img.shape[:2]
                focal = .5 * W / np.tan(.5 * camera_angle_x)                            

                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        elif s == "video_spherical":
            ### Use spherical poses
            render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

            poses = []
            intrinsics = []
            for i in range(render_poses.shape[0]):
                poses.append(render_poses[i])
                focal = .5 * W / np.tan(.5 * camera_angle_x)
                focal /= downsample                               

                H, W = img.shape[:2]
                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    
    return imgs, None, None, poses, H, W, intrinsics, near, far, i_split, None, None



### With depth
def load_scene_blender_depth(basedir, train_json = "transforms_train.json", half_res=True, train_skip=1, near_plane=2.0):
    splits = ['train', 'val', 'test', 'video']
    # splits = ['test']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = near_plane
                far = 6.
                camera_angle_x = float(meta['camera_angle_x'])

            imgs = []
            depths = []
            valid_depths = []            
            poses = []
            intrinsics = []

            if s=='train':
                skip = train_skip
            elif s =="test":
                skip = 8
            elif "video" in s:
                skip = 1
            
            for frame in meta['frames'][::skip]:
                if len(frame['file_path']) != 0 :
                    if half_res :
                        downsample = 2
                    else:
                        downsample = 1

                    img = read_files(os.path.join(basedir, frame['file_path']+".png"), downsample_scale=downsample)

                    max_depth = frame["max_depth"]
                    depth_scaling_factor = (255. / max_depth)
                    if "chair" in basedir:
                        depth = load_ground_truth_depth(os.path.join(basedir, frame['depth_file_path']+"0000.png"), depth_scaling_factor, near, far)
                    else:
                        depth = load_ground_truth_depth(os.path.join(basedir, frame['depth_file_path']+"0001.png"), depth_scaling_factor, near, far)

                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = np.logical_and(depth[:, :, 0] > near, depth[:, :, 0] < far) # 0 values are invalid depth
                    depth = np.clip(depth, near, far)

                    filenames.append(frame['file_path'])
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                # poses.append(np.array(frame['transform_matrix'])@ BLENDER2OPENCV)
                poses.append(np.array(frame['transform_matrix']))

                H, W = img.shape[:2]
                focal = .5 * W / np.tan(.5 * camera_angle_x)                            

                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))

            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        elif s == "video":
            ### Use spherical poses
            render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

            poses = []
            intrinsics = []
            for i in range(render_poses.shape[0]):
                poses.append(render_poses[i])
                focal = .5 * W / np.tan(.5 * camera_angle_x)
                focal /= downsample                               

                H, W = img.shape[:2]
                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)

    gt_depths = depths
    gt_valid_depths = valid_depths

    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths



def load_scene_blender2_depth(basedir, train_json = "transforms_train.json", half_res=True, train_skip=1, near_plane=2.0):
    splits = ['train', 'val', 'test', 'video']
    # splits = ['test']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, '{}_transforms.json'.format(s))):

            json_fname =  os.path.join(basedir, '{}_transforms.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = near_plane
                far = 6.
                camera_angle_x = float(meta['camera_angle_x'])

            imgs = []
            depths = []
            valid_depths = []               
            poses = []
            intrinsics = []

            if s=='train':
                skip = train_skip
            elif s =="test":
                skip = 8
            elif "video" in s:
                skip = 1
            
            for frame in meta['frames'][::skip]:
                if len(frame['file_path']) != 0 :
                    if half_res :
                        downsample = 2
                    else:
                        downsample = 1

                    img = read_files(os.path.join(basedir, frame['file_path']+".png"), downsample_scale=downsample)


                    max_depth = frame["max_depth"]
                    depth_scaling_factor = (255. / max_depth)

                    # if "chair" in basedir:
                    #     depth = load_ground_truth_depth(os.path.join(basedir, frame['depth_file_path']+"0000.png"), depth_scaling_factor, near, far)
                    # else:
                    #     depth = load_ground_truth_depth(os.path.join(basedir, frame['depth_file_path']+"0001.png"), depth_scaling_factor, near, far)

                    depth = load_ground_truth_depth(os.path.join(basedir, frame['depth_file_path'][:-1]+".png"), depth_scaling_factor, near, far)


                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = np.logical_and(depth[:, :, 0] > near, depth[:, :, 0] < far) # 0 values are invalid depth

                    depth = np.clip(depth, near, far)


                    filenames.append(frame['file_path'])
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                # poses.append(np.array(frame['transform_matrix'])@ BLENDER2OPENCV)
                poses.append(np.array(frame['transform_matrix']))

                H, W = img.shape[:2]
                focal = .5 * W / np.tan(.5 * camera_angle_x)                            

                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))

            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        elif s == "video":
            ### Use spherical poses
            render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

            poses = []
            intrinsics = []
            for i in range(render_poses.shape[0]):
                poses.append(render_poses[i])
                focal = .5 * W / np.tan(.5 * camera_angle_x)
                focal /= downsample                               

                H, W = img.shape[:2]
                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)

    gt_depths = depths
    gt_valid_depths = valid_depths

    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths





















