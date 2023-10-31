
import os
import numpy as np
from PIL import Image
import torch
import cv2
import imageio

N_VIEWS = 49
LIGHTING_ID= 3
_opencv2blender = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
_coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32)
_coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32)

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).astype(np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).astype(np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).astype(np.float32)



def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def read_cam_file(filename, scale_factor=1./200.):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4)) @ _opencv2blender # flip camera axis
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0]) * scale_factor
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * scale_factor
    return intrinsics, extrinsics, [depth_min, depth_max]

def read_poses(root_dir, vid, scale_factor=1./200., downSample=1.0):
    proj_mat_filename = os.path.join(root_dir, f'Cameras/train/{vid:08d}_cam.txt')
    intrinsic, extrinsic, near_far = read_cam_file(proj_mat_filename)
    intrinsic[:2] *= 4 # why * 4 ?
    extrinsic[:3, 3] *= scale_factor
    intrinsic[:2] *= downSample

    return near_far, intrinsic, extrinsic, np.linalg.inv(extrinsic)


def load_dtu(root_dir, scene_id, num_train=42, scale_factor = 1. / 200., half_res=True, train_split = None):
    if train_split is None:
        # i_perm = np.random.RandomState(seed=0).permutation(N_VIEWS) # fix a seed so that we get that same split every time 
        # i_train, i_test = i_perm[:num_train], i_perm[num_train:]
        i_test = list(range(N_VIEWS))[::8]
        i_train = [i for i in range(N_VIEWS) if i not in i_test]
    else:
        assert len(train_split) == num_train 
        i_train = train_split
        i_test = [i for i in range(N_VIEWS) if i not in i_train]
    print("USING TRAINGING VIEWS %s and TESTING VIEWS %s" % (i_train, i_test))
    imgs = []
    intrinsics, w2cs, c2ws, near_fars = [], [], [], []  # record proj mats between views
    if half_res :
        downSample = 0.5
    else:
        downSample = 1.0
    counts = [0]
    for vid in i_train:
        img_filename = os.path.join(root_dir, f'Rectified/scan{scene_id}_train/rect_{vid + 1:03d}_{LIGHTING_ID}_r5000.png')
        # depth_filename = os.path.join(root_dir,f'Depths/scan{scene_id}_train/depth_map_{vid:04d}.pfm')
        img = Image.open(img_filename)
        img_wh = np.round(np.array(img.size) * downSample).astype('int')
        img = img.resize(img_wh, Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.
        imgs += [img]
        near_far, intrinsic, w2c, c2w = read_poses(root_dir, vid, scale_factor=scale_factor, downSample=downSample)
        intrinsics.append(intrinsic)
        w2cs.append(w2c)
        c2ws.append(c2w)
        near_fars.append(near_far)
        H, W = img.shape[:2]
        focal = intrinsic[0, 0]
    counts.append(len(i_train))
    for vid in i_test:
        img_filename = os.path.join(root_dir, f'Rectified/scan{scene_id}_train/rect_{vid + 1:03d}_{LIGHTING_ID}_r5000.png')
        # depth_filename = os.path.join(root_dir,f'Depths/scan{scene_id}_train/depth_map_{vid:04d}.pfm')
        img = Image.open(img_filename)
        img_wh = np.round(np.array(img.size) * downSample).astype('int')
        img = img.resize(img_wh, Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.
        imgs += [img]
        near_far, intrinsic, w2c, c2w = read_poses(root_dir, vid, scale_factor=scale_factor, downSample=downSample)
        intrinsics.append(intrinsic)
        w2cs.append(w2c)
        c2ws.append(c2w)
        near_fars.append(near_far)
    near = min([m for m, M in near_fars]) # near plane is the min of all near planes for each view
    far = max([M for m, M in near_fars]) # far plane is the max of all far planes for each view
    counts.append(N_VIEWS)
        

    imgs = np.stack(imgs, axis=0).astype(np.float32)
    intrinsics = np.stack(intrinsics, axis=0).astype(np.float32)
    w2cs = np.stack(w2cs, axis=0).astype(np.float32)
    c2ws = np.stack(c2ws, axis=0).astype(np.float32)
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(2)] # train and test
    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    return imgs, intrinsics, w2cs, render_poses, [H, W, focal], i_split, near, far, [i_train, i_test]




def load_dtu2(root_dir, scene_id, num_train=42, half_res=True, train_split = None):
    scene_dir = os.path.join(root_dir, f"scan{scene_id}")
    image_dirs = os.path.join(scene_dir, "image")
    camera_file_path = os.path.join(scene_dir, "cameras.npz")
    all_cam = np.load(camera_file_path)
    # Prepare to average intrinsics over images
    fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0
    all_imgs = []
    all_poses = []
    if half_res :
        downSample = 0.5
    else:
        downSample = 1.0
    for i in range(N_VIEWS):
        image_path = os.path.join(image_dirs, "%06d.png" % i)
        img = Image.open(image_path)
        img_wh = np.round(np.array(img.size) * downSample).astype('int')
        W, H = img_wh
        img = img.resize(img_wh, Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.
        img = torch.tensor(img)
        
        P = all_cam[f"world_mat_{i}"]
        P = P[:3]

        K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
        K = K / K[2, 2]

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        scale_mtx = all_cam.get(f"scale_mat_{i}")
        if scale_mtx is not None:
            norm_trans = scale_mtx[:3, 3:]
            norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

            pose[:3, 3:] -= norm_trans
            pose[:3, 3:] /= norm_scale

        fx += torch.tensor(K[0, 0]) * downSample
        fy += torch.tensor(K[1, 1]) * downSample
        cx += torch.tensor(K[0, 2]) * downSample
        cy += torch.tensor(K[1, 2]) * downSample
        pose = (_coord_trans_world @ torch.tensor(pose, dtype=torch.float32).cpu() @ _coord_trans_cam)
        all_imgs.append(img)
        all_poses.append(pose)
        
    fx /= N_VIEWS
    fy /= N_VIEWS
    cx /= N_VIEWS
    cy /= N_VIEWS
    focal = torch.tensor((fx, fy), dtype=torch.float32)
    c = torch.tensor((cx, cy), dtype=torch.float32)
    K = torch.tensor([[focal[0], 0, c[0]], [0, focal[1], c[1]], [0, 0, 1.]]).float()
    all_imgs = torch.stack(all_imgs)
    all_poses = torch.stack(all_poses)
    
    if train_split is None:
        # i_perm = np.random.RandomState(seed=0).permutation(N_VIEWS) # fix a seed so that we get that same split every time 
        # i_train, i_test = i_perm[:num_train], i_perm[num_train:]
        i_test = list(range(N_VIEWS))[::8]
        i_train = [i for i in range(N_VIEWS) if i not in i_test]
        num_train = len(i_train)
    else:
        assert len(train_split) == num_train 
        i_train = train_split
        i_test = [i for i in range(N_VIEWS) if i not in i_train]
    print("USING TRAINGING VIEWS %s and TESTING VIEWS %s" % (i_train, i_test))
    counts = [0, num_train, N_VIEWS]
    all_imgs_out = torch.zeros_like(all_imgs)
    all_poses_out = torch.zeros_like(all_poses)
    all_imgs_out[:num_train] = all_imgs[i_train]
    all_imgs_out[num_train:] = all_imgs[i_test]
    all_poses_out[:num_train] = all_poses[i_train]
    all_poses_out[num_train:] = all_poses[i_test]
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(2)] # train and test
    render_poses = torch.stack([torch.tensor(pose_spherical(angle, -30.0, 4.0)) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    # near 0.1, far 5.0 data from pixel nerf
    return all_imgs_out, K, all_poses_out, render_poses, [H, W, focal[0]], i_split, 0.1, 5.0, [i_train, i_test]

