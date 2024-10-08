import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
import math
import glob
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import kornia.geometry.conversions as convert
from utils.graphics_utils import focal2fov
import json
from utils.loss_utils import l1_loss
from utils.general_utils import PILtoTorch, adaptive_thresholding
from utils.camera_utils import nopose_camera_to_JSON
from kornia.geometry.epipolar import essential_from_Rt
from kornia.geometry.epipolar import sampson_epipolar_distance
from kornia.geometry.epipolar import fundamental_from_essential
import imageio
from torch.nn.modules.utils import _pair, _quadruple
UINT16_MAX = 65535

def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])

def load_image(imfile, size = (480, 854)):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    # print(img.shape)
    # img = F.interpolate(img[None], size)
    # print(img.shape)
    return img.cuda()

def get_pointcloud(depth, intrinsics, w2c, sampled_indices, return_sample=False):
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    xx = (sampled_indices[:, 1] - CX)/FX
    yy = (sampled_indices[:, 0] - CY)/FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4)).cuda().float()
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    valid_pt_idx = valid_pt_idx.to(pts.device)
    pts = pts[valid_pt_idx]
    if return_sample:
        sampled_indices_ = sampled_indices[valid_pt_idx]
        return pts, sampled_indices_
    else:
        return pts
def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]
def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform

def setup_ellipse_sampling(views):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view[:3, :3].T, view[:3, 3][:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    poses, transform = transform_poses_pca(poses)

    ts = poses[:, :3, 3]
    t_thetas = np.arctan2(ts[:, 1], ts[:, 0])

    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset

    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

     # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return transform, center, up, low, high, z_low, z_high, ts, t_thetas


def projection_flow_loss(index, render_depth_1, w2c_1, w2c_2, data_record, rigid_mask=None):
    gt_flow = data_record['flows_fw'][index-1].float().cuda()
    intrinsic = data_record['intrinsic']
    render_depth_1 = render_depth_1.float().cuda()
    width, height = render_depth_1.shape[2], render_depth_1.shape[1]
    
    # Create a depth mask
    depth_mask = render_depth_1[0]
    if rigid_mask is not None:
        depth_mask = depth_mask * rigid_mask
    
    # Find valid depth indices
    valid_depth_indices = torch.where(depth_mask > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)

    # Back Project the selected pixels to 3D Pointcloud
    w2c_1 = torch.from_numpy(w2c_1).float().cuda()
    pts, valid_depth_indices = get_pointcloud(render_depth_1, intrinsic, w2c_1, valid_depth_indices, True)

    # Transform the 3D pointcloud to the keyframe's camera space
    pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
    transformed_pts = (w2c_2 @ pts4.T).T[:, :3]

    # Project the 3D pointcloud to the keyframe's image space
    intrinsic = torch.from_numpy(intrinsic).float().cuda()
    points_2d = torch.matmul(intrinsic, transformed_pts.transpose(0, 1)).transpose(0, 1)
    points_z = points_2d[:, 2:] + 1e-5  # Small epsilon to prevent division by zero
    points_2d = points_2d / points_z
    
    projected_pts = points_2d[:, :2]
    edge = 20
    mask = (projected_pts[:, 0] < width - edge) & (projected_pts[:, 0] > edge) & \
           (projected_pts[:, 1] < height - edge) & (projected_pts[:, 1] > edge) & \
           (points_z[:, 0] > 0)

    projected_pts = projected_pts[mask]
    valid_depth_indices = valid_depth_indices[mask]

    if projected_pts.numel() == 0 or valid_depth_indices.numel() == 0:
        # Handle the case where no valid points remain
        return torch.tensor(0.0, device=gt_flow.device)  # Or another appropriate value

    valid_depth_indices = valid_depth_indices[:, [1, 0]]  # Ensure correct indexing

    # Calculate projection flow and ground truth backward flow
    projection_flow = projected_pts - valid_depth_indices.float()  # Ensure float type
    gt_bw_flow = gt_flow[:, valid_depth_indices[:, 1], valid_depth_indices[:, 0]].permute(1, 0)

    # Ensure no NaNs before loss calculation
    if torch.isnan(projection_flow).any() or torch.isnan(gt_bw_flow).any():
        return torch.tensor(0.0, device=gt_flow.device)  # Or handle appropriately

    loss = l1_loss(projection_flow, gt_bw_flow)

    return loss



def unprojection(xy, depth, K):
    # xy: [N, 2] image coordinates of match points
    # depth: [N] depth value of match points
    N = xy.shape[0]
    # initialize regular grid
    ones = np.ones((N, 1))
    xy_h = np.concatenate([xy, ones], axis=1)
    xy_h = np.transpose(xy_h, (1,0)) # [3, N]
    #depth = np.transpose(depth, (1,0)) # [1, N]
    
    K_inv = np.linalg.inv(K)
    points = np.matmul(K_inv, xy_h) * depth
    points = np.transpose(points) # [N, 3]
    return points

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
import re
def extract_number(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def project_points(points_3d, intrinsics):
    """
    Function to project 3D points to image plane.
    params:
    points_3d: [num_gaussians, 3]
    intrinsics: [3, 3]
    out: [num_gaussians, 2]
    """
    points_2d = torch.matmul(intrinsics, points_3d.transpose(0, 1))
    points_2d = points_2d.transpose(0, 1)
    points_2d = points_2d / points_2d[:, 2:]
    points_2d = points_2d[:, :2]
    return points_2d

def load_cameras(
    path: str, H: int, W: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert os.path.exists(path), f"Camera file {path} does not exist."
    recon = np.load(path, allow_pickle=True).item()
    # guru.debug(f"{recon.keys()=}")
    traj_c2w = recon["traj_c2w"]  # (N, 4, 4)
    h, w = recon["img_shape"]
    sy, sx = H / h, W / w
    traj_w2c = np.linalg.inv(traj_c2w)
    fx, fy, cx, cy = recon["intrinsics"]  # (4,)
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])  # (3, 3)
    Ks = np.tile(K[None, ...], (len(traj_c2w), 1, 1))  # (N, 3, 3)
    kf_tstamps = recon["tstamps"].astype("int")
    return (
        torch.from_numpy(traj_w2c).float(),
        torch.from_numpy(Ks).float(),
        torch.from_numpy(kf_tstamps),
    )


def _get_padding(x, k, stride, padding, same: bool):
    if same:
        ih, iw = x.size()[2:]
        if ih % stride[0] == 0:
            ph = max(k[0] - stride[0], 0)
        else:
            ph = max(k[0] - (ih % stride[0]), 0)
        if iw % stride[1] == 0:
            pw = max(k[1] - stride[1], 0)
        else:
            pw = max(k[1] - (iw % stride[1]), 0)
        pl = pw // 2
        pr = pw - pl
        pt = ph // 2
        pb = ph - pt
        padding = (pl, pr, pt, pb)
    else:
        padding = padding
    return padding

def median_filter_2d(x, kernel_size=3, stride=1, padding=1, same: bool = True):
    """
    :param x [B, C, H, W]
    """
    k = _pair(kernel_size)
    stride = _pair(stride)  # convert to tuple
    padding = _quadruple(padding)  # convert to l, r, t, b
    # using existing pytorch functions and tensor ops so that we get autograd,
    # would likely be more efficient to implement from scratch at C/Cuda level
    x = F.pad(x, _get_padding(x, k, stride, padding, same), mode="reflect")
    x = x.unfold(2, k[0], stride[0]).unfold(3, k[1], stride[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x

def load_depth(path) -> torch.Tensor:
    disp = imageio.imread(path)
    if disp.dtype == np.uint16:
        disp = disp.astype(np.float32) / UINT16_MAX
    depth = 1.0 / np.clip(disp, a_min=1e-6, a_max=1e6)
    depth = torch.from_numpy(depth)
    depth = median_filter_2d(depth[None, None], 11, 1)[0, 0]
    return depth


class PoseModel:
    def __init__(self, args, device="cuda", sample_rate=8):
        self.data_path = args.source_path
        self.data_type = args.data_type
        self.start_frame = args.frame_start
        self.end_frame = args.frame_end
        rgb_paths = glob.glob(os.path.join(self.data_path, 'input', '*.png')) + \
                 glob.glob(os.path.join(self.data_path, 'input', '*.jpeg'))+ \
                 glob.glob(os.path.join(self.data_path, 'input', '*.jpg'))
        self.W, self.H = Image.open(rgb_paths[0]).size
            
        rgb_paths = sorted(rgb_paths)
        if self.end_frame == -1:
            num_cams = len(rgb_paths)
        else:
            print("from", rgb_paths[0], "to", rgb_paths[-1])
            rgb_paths = rgb_paths[self.start_frame:self.end_frame]
            num_cams = len(rgb_paths)
            print("up from", rgb_paths[0], "to", rgb_paths[-1])
        pose_path_dir = os.path.join(self.data_path, "poses")  
        origin_images = []
        flows_fw = []
        flows_bw = []
        mono_deps = []
        gt_poses = {}
        print("get rgb images: ", num_cams)
        for i in range(num_cams):
            rgb_name = rgb_paths[i].split('/')[-1]
            scene_ind = rgb_name.split('_')[0]
            data_ind = rgb_name.split('_')[1]
            img_name = rgb_name.split('_')[3].split('.')[0]
            pose_path = os.path.join(pose_path_dir, f'{scene_ind}_{data_ind}', f'frame_{img_name}.json')
            with open(pose_path, 'r') as json_file:
                data = json.load(json_file)
            gt_pose = np.array(data["camera-pose"])
            self.intrinsic = np.array(data["camera-calibration"]["KL"])
            if data_ind in gt_poses:
                gt_poses[data_ind] += [gt_pose]
            else:
                gt_poses[data_ind] = [gt_pose]
            # data_inds.append(data_ind)
            rgb_name_only = rgb_name.split('.')[0]
            flow_bw_path = os.path.join(self.data_path, f"flow/flow_bw_{rgb_name_only}.npz")
            flow_fw_path = os.path.join(self.data_path, f"flow/flow_fw_{rgb_name_only}.npz")
            depth_path = os.path.join(self.data_path, f"monodep/depth_{rgb_name_only}.npz")
            
            if i < num_cams-1:
                flow_fw = np.load(flow_fw_path)['pred']
                flows_fw.append(flow_fw)
                flow_bw = np.load(flow_bw_path)['pred']
                flows_bw.append(flow_bw)
                
            mono_dep = 1. / np.load(depth_path)['pred']
            mono_dep = (mono_dep - mono_dep.min())/(mono_dep.max() - mono_dep.min()) * 1.0 + 0.5
        
            mono_deps.append(mono_dep)
            image = PILtoTorch(Image.open(rgb_paths[i]))
            origin_images.append(image)
        self.scene = f"scared_{scene_ind}"
        self.intrinsic[0,:] = self.intrinsic[0,:] * self.W / 1280
        self.intrinsic[1,:] = self.intrinsic[1,:] * self.H / 1024
    
        all_indexes = np.array(range(0, len(rgb_paths)))
        
        self.i_test = all_indexes[int(sample_rate/2)::sample_rate]
        self.i_train = np.array([i for i in all_indexes if i not in self.i_test])
        
        
        self.zfar = 100.0
        self.znear = 0.01
        
        self.FovX = focal2fov(self.intrinsic[0,0], self.W)
        self.FovY = focal2fov(self.intrinsic[1,1], self.H)
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.FovX, self.FovY).transpose(0,1).cuda()
        

        data_inds = [0]
        pre_ind = 0
        weights = []
        for key, value in gt_poses.items():
            data_inds.append(len(value) + pre_ind)
            weights.append(len(value) / num_cams * 1.0)
            pre_ind = len(value) + pre_ind
            print("dataind at", key, len(value), data_inds)
            gt_poses[key] =  torch.from_numpy(np.stack(value)).float()
            

        self.record_data = {
            'image_name': rgb_paths,
            'colors': torch.from_numpy(np.stack(origin_images)).float(),
            "flows_fw": torch.from_numpy(np.concatenate(flows_fw)).float(),
            "flows_bw": torch.from_numpy(np.concatenate(flows_bw)).float(),
            "monodeps": torch.from_numpy(np.stack(mono_deps)).float(),
            "data_ind": data_inds,
            "weights": weights,
            'gt_poses': gt_poses,
            'pred_colors': torch.zeros((num_cams, 3, self.H, self.W)),
            'pred_depths': torch.zeros((num_cams, self.H, self.W)),
            'pred_masks': np.ones((num_cams, self.H, self.W)),
            'rigid_masks': np.ones((num_cams , self.H, self.W)),
            'pred_w2c': np.zeros((num_cams, 4, 4)),
            'update_c2w':np.zeros((num_cams, 4, 4)),
            'intrinsic': self.intrinsic,
            'image_height': self.H,
            'image_width': self.W,
            'pose_scales': np.zeros(num_cams)
        }
        self.keyframe_list = []
        self.mapping_window_size = 5
        

        self.init_c2ws = None
        self.gs_mask = None

        self.pose_param_net = LearnPose(num_cams, self.H, self.W, self.FovX, self.FovY, self.init_c2ws).to(device)
        self.keyframe_selection_method = 'overlap'
        self.num_cams = num_cams

    def capture(self):
        return (
            self.optimizer.state_dict(),
            self.pose_param_net.r,
            self.pose_param_net.t,
            self.record_data['pred_w2c'],
            self.record_data['intrinsic']
        )
    
    def restore(self, model_args):
        (opt_dict, 
        self.pose_param_net.r,
        self.pose_param_net.t,
        self.record_data['pred_w2c'],
        self.record_data['intrinsic'],
        ) = model_args
        
    def initialize_tracking_optimizer(self, tracking_iter):
        self.optimizer = optim.Adam([
                {'params': self.pose_param_net.r, 'lr': 0.01},
                {'params': self.pose_param_net.t, 'lr': 0.01}
            ], lr=0.001, eps=1e-15)
        self.scheduler_eval_pose = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                   milestones=list(range(0, int(tracking_iter), int(tracking_iter/3))),
                                                                gamma=0.5)   

    def initialize_pose(self, curr_time_idx, forward_prop=True, pnp = True):
        with torch.no_grad():
            if pnp == False:
                if curr_time_idx > 1 and forward_prop:
                    # Initialize the camera pose for the current frame based on a constant velocity model
                    # Rotation
                    prev_rot1 = F.normalize(self.pose_param_net.r[..., curr_time_idx-1].detach())
                    prev_rot2 = F.normalize(self.pose_param_net.r[..., curr_time_idx-2].detach())
                    new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
                    self.pose_param_net.r[..., curr_time_idx] = new_rot.detach()
                    # Translation
                    prev_tran1 = self.pose_param_net.t[..., curr_time_idx-1].detach()
                    prev_tran2 = self.pose_param_net.t[..., curr_time_idx-2].detach()
                    new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
                    self.pose_param_net.t[..., curr_time_idx] = new_tran.detach()
                else:
                    # Initialize the camera pose for the current frame
                    self.pose_param_net.r[..., curr_time_idx] = self.pose_param_net.r[..., curr_time_idx-1].detach()
                    self.pose_param_net.t[..., curr_time_idx] = self.pose_param_net.t[..., curr_time_idx-1].detach()
            else:
                match, mask = self.get_matches(curr_time_idx-1, curr_time_idx)
                points1 = match[:,:2,:].squeeze(0).permute(1,0)
                points2 = match[:,2:,:].squeeze(0).permute(1,0)
                previous_pose = self.record_data['pred_w2c'][curr_time_idx-1]
                rel_pose = self.solve_pose_pnp(points1.cpu().numpy(), points2.cpu().numpy(), self.record_data['pred_depths'][curr_time_idx-1].numpy())
                pred_pose = np.zeros_like(previous_pose)
                pred_pose[:3,3:] = np.matmul(previous_pose[:3,:3], rel_pose[:3,3:]) + previous_pose[:3,3:]
                pred_pose[:3,:3] = np.matmul(previous_pose[:3,:3], rel_pose[:3,:3])
                # rot_update_pose(pred_pose, curr_time_idx, self)
                R = torch.from_numpy(pred_pose[:3, :3])
                t = torch.from_numpy(pred_pose[:3, 3])
                quat = convert.rotation_matrix_to_quaternion(R.contiguous())[0]
                with torch.no_grad():
                    self.pose_param_net.r[..., curr_time_idx] = quat
                    self.pose_param_net.t[..., curr_time_idx] = t
    
    def keyframe_selection_overlap(self, gt_depth, w2c, intrinsics, keyframe_list, record_data, k, pixels=1600):
        # Radomly Sample Pixel Indices from valid depth pixels
        width, height = gt_depth.shape[2], gt_depth.shape[1]
        valid_depth_indices = torch.where(gt_depth[0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
        sampled_indices = valid_depth_indices[indices]

        # Back Project the selected pixels to 3D Pointcloud
        pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

        list_keyframe = []
        for index in keyframe_list:
            # Get the estimated world2cam of the keyframe
            est_w2c = torch.from_numpy(record_data['pred_w2c'][index]).float()
            # Transform the 3D pointcloud to the keyframe's camera space
            pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
            transformed_pts = (est_w2c @ pts4.T).T[:, :3]
            # Project the 3D pointcloud to the keyframe's image space
            points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
            points_2d = points_2d.transpose(0, 1)
            points_z = points_2d[:, 2:] + 1e-5
            points_2d = points_2d / points_z
            projected_pts = points_2d[:, :2]
            # Filter out the points that are outside the image
            edge = 20
            mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
                (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
            mask = mask & (points_z[:, 0] > 0)
            # Compute the percentage of points that are inside the image
            percent_inside = mask.sum()/projected_pts.shape[0]
            list_keyframe.append(
                {'id': index, 'percent_inside': percent_inside})

        # Sort the keyframes based on the percentage of points that are inside the image
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # Select the keyframes with percentage of points inside the image > 0
        selected_keyframe_list = [keyframe_dict['id']
                                  for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])

        return selected_keyframe_list

    def save_json_file(self, path):
        json_cams = []
        # last_data = None
        for i in range(self.num_cams):
            if i <= self.record_data['timestep']:
                data = nopose_camera_to_JSON(i, self.record_data)
                json_cams.append(data)
                last_data = data
            else:
                json_cams.append(last_data)
        with open(os.path.join(path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        with open(os.path.join(self.data_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

    def update_learning_rate(self, lr):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            return lr

    def setup_camera(self, w2c, visualize_data=None, near=0.01, far=100):
        # fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        w2c = torch.tensor(w2c).cuda().float()
        self.cam_center = torch.inverse(w2c)[:3, 3]
        w2c = w2c.unsqueeze(0).transpose(1, 2)
        if visualize_data is None:
            fx, fy, cx, cy = self.record_data['intrinsic'][0][0], self.record_data['intrinsic'][1][1], \
                self.record_data['intrinsic'][0][2], self.record_data['intrinsic'][1][2]
            w, h = self.record_data['image_width'], self.record_data['image_height']
            
        else:
            fx, fy, cx, cy = visualize_data['K'][0][0], visualize_data['K'][1][1], \
                visualize_data['K'][0][2], visualize_data['K'][1][2]
            w, h = visualize_data['W'], visualize_data['H']
        opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                    [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
        full_proj = w2c.bmm(opengl_proj)
        cam = Camera(
            image_height=h,
            image_width=w,
            tanfovx=w / (2 * fx),
            tanfovy=h / (2 * fy),
            bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=self.cam_center,
            prefiltered=False,
            debug=False,
        )
        return cam

    def get_pose(self, timestep):
        update_pose = self.pose_param_net.forward(timestep)
        self.record_data['pred_w2c'][timestep] = update_pose.detach().cpu().numpy()
        return update_pose

    def get_fundamental_matrix(self, timestep_1, timestep_2):
        Rt_1 = self.pose_param_net.forward(timestep_1).detach()
        R_1, t_1 = Rt_1[:3, :3], Rt_1[:3, 3]
        Rt_2 = self.pose_param_net.forward(timestep_2).detach()
        R_2, t_2 = Rt_2[:3, :3], Rt_2[:3, 3]
        intrinsic = torch.from_numpy(self.record_data['intrinsic']).unsqueeze(0).float().cuda()
        F = fundamental_from_essential(essential_from_Rt(R_1.unsqueeze(0), t_1.reshape(1, 3, 1), \
            R_2.unsqueeze(0), t_2.reshape(1, 3, 1)), intrinsic, intrinsic)
        return F
    
    def get_flow(self, timestep_1, timestep_2):
        image1 = load_image(self.record_data['image_name'][timestep_1])
        image2 = load_image(self.record_data['image_name'][timestep_2])

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_low, flow_up_fw, up_mask = self.raft_model(image1, image2, iters=20, test_mode=True)
        return flow_up_fw.detach().float()
    
    def ref_projection_flow_loss(self, index_ref, index, render_depth_2, w2c_2, data_record):
        # gt_flow = data_record['flows_bw'][index-1].float().cuda()
        gt_flow = self.get_flow(index_ref, index).squeeze(0)
        intrinsic = data_record['intrinsic']
        width, height = render_depth_2.shape[2], render_depth_2.shape[1]
        valid_depth_indices = torch.where(render_depth_2[0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        w2c_1 = self.get_pose(index_ref)
        # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
        # sampled_indices = valid_depth_indices[valid_depth_indices]

        # Back Project the selected pixels to 3D Pointcloud=
        
        pts, valid_depth_indices = get_pointcloud(render_depth_2, intrinsic, w2c_2, valid_depth_indices, True)

        # for index in keyframe_list:
        # Get the estimated world2cam of the keyframe
        # est_w2c = torch.from_numpy(w2c_2).float()
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        # w2c_1 = torch.from_numpy(w2c_1).float().cuda()
        transformed_pts = (w2c_1 @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        intrinsic = torch.from_numpy(intrinsic).float().cuda()
        points_2d = torch.matmul(intrinsic, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]
        edge = 20
        mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
            (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)
        projected_pts = projected_pts[mask]
        valid_depth_indices = valid_depth_indices[mask]
        valid_depth_indices = valid_depth_indices[:, [1,0]]
        projection_flow = projected_pts - valid_depth_indices 
        gt_bw_flow = gt_flow[:, valid_depth_indices[:, 1], valid_depth_indices[:, 0]].permute(1,0)
        loss = l1_loss(projection_flow, -gt_bw_flow)
        return loss

    def get_matches(self, timestep_1, timestep_2, seed=None):
        flow = self.record_data['flows_fw'][timestep_1].permute(1,2,0).cuda()
        # flow = self.get_flow(timestep_1, timestep_2).squeeze(0).permute(1,2,0)
        h, w = self.record_data['image_height'], self.record_data['image_width']
        x = torch.arange(w, dtype=torch.long).cuda()
        y = torch.arange(h, dtype=torch.long).cuda()
        yy, xx = torch.meshgrid(y, x)
        pts_source = torch.stack([xx, yy], dim=-1)

        # Yield Correspondence
        xx_source, yy_source = torch.split(pts_source, 1, dim=-1)
        xx_target, yy_target = torch.split(pts_source + flow, 1, dim=-1)
        xx_source, yy_source, xx_target, yy_target = xx_source.squeeze(), yy_source.squeeze(), xx_target.squeeze(), yy_target.squeeze()

        # Ignore Regeion
        # if valid_regeion is None:
        valid_regeion = torch.ones((h,w)).cuda()
        valid_regeion = valid_regeion * (xx_target > 0) * (xx_target < w) * (yy_target > 0) * (yy_target < h) #* (depth > 0)
        valid_regeion = valid_regeion.bool()
        # Seeding for reproduction
        # if seed is not None: np.random.seed(seed)

        # pts_idx = np.random.randint(0, torch.sum(valid_regeion).item(), self.match_num)

        # Sample Correspondence
        # xxf_source, yyf_source = xx_source[valid_regeion][pts_idx], yy_source[valid_regeion][pts_idx]
        # xxf_target, yyf_target = xx_target[valid_regeion][pts_idx], yy_target[valid_regeion][pts_idx]
        pts_source, pts_target = torch.stack([xx_source, yy_source], axis=0).float(), torch.stack([xx_target, yy_target], axis=0).float()
        matches = torch.cat([pts_source, pts_target], dim=0).unsqueeze(0).view([1, 4, -1])
        valid_regeion = valid_regeion.view(1, 1, -1)
        return matches, valid_regeion

    def compute_epipolar_loss(self, timestep_1, timestep_2):
        # fmat: [b, 3, 3] match: [b, 4, h*w] mask: [b,1,h*w]
        match, mask = self.get_matches(timestep_1, timestep_2)
        fmat = self.get_fundamental_matrix(timestep_1, timestep_2)
        num_batch = match.shape[0]
        match_num = match.shape[-1]
        h, w = self.record_data['image_height'], self.record_data['image_width']
        
        points1 = match[:,:2,:].permute(0, 2, 1)
        points2 = match[:,2:,:].permute(0, 2, 1)
        
        samp_dis = sampson_epipolar_distance(points1, points2, fmat)
        img1_rigid_mask = (samp_dis.view([h, w])).float()
        loss = samp_dis.mean()
        return loss, img1_rigid_mask



            


        

class LearnPose(nn.Module):
    def __init__(self, num_cams, image_height, image_width, FoVx, FoVy, init_c2w=None, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, learn_R=True, learn_t=True, device="cuda"):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        # if init_c2w is not None:
        #     self.init_c2w = nn.Parameter(init_c2w.float(), requires_grad=False).cuda()
        # identity_matrix = torch.eye(3, dtype=torch.float32)
        # Create a batch of identity matrices, one for each camera
        # initial_R = identity_matrix.repeat(num_cams, 1, 1)
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_cams))
        self.r =  nn.Parameter(torch.from_numpy(cam_rots).cuda().float().contiguous().requires_grad_(True))
        
        # self.R = nn.Parameter(torch.zeros(size=(num_cams, 3, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t =  nn.Parameter(torch.zeros(size=(3, num_cams)).cuda().float().contiguous().requires_grad_(True))  # (N, 3)
        # self.t[0].requires_grad_(False)
        # self.uids = uids
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.zfar = 100.0
        self.znear = 0.01
        
        self.trans = trans
        self.scale = scale
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        # w2c_matrix, full_proj_transform, camera_center = self.forward(0)
        # self.first_frame = {"w2c_matrix": w2c_matrix.detach(), "full_proj_transform": full_proj_transform.detach(), "camera_center": camera_center.detach()}

    def visualization(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for cam_id in range(self.num_cams):
            r = F.normalize(self.r[..., cam_id])  # (3, ) axis-angle
            t = self.t[..., cam_id]  # (3, )
            w2c_matrix = self.getWorld2View2(r, t).transpose(0, 1)

            # Extract camera position
            camera_position = w2c_matrix.inverse()[3, :3].detach().cpu().numpy()

            # Extract the camera direction (using the first column of the rotation matrix)
            camera_direction = w2c_matrix[:3, 0].detach().cpu().numpy()

            # Plot camera position as a point
            ax.scatter(camera_position[0], camera_position[1], camera_position[2], marker='o')

            # Plot camera orientation as an arrow
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      camera_direction[0], camera_direction[1], camera_direction[2],
                      length=0.1, normalize=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Camera Poses Visualization')
        plt.show()

    def forward(self, cam_id, c2w_vo = None):
        cam_id = int(cam_id)
        
            
        r = F.normalize(self.r[..., cam_id])  # (3, ) axis-angle
        t = self.t[..., cam_id]  # (3, )
        # g = make_dot(loss)
        # g.view()
        # if self.init_c2w is not None:
        #     w2c_matrix = self.getWorld2View2(r, t, self.init_c2w[cam_id]).transpose(0, 1)
        # elif c2w_vo is not None:
        #     w2c_matrix = self.getWorld2View2(r, t, c2w_vo).transpose(0, 1)
        # else:
        w2c_matrix = self.getWorld2View2(r, t)
            
        # full_proj_transform = (w2c_matrix.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0).to(self.device))).squeeze(0)
        # camera_center = w2c_matrix.inverse()[3, :3]
        
        return w2c_matrix#, full_proj_transform, camera_center


    def q2rot(self, q):
        norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
        q = q / norm[:, None]
        rot = torch.zeros((q.size(0), 3, 3), device='cuda')
        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]
        rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
        rot[:, 0, 1] = 2 * (x * y - r * z)
        rot[:, 0, 2] = 2 * (x * z + r * y)
        rot[:, 1, 0] = 2 * (x * y + r * z)
        rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
        rot[:, 1, 2] = 2 * (y * z - r * x)
        rot[:, 2, 0] = 2 * (x * z - r * y)
        rot[:, 2, 1] = 2 * (y * z + r * x)
        rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return rot

    def getWorld2View2(self, r, t, init_c2w=None, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
        Rt = torch.eye(4).cuda().float()
        R = self.q2rot(r)[0]
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        # Rt[3, 3] = 1.0

        # C2W = torch.inverse(Rt)
        # cam_center = C2W[:3, 3].clone()
        # cam_center = (cam_center + translate.to(self.device)) * scale
        # C2W = C2W.clone()
        # C2W[:3, 3] = cam_center
        # if init_c2w is not None:
        #     C2W = C2W @ init_c2w
        # Rt = torch.inverse(C2W)
        return Rt


    def get_t(self):
       return self.t


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)





# def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
#                    mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
#     width, height = color.shape[2], color.shape[1]
#     CX = intrinsics[0][2]
#     CY = intrinsics[1][2]
#     FX = intrinsics[0][0]
#     FY = intrinsics[1][1]

#     # Compute indices of pixels
#     x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
#                                     torch.arange(height).cuda().float(),
#                                     indexing='xy')
#     xx = (x_grid - CX)/FX
#     yy = (y_grid - CY)/FY
#     xx = xx.reshape(-1)
#     yy = yy.reshape(-1)
#     depth_z = depth[0].reshape(-1)

#     # Initialize point cloud
#     pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
#     if transform_pts:
#         pix_ones = torch.ones(height * width, 1).cuda().float()
#         pts4 = torch.cat((pts_cam, pix_ones), dim=1)
#         c2w = torch.inverse(w2c)
#         pts = (c2w @ pts4.T).T[:, :3]
#     else:
#         pts = pts_cam

#     # Compute mean squared distance for initializing the scale of the Gaussians
#     if compute_mean_sq_dist:
#         if mean_sq_dist_method == "projective":
#             # Projective Geometry (this is fast, farther -> larger radius)
#             scale_gaussian = depth_z / ((FX + FY)/2)
#             mean3_sq_dist = scale_gaussian**2
#         else:
#             raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
#     # Colorize point cloud
#     cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
#     point_cld = torch.cat((pts, cols), -1)

#     # Select points based on mask
#     if mask is not None:
#         point_cld = point_cld[mask]
#         if compute_mean_sq_dist:
#             mean3_sq_dist = mean3_sq_dist[mask]

#     if compute_mean_sq_dist:
#         return point_cld, mean3_sq_dist
#     else:
#         return point_cld


def transformed_params2rendervar(params, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar
   
def transform_to_frame(means3D, viewmatrix, gaussians_grad = True, camera_grad = True):
    """
    Function to transform Isotropic Gaussians from world frame to camera frame.
    
    Args:
        params: dict of parameters
        time_idx: time index to transform to
        gaussians_grad: enable gradients for Gaussians
        camera_grad: enable gradients for camera pose
    
    Returns:
        transformed_pts: Transformed Centers of Gaussians
    """
    # Get Frame Camera Pose
    if camera_grad:
        rel_w2c = viewmatrix
    else:
        rel_w2c = viewmatrix.detach()
    # Get Centers and norm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = means3D
    else:
        pts = means3D.detach()
    
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).to(pts.device).float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

    return transformed_pts 

