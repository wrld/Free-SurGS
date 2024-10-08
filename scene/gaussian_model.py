#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import torch.nn.functional as F
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
# from networks.pose_optimizer import transform_to_frame
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import create_random_mask
from utils.geometry_utils import get_pointcloud
from utils.sh_utils import eval_sh
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, training_args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  

        self.params = {
            '_xyz': torch.empty(0),
            '_features_dc': torch.empty(0),
            '_features_rest': torch.empty(0),
            '_opacity': torch.empty(0),
            '_scaling': torch.empty(0),
            '_rotation': torch.empty(0),
        }
        self.variables = {'max_radii2D': torch.empty(0),
                 'xyz_gradient_accum': torch.empty(0),
                 'denom': torch.empty(0),
                 'n_obs': torch.empty(0).int()}
        self.optimizer = None
        self.percent_dense = 0
        self.point_threshold = 20
        self.spatial_lr_scale = 0

        self.net_width = 64
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6  

        self.ch_num = 11
        self.curve_num = 17
        self.init_param = 0.01

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self.params['_xyz'],
            self.params['_features_dc'],
            self.params['_features_rest'],
            self.params['_scaling'],
            self.params['_rotation'],
            self.params['_opacity'],
            self.variables['max_radii2D'],
            self.variables['xyz_gradient_accum'],
            self.variables['denom'],
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self.params['_xyz'], 
        self.params['_features_dc'], 
        self.params['_features_rest'],
        self.params['_scaling'],
        self.params['_rotation'],
        self.params['_opacity'],
        self.variables['max_radii2D'],
        self.variables['xyz_gradient_accum'],
        self.variables['denom'],
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self.params['_scaling'])
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self.params['_rotation'])
    
    @property
    def get_xyz(self):
        return self.params['_xyz']
   
    @property
    def get_features(self):
        features_dc = self.params['_features_dc']
        features_rest = self.params['_features_rest']
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self.params['_opacity'])
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.params['_rotation'])
        
    def get_opacity_bias(self, opacity_bias=None):
        opacity_bias = opacity_bias if opacity_bias is not None else 0.
        return self.opacity_activation(self.params['_opacity'] + opacity_bias)
    
    def get_scale_bias(self, scale_bias=None):
        scale_bias = scale_bias if scale_bias is not None else 0.
        return self.scaling_activation(self.params['_scaling'] + scale_bias)
    
    def get_rotation_bias(self, rotation_bias=None):
        rotation_bias = rotation_bias if rotation_bias is not None else 0.
        return self.rotation_activation(self.params['_rotation'] + rotation_bias)

    def update_pts(self):
        # with torch.no_grad():
        #     self.params['_xyz'].data.add_(self.params['_delta_xyz'].data + self.params['_xyz'].data)

        # Alternatively, if you need to maintain gradient tracking
        self.params['_xyz'] = torch.nn.Parameter((self.params['_xyz'] + self.params['_delta_xyz']).requires_grad_(True))
        # optimizable_tensors = self.replace_tensor_to_optimizer(xyz_new, "_xyz")
        # self.params['_xyz'] = optimizable_tensors["_xyz"]
        # self.params['_xyz'].values = self.params['_xyz'].values + self.params['_delta_xyz'].values
        # self.params['_delta_xyz'] = 

    def reset_deltaxyz(self):
        self.params['_delta_xyz'] = torch.nn.Parameter((torch.zeros_like(self.params['_delta_xyz'])).requires_grad_(True))
        # optimizable_tensors = self.replace_tensor_to_optimizer(xyz_new, "_opacity")
        # self.params['_delta_xyz'] = optimizable_tensors["_delta_xyz"]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
    #     self.spatial_lr_scale = spatial_lr_scale
    #     fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    #     fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    #     features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    #     features[:, :3, 0 ] = fused_color
    #     features[:, 3:, 1:] = 0.0

    #     # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    #     dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    #     scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    #     rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #     rots[:, 0] = 1

    #     opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    #     self.params['_xyz'] = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     self.params['_features_dc'] = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    #     self.params['_features_rest'] = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     self.params['_scaling'] = nn.Parameter(scales.requires_grad_(True))
    #     self.params['_rotation'] = nn.Parameter(rots.requires_grad_(True))
    #     self.params['_opacity'] = nn.Parameter(opacities.requires_grad_(True))
    #     self.variables['max_radii2D'] = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float=5., print_info=True, max_point_num=150_000):
        self.spatial_lr_scale = 5
        if type(pcd.points) == np.ndarray:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        else:
            fused_point_cloud = pcd.points
        if type(pcd.colors) == np.ndarray:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = pcd.colors
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        if print_info:
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self.params['_xyz'] = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.params['_features_dc'] = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.params['_features_rest'] = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.params['_scaling'] = nn.Parameter(scales.requires_grad_(True))
        self.params['_rotation'] = nn.Parameter(rots.requires_grad_(True))
        self.params['_opacity'] = nn.Parameter(opacities.requires_grad_(True))
        self.variables['max_radii2D'] = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # self.feature = nn.Parameter(-1e-2 * torch.ones([self.get_xyz.shape[0], self.fea_dim], dtype=torch.float32).to("cuda:0"), requires_grad=True)
        # if self.with_motion_mask:
        #     self.feature.data[..., -1] = torch.zeros_like(self.feature[..., -1])
    

    def initialize_first_timestep(self, timestep, vo, densify_dataset=None):
        # Get RGB-D Data & Camera Parameters
        self.intrinsic = vo.record_data['intrinsic']
        color, depth, intrinsics, pose = \
            vo.record_data['colors'][timestep], vo.record_data['pred_depths'][timestep], vo.record_data['intrinsic'], vo.record_data['pred_w2c'][timestep]
        color = color.cuda()
        depth = depth.unsqueeze(0).cuda() # (H, W, C) -> (C, H, W)
        
        w2c = torch.from_numpy(pose)
        self.cam = vo.setup_camera(w2c.cpu().numpy())
        mask = create_random_mask(depth.shape[1], depth.shape[2], 0.1)
        mask = mask.reshape(-1).cuda()
        
        init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, intrinsics, w2c, 
                                                    mask=mask, compute_mean_sq_dist=True)
 
        # Initialize Parameters
        
        self.initialize_first_params(init_pt_cld, mean3_sq_dist)
        self.variables['scene_radius'] = torch.max(depth)/2.0
        self.spatial_lr_scale = 5.0 
        print("initialize gaussian points: ", init_pt_cld.shape, "initialize scene radius: ", self.variables['scene_radius'])

    def get_depth_and_silhouette(self, pts_3D, w2c):
        """
        Function to compute depth and silhouette for each gaussian.
        These are evaluated at gaussian center.
        """
        # Depth of each gaussian center in camera frame
        pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
        pts_in_cam = (w2c[0] @ pts4.transpose(0, 1)).transpose(0, 1)
        depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
        depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]
        
        depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
        depth_silhouette[:, 0] = depth_z.squeeze(-1)
        depth_silhouette[:, 1] = 1.0
        depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
        return depth_silhouette
        
    def transformed_params2depthplussilhouette(self, w2c, transformed_pts, scales, rotations, opacity, transformed_next_pts=None):
        if transformed_next_pts is not None:
            means3D = transformed_next_pts
        else:
            means3D = transformed_pts

        rendervar = {
            'means3D': means3D,
            'colors_precomp': self.get_depth_and_silhouette(transformed_pts, w2c),
            'rotations': rotations,
            'opacities': opacity,
            'scales': scales,
            'means2D': torch.zeros_like(self.params['_xyz'], requires_grad=True, device="cuda") + 0
        }
        return rendervar
    
    def project_points(self, points_3d, intrinsics):
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


    def transformed_params2rendervar(self, transformed_pts, scales, rotations, features, opacity, means2D, d_color=None, node_color=None, camera_center=None):
        cov3D_precomp = None
        if d_color is not None:
            sh_features = features
            sh_features = torch.cat([sh_features[:, :1] + d_color[:, None], sh_features[:, 1:]], dim=1) 
        else:
            sh_features = features
            
        shs_view = sh_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        dir_pp = (self.get_xyz - camera_center.repeat(self.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        if node_color is not None:
            colors_precomp = torch.cat([colors_precomp, node_color], dim=0)
        rendervar = {
            'means3D': transformed_pts,
            'shs': None,
            'colors_precomp': colors_precomp,
            'rotations': rotations,
            'opacities': opacity,
            'scales': scales,
            'cov3D_precomp': cov3D_precomp,
            'means2D': means2D
        }
        return rendervar
            
    def initialize_first_params(self, new_pt_cld, mean3_sq_dist):
        num_pts = new_pt_cld.shape[0]
        means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
        unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
        opacities = inverse_sigmoid(0.1 * torch.ones((means3D.shape[0], 1), dtype=torch.float, device="cuda"))

        fused_color = RGB2SH(torch.tensor(new_pt_cld[:, 3:6]).float().cuda())
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        dist2 = torch.clamp_min(distCUDA2(means3D), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        
        params = {
            '_xyz': means3D,
            '_features_dc': features[:,:,0:1].transpose(1, 2).contiguous(),
            '_features_rest': features[:,:,1:].transpose(1, 2).contiguous(),
            '_rotation': unnorm_rots,
            '_opacity': opacities,
            '_scaling': scales,
        }
        self.variables = {'max_radii2D': torch.zeros(params['_xyz'].shape[0]).cuda().float(),
                 'xyz_gradient_accum': torch.zeros(params['_xyz'].shape[0], 1).cuda().float(),
                 'denom': torch.zeros(params['_xyz'].shape[0], 1).cuda().float(),
                 'n_obs': torch.zeros((params['_xyz'].shape[0])).int()}
              
        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
        self.params = params


    def initialize_optimizer(self):
        # lrs = lrs_dict
        param_groups = [{'params': [v], 'name': k, 'lr': self.mapping_lr[k]} for k, v in self.params.items()]
        # if tracking:
        #     return torch.optim.Adam(param_groups)
        # else:
        self.optimizer = torch.optim.Adam(param_groups)

        

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.variables['xyz_gradient_accum'] = torch.zeros((self.params['_xyz'].shape[0], 1), device="cuda")
        self.variables['denom'] = torch.zeros((self.params['_xyz'].shape[0], 1), device="cuda")
        
        self.mapping_lr = {"_xyz": training_args.position_lr_init * self.spatial_lr_scale,
            "_delta_xyz": training_args.position_lr_init * self.spatial_lr_scale,
            "_features_dc": training_args.feature_lr,
            "_features_rest": training_args.feature_lr / 20.0,
            "_opacity": training_args.opacity_lr ,
            "_scaling": training_args.scaling_lr,
            "_rotation": training_args.rotation_lr,
            }

        l = [
            {'params': [self.params['_xyz']], 'lr': self.mapping_lr["_xyz"], "name": "_xyz"},
            {'params': [self.params['_features_dc']], 'lr': self.mapping_lr["_features_dc"], "name": "_features_dc"},
            {'params': [self.params['_features_rest']], 'lr': self.mapping_lr["_features_rest"], "name": "_features_rest"},
            {'params': [self.params['_opacity']], 'lr': self.mapping_lr["_opacity"], "name": "_opacity"},
            {'params': [self.params['_scaling']], 'lr': self.mapping_lr["_scaling"], "name": "_scaling"},
            {'params': [self.params['_rotation']], 'lr': self.mapping_lr["_rotation"], "name": "_rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "_xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.params['_features_dc'].shape[1]*self.params['_features_dc'].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.params['_features_rest'].shape[1]*self.params['_features_rest'].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('_opacity')
        for i in range(self.params['_scaling'].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.params['_rotation'].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.params['_xyz'].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.params['_features_dc'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.params['_features_rest'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.params['_opacity'].detach().cpu().numpy()
        scale = self.params['_scaling'].detach().cpu().numpy()
        rotation = self.params['_rotation'].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        print(opacities_new.shape)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "_opacity")
        self.params['_opacity'] = optimizable_tensors["_opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self.params = {
            '_xyz': nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)),
            '_features_dc': nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)),
            '_features_rest': nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)),
            '_opacity': nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)),
            '_scaling': nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)),
            '_rotation': nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        }
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.params['_xyz'] = optimizable_tensors["_xyz"]
        self.params['_features_dc'] = optimizable_tensors["_features_dc"]
        self.params['_features_rest'] = optimizable_tensors["_features_rest"]
        self.params['_opacity'] = optimizable_tensors["_opacity"]
        self.params['_scaling'] = optimizable_tensors["_scaling"]
        self.params['_rotation'] = optimizable_tensors["_rotation"]

        self.variables['xyz_gradient_accum'] = self.variables['xyz_gradient_accum'][valid_points_mask]
        self.variables['denom'] = self.variables['denom'][valid_points_mask]
        self.variables['max_radii2D'] = self.variables['max_radii2D'][valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # assert len(group["params"]) == 1
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def cat_params_to_optimizer(self, new_params, params, optimizer):
        for k, v in new_params.items():
            group = [g for g in optimizer.param_groups if g['name'] == k][0]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state
                params[k] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
                params[k] = group["params"][0]
        return params

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"_xyz": new_xyz,
        "_features_dc": new_features_dc,
        "_features_rest": new_features_rest,
        "_opacity": new_opacities,
        "_scaling" : new_scaling,
        "_rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.params['_xyz'] = optimizable_tensors["_xyz"]
        self.params['_features_dc'] = optimizable_tensors["_features_dc"]
        self.params['_features_rest'] = optimizable_tensors["_features_rest"]
        self.params['_opacity'] = optimizable_tensors["_opacity"]
        self.params['_scaling'] = optimizable_tensors["_scaling"]
        self.params['_rotation'] = optimizable_tensors["_rotation"]
        
        
        self.variables['xyz_gradient_accum'] = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.variables['denom'] = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.variables['max_radii2D'] = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def densify_and_split(self, grads, grad_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze() 
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.variables['scene_radius'] * 0.01)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.params['_rotation'][selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self.params['_rotation'][selected_pts_mask].repeat(N,1)
        new_features_dc = self.params['_features_dc'][selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.params['_features_rest'][selected_pts_mask].repeat(N,1,1)
        new_opacity = self.params['_opacity'][selected_pts_mask].repeat(N,1)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.variables['scene_radius'] * 0.01)
        
        new_xyz = self.params['_xyz'][selected_pts_mask]
        new_features_dc = self.params['_features_dc'][selected_pts_mask]
        new_features_rest = self.params['_features_rest'][selected_pts_mask]
        new_opacities = self.params['_opacity'][selected_pts_mask]
        new_scaling = self.params['_scaling'][selected_pts_mask]
        new_rotation = self.params['_rotation'][selected_pts_mask]        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, max_screen_size, gs_mask=None):
        if len(self.variables['xyz_gradient_accum'].shape) != 2:
            accum_grad = self.variables['xyz_gradient_accum'].reshape(-1, 1)
        else:
            accum_grad = self.variables['xyz_gradient_accum']
        if len(self.variables['denom'].shape) != 2:
            denom = self.variables['denom'].unsqueeze(1)
        else:
            denom = self.variables['denom']
        grads =  accum_grad / denom
        self.densify_and_clone(grads, max_grad)
        self.densify_and_split(grads, max_grad)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.variables['max_radii2D'] > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * self.variables['scene_radius']
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        print(f"prune points: {torch.sum(prune_mask)}")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        grad = viewspace_point_tensor.grad[update_filter]
        self.variables['xyz_gradient_accum'][update_filter] += torch.norm(grad, dim=-1, keepdim=True)
        self.variables['denom'][update_filter] += 1
    