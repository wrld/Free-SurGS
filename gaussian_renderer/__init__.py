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
import math
from torchviz import make_dot
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.pose_optimizer import transform_to_frame

def depths_to_points(viewmatrix, camera, depthmap):
    c2w = (viewmatrix.T).inverse()
    W, H = camera.new_img_w, camera.new_img_h
    fx = W / (2 * math.tan(camera.FovX / 2.))
    fy = H / (2 * math.tan(camera.FovY / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(viewmatrix, camera, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(viewmatrix, camera, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def render(viewpoint_camera, index, pc : GaussianModel, gs_grad=True, cam_grad=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    means2D = torch.zeros_like(pc.params['_xyz'], requires_grad=True, device="cuda") + 0
    if gs_grad == True:
        means2D.retain_grad()
    viewmatrix_cur = viewpoint_camera.get_pose(index)
    transformed_means3D = transform_to_frame(pc.params['_xyz'], viewmatrix_cur, gs_grad, cam_grad)
    opacity_1 = pc.get_opacity
    scales_1 = pc.get_scaling
    rotation_1 = pc.get_rotation
    rendervar = pc.transformed_params2rendervar(transformed_means3D, scales_1, rotation_1, pc.get_features, \
        opacity_1, means2D,camera_center=viewpoint_camera.cam_center) 
    depth_sil_rendervar = pc.transformed_params2depthplussilhouette(pc.cam.viewmatrix, \
        transformed_means3D, scales_1, rotation_1, opacity_1)
    im, radius, depth = GaussianRasterizer(raster_settings=pc.cam)(**rendervar)
    depth_sil, _, _ = GaussianRasterizer(raster_settings=pc.cam)(**depth_sil_rendervar)
    depth = depth_sil[0, :, :]
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > 0.3)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    pc.variables['means2D'] = rendervar['means2D']
    seen = radius > 0
    pc.variables['max_radii2D'][seen] = torch.max(radius[seen], pc.variables['max_radii2D'][seen])
    pc.variables['seen'] = seen
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    
    return {"render": im,
            "render_dep": depth,
            "render_w2c": viewmatrix_cur,
            "render_opacity": silhouette,
            "nan_mask": nan_mask,
            'presence_mask': presence_sil_mask,
            "uncertainty": uncertainty,
            "viewspace_points": rendervar['means2D'],
            "visibility_filter" : radius > 0,
            "radii": radius}

def inference(viewmatrix_cur, pc : GaussianModel):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    transformed_means3D = transform_to_frame(pc.params['_xyz'], viewmatrix_cur)

    rendervar = pc.transformed_params2rendervar(transformed_means3D)

    im, radius, depth, _ = GaussianRasterizer(raster_settings=pc.cam)(**rendervar)
    return {"render": im,
            "render_depth": depth,
            "viewspace_points": rendervar['means2D'],
            "visibility_filter" : radius > 0,
            "radii": radius}

def render_custom(visualizer_cam, viewpoint_camera, viewmatrix_cur, pc : GaussianModel):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    xyz_final = pc.params['_xyz']
    opacity_final = pc.get_opacity
    scales_final = pc.get_scaling
    rotations_final = pc.get_rotation
    color_final = None
    node_color = None
    transformed_means3D = transform_to_frame(xyz_final, viewmatrix_cur)
    means2D = torch.zeros_like(xyz_final, requires_grad=True, device="cuda") + 0
    shs_final = pc.get_features

    rendervar = pc.transformed_params2rendervar(transformed_means3D, scales_final, rotations_final, shs_final, \
        opacity_final, means2D, color_final, node_color, camera_center=viewpoint_camera.cam_center)
    im, radius, _ = GaussianRasterizer(raster_settings=visualizer_cam)(**rendervar)
    return {"render": im,
            "viewspace_points": rendervar['means2D'],
            "visibility_filter" : radius > 0,
            "radii": radius}
