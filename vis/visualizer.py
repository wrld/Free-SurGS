import argparse
import os
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
# from utils.recon_helpers import setup_camera
# from utils.slam_helpers import get_depth_and_silhouette
# from utils.slam_external import build_rotation


"""This file contains useful layout utilities for images. They are:

- add_border: Add a border to an image.
- cat/hcat/vcat: Join images by arranging them in a line. If the images have different
  sizes, they are aligned as specified (start, end, center). Allows you to specify a gap
  between images.

Images are assumed to be float32 tensors with shape (channel, height, width).
"""

from typing import Any, Generator, Iterable, Literal, Optional, Union

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

Alignment = Literal["start", "center", "end"]
Axis = Literal["horizontal", "vertical"]
Color = Union[
    int,
    float,
    Iterable[int],
    Iterable[float],
    Float[Tensor, "#channel"],
    Float[Tensor, ""],
]

from einops import rearrange, repeat
from jaxtyping import Float, UInt8
FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]

def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()

def _sanitize_color(color: Color) -> Float[Tensor, "#channel"]:
    # Convert tensor to list (or individual item).
    if isinstance(color, torch.Tensor):
        color = color.tolist()

    # Turn iterators and individual items into lists.
    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]

    return torch.tensor(color, dtype=torch.float32)


def _intersperse(iterable: Iterable, delimiter: Any) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item


def _get_main_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 2,
        "vertical": 1,
    }[main_axis]


def _get_cross_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 1,
        "vertical": 2,
    }[main_axis]


def _compute_offset(base: int, overlay: int, align: Alignment) -> slice:
    assert base >= overlay
    offset = {
        "start": 0,
        "center": (base - overlay) // 2,
        "end": base - overlay,
    }[align]
    return slice(offset, offset + overlay)


def overlay(
    base: Float[Tensor, "channel base_height base_width"],
    overlay: Float[Tensor, "channel overlay_height overlay_width"],
    main_axis: Axis,
    main_axis_alignment: Alignment,
    cross_axis_alignment: Alignment,
) -> Float[Tensor, "channel base_height base_width"]:
    # The overlay must be smaller than the base.
    _, base_height, base_width = base.shape
    _, overlay_height, overlay_width = overlay.shape
    assert base_height >= overlay_height and base_width >= overlay_width

    # Compute spacing on the main dimension.
    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(
        base.shape[main_dim], overlay.shape[main_dim], main_axis_alignment
    )

    # Compute spacing on the cross dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim], overlay.shape[cross_dim], cross_axis_alignment
    )

    # Combine the slices and paste the overlay onto the base accordingly.
    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[selector] = overlay
    return result


def cat(
    main_axis: Axis,
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Alignment = "center",
    gap: int = 8,
    gap_color: Color = 1,
) -> Float[Tensor, "channel height width"]:
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""
    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)

    # Find the maximum image side length in the cross axis dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)

    # Pad the images.
    padded_images = []
    for image in images:
        # Create an empty image with the correct size.
        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(overlay(base, image, main_axis, "start", align))

    # Intersperse separators if necessary.
    if gap > 0:
        # Generate a separator.
        c, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim - 1] = cross_axis_length
        separator = torch.ones((c, *separator_size), dtype=torch.float32, device=device)
        separator = separator * gap_color[:, None, None]

        # Intersperse the separator between the images.
        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))


def hcat(
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Literal["start", "center", "end", "top", "bottom"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "horizontal",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "top": "start",
            "bottom": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def vcat(
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "vertical",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def add_border(
    image: Float[Tensor, "channel height width"],
    border: int = 8,
    color: Color = 1,
) -> Float[Tensor, "channel new_height new_width"]:
    color = _sanitize_color(color).to(image)
    c, h, w = image.shape
    result = torch.empty(
        (c, h + 2 * border, w + 2 * border), dtype=torch.float32, device=image.device
    )
    result[:] = color[:, None, None]
    result[:, border : h + border, border : w + border] = image
    return result


def resize(
    image: Float[Tensor, "channel height width"],
    shape: Optional[tuple[int, int]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Float[Tensor, "channel new_height new_width"]:
    assert (shape is not None) + (width is not None) + (height is not None) == 1
    _, h, w = image.shape

    if width is not None:
        shape = (int(h * width / w), width)
    elif height is not None:
        shape = (height, int(w * height / h))

    return F.interpolate(
        image[None],
        shape,
        mode="bilinear",
        align_corners=False,
        antialias="bilinear",
    )[0]

def log_image(key: str, images, step: Optional[int] = None, **kwargs: Any) -> None:
        """Log images (tensors, numpy arrays, PIL Images or file paths).

        Optional kwargs are lists passed to each image (ex: caption, masks, boxes).

        """
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        n = len(images)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]

        import wandb

        metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip(images, kwarg_list)]}
        return metrics
        # self.log_metrics(metrics, step)  # type: ignore[arg-type]

def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    for key, value in all_params.items() :
        print(key, value)
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]

    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k


# def load_scene_data(scene_path):
#     # Load Scene Data
#     all_params = dict(np.load(scene_path, allow_pickle=True))
#     all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}
#     params = all_params

#     all_w2cs = []
#     num_t = params['cam_unnorm_rots'].shape[-1]
#     for t_i in range(num_t):
#         cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
#         cam_tran = params['cam_trans'][..., t_i]
#         rel_w2c = torch.eye(4).cuda().float()
#         rel_w2c[:3, :3] = build_rotation(cam_rot)
#         rel_w2c[:3, 3] = cam_tran
#         all_w2cs.append(rel_w2c.cpu().numpy())
    
#     keys = [k for k in all_params.keys() if
#             k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
#                       'gt_w2c_all_frames', 'cam_unnorm_rots',
#                       'cam_trans', 'keyframe_time_indices']]

#     for k in keys:
#         if not isinstance(all_params[k], torch.Tensor):
#             params[k] = torch.tensor(all_params[k]).cuda().float()
#         else:
#             params[k] = all_params[k].cuda().float()

#     return params, all_w2cs


# def get_rendervars(params, w2c, curr_timestep):
#     params_timesteps = params['timestep']
#     selected_params_idx = params_timesteps <= curr_timestep
#     keys = [k for k in params.keys() if
#             k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
#                       'gt_w2c_all_frames', 'cam_unnorm_rots',
#                       'cam_trans', 'keyframe_time_indices']]
#     selected_params = deepcopy(params)
#     for k in keys:
#         selected_params[k] = selected_params[k][selected_params_idx]
#     transformed_pts = selected_params['means3D']
#     w2c = torch.tensor(w2c).cuda().float()
#     rendervar = {
#         'means3D': transformed_pts,
#         'colors_precomp': selected_params['rgb_colors'],
#         'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
#         'opacities': torch.sigmoid(selected_params['logit_opacities']),
#         'scales': torch.exp(torch.tile(selected_params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
#     }
#     depth_rendervar = {
#         'means3D': transformed_pts,
#         'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
#         'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
#         'opacities': torch.sigmoid(selected_params['logit_opacities']),
#         'scales': torch.exp(torch.tile(selected_params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
#     }
#     return rendervar, depth_rendervar


def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets



def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    
    # Colorize point cloud
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols

