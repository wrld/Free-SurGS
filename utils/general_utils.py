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
import sys
from datetime import datetime
import numpy as np
import random
from utils.loss_utils import l1_loss, ssim
import kornia.geometry.conversions as convert
import matplotlib.pyplot as plt
from utils.common_utils import visualize_torch_image, flow_to_image, normalize_tensor_image, visualize_depth
import lpips
import skimage

def rgb_evaluation(gts, predicts):
    assert gts.max() <= 1
    gts = gts.astype(np.float32)
    predicts = predicts.astype(np.float32)
    ssim_list = []
    mse = ((gts - predicts)**2).mean(-1).mean(-1).mean(-1)
    psnr = (-10 * np.log10(mse)).mean()
    lpips_metric = lpips.LPIPS(net='alex', version='0.1', verbose = False)
    gts_torch = torch.from_numpy((2 * gts - 1)).type(torch.FloatTensor)
    predicts_torch = torch.from_numpy(
        (2 * predicts - 1)).type(torch.FloatTensor)
    lpips_ = lpips_metric(gts_torch, predicts_torch).mean()
    for i in range(gts.shape[0]):
        gt = gts[i]
        predict = predicts[i]
        gt = np.moveaxis(gt, 0, -1)
        predict = np.moveaxis(predict, 0, -1)
        ssim_list.append(
            skimage.metrics.structural_similarity(gt,
                                                  predict,
                                                  data_range=1,
                                                  multichannel=True,
                                                  channel_axis=2))

    ssim = np.array(ssim_list).mean()
    print(f'psnr: {psnr}, ssim: {ssim}, lpips: {lpips_}\n')
    return psnr, ssim, lpips_


def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)


def plot_loss(loss1, loss2, loss3, timestep, save_path):
    epochs = range(1, len(loss1) + 1)
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Plot loss 1
    axs[0].plot(epochs, loss1, 'b')
    axs[0].set_title('epipolar loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')

    # Plot loss 2
    axs[1].plot(epochs, loss2, 'r')
    axs[1].set_title('flow loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')

    # Plot loss 3
    axs[2].plot(epochs, loss3, 'g')
    axs[2].set_title('rgb loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # if iter == None:
    filename = f"{save_path}/curve_{timestep}.png"
    
    # Save the figure
    plt.savefig(filename)
    plt.close(fig)

def adaptive_thresholding(tensor, factor=2.0):
    """
    Apply adaptive thresholding to segment the object from the background.

    Args:
        tensor (torch.Tensor): The input tensor to threshold.
        factor (float): The factor to determine the threshold based on mean and std deviation.

    Returns:
        torch.Tensor: Binary mask tensor where object pixels are 1 and background pixels are 0.
    """
    # Calculate mean and standard deviation of the tensor
    mean_value = tensor.mean().item()
    std_value = tensor.std().item()

    # Set threshold as mean + factor * std
    threshold = mean_value + factor * std_value
    # print(mean_value, std_value, threshold)
    # Create a binary mask based on the threshold
    mask = tensor <= threshold
    return mask

def rot_update_pose(w2c, timestep, poses):
    R = w2c[:, :3, :3].detach()
    t = w2c[0, :3, 3].detach()
    quat = convert.rotation_matrix_to_quaternion(R.contiguous())[0].detach()
    with torch.no_grad():
        poses.pose_param_net.r[..., timestep] = quat
        poses.pose_param_net.t[..., timestep] = t

def create_random_mask(H, W, alpha):
    total_pixels = H * W
    true_pixels = int(alpha * total_pixels)

    # Create a tensor with 'true_pixels' ones and the rest zeros
    mask = torch.cat((torch.ones(true_pixels), torch.zeros(total_pixels - true_pixels)))

    # Shuffle the tensor to randomize the ones and zeros positions
    mask = mask[torch.randperm(total_pixels)]

    # Reshape the tensor back to the original image shape
    mask = mask.view(1, H, W).bool()

    return mask
    
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution=None):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # torch.cuda.set_device(torch.device("cuda:0"))
