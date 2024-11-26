import os
import torch

import torchvision
from gaussian_renderer import render, render_custom, inference
import sys
import random
from vis.utils import get_server
from vis.viewer import GSViewer
from torchvision.utils import save_image
from scene import GaussianModel
from scene.pose_optimizer import PoseModel, projection_flow_loss
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
import wandb
import cv2
import numpy as np
from vis.annotation import add_label
from utils.geometry_utils import align_pose, rotation_matrix_to_euler_degrees, euler_degrees_to_rotation_matrix
import time
import os.path as osp
from datetime import datetime
import shutil
from utils.common_utils import visualize_depth
from utils.server_utils import get_server
from vis.visualizer import hcat, vcat, prep_image, log_image
from vis.layout import add_border
from utils.loss_utils import rgb_loss_func, pearson_depth_loss, local_pearson_loss
from utils.general_utils import safe_state, adaptive_thresholding, rgb_evaluation
class FreeSurGS:
    def __init__(self, args, dataset, opt, pipe):
        
        prepare_output_and_logger(dataset)
        backup_code(dataset.model_path)
        self.gaussians = GaussianModel(dataset.sh_degree, opt)
        self.poses = PoseModel(dataset, device="cuda")
        timestep = 0

        self.poses.record_data['pred_w2c'][timestep] = np.eye(4)
        self.poses.record_data['pred_depths'][0, ...] = self.poses.record_data['monodeps'][timestep].cuda().float().cpu()

        self.gaussians.initialize_first_timestep(0, self.poses)
        print("initialize gs with points num: ", len(self.gaussians.params['_xyz']))
        self.gaussians.training_setup(opt)

        self.h, self.w = self.poses.record_data[
            'image_height'], self.poses.record_data['image_width']
        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)

        self.ema_loss_for_log = 0.0
        first_iter = 0
        self.progress_bar = tqdm(range(first_iter, opt.iterations),
                                 desc="Training progress")
        
        self.tracking_iter = 50
        self.mapping_iter = 30
        print(f"start training with tracking {self.tracking_iter} iters, mapping {self.mapping_iter} iters")
        
        self.mapping_window_size = 3
        self.iter_optimize = 0
        self.mapping_interval = 1
        self.epipolar_thres = 10
        self.visible_mask = None
        self.iteration = first_iter
        self.first_iter = first_iter
        self.epipolar_loss = 0
        self.densifi_interval = 100
        self.logging_interval = 30
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.args = args
        self.lambda_diffusion = 0.001 
        self.step_ratio = 0.99
        self.lambda_reg = 0.1
        self.SDS_freq = 0.1
        self.loss_weight_mapping = {
                    "rgb": 5.0,
                    "depth": 1.0,
                    "flow": 1.0,
                    "iso": 10.0,
                }
        self.loss_weight_tracking = {
                    "rgb": 1.0,
                    "flow": 0.1,
                }
        self.viewer = None
        self.init_vis_rot = None
        self.init_vis_trans = None
        if args.visualize:
            server = get_server(port=args.port)
            self.viewer = GSViewer(
                server, self.render_fn, self.poses.num_cams, dataset.model_path, mode="training")
           
        print('Initialize nodes with Random point cloud.')
        self.num_rays_per_step = self.poses.H * self.poses.W * 3 
        self.kf_overlap = 0.9
        self.kf_translation = 0.1
        self.kf_min_translation = 0.02
        self.occ_aware_visibility = {}
        from utils.loss_utils import ScaleAndShiftInvariantLoss
        self.scale_variant_dep_loss = ScaleAndShiftInvariantLoss()
        if args.start_checkpoint is not None:
            (model_params, first_iter) = torch.load(args.start_checkpoint)
            self.gaussians.restore(model_params, opt)
            pose_ckt = args.start_checkpoint.replace('chkpnt', 'poses')
            (pose_params, first_iter) = torch.load(pose_ckt)
            self.poses.restore(pose_params)
            print("loading checkpoints: ", args.start_checkpoint, pose_ckt)
            self.iteration = first_iter

        if args.test == True:
            self.validation()
        else:
            if args.start_checkpoint is not None:
                self.global_run()
            else:
                self.progressive_run()
                self.global_run()

    def render_fn(self, camera_state, img_wh):
        initialize_w2c = self.poses.record_data['pred_w2c'][0]
        # W, H = img_wh
        W = 2048 
        H = 1200
        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]]
        )
        visualize_data = {
            'K': K,
            'W': W,
            'H': H,
        }
        visualizer_cam = self.poses.setup_camera(initialize_w2c, visualize_data=visualize_data)
        rot_xyz = rotation_matrix_to_euler_degrees(camera_state.c2w[:3, :3])
        if self.init_vis_rot is None:
            self.init_vis_rot = rot_xyz
            self.init_vis_trans = camera_state.c2w[:3, 3]
        cur_rot = rot_xyz - self.init_vis_rot
        cur_trans = camera_state.c2w[:3, 3] - self.init_vis_trans
        rotation_mat = euler_degrees_to_rotation_matrix(cur_rot* 0.1)
        w2c = torch.eye(4).float().cuda()
        w2c[:3, :3] = torch.from_numpy(rotation_mat).float().cuda()
        w2c[:3, 3] = torch.from_numpy(cur_trans* 0.1).float().cuda()

        img = render_custom(visualizer_cam, self.poses, w2c, self.gaussians)["render"].permute(1, 2, 0)
        img = torch.clamp(img, 0., 1.0)
        return (img.detach().cpu().numpy() * 255.0).astype(np.uint8)

    def tracking(self, timestep):
        rgb_losses = []
        flow_losses = []
        progress_bar = tqdm(range(1), desc=f"Tracking Time Step: {timestep}")
        if timestep > 1:
            _, sampson_dist = self.poses.compute_epipolar_loss(
                timestep - 2, timestep - 1)
            
            rigid_mask = sampson_dist < adaptive_thresholding(sampson_dist)
        else:
            rigid_mask = torch.ones((self.h, self.w)).bool().cuda()
            sampson_dist = torch.ones((self.h, self.w)).cuda()
        for iter in range(self.tracking_iter):
            render_pkg = render(self.poses,
                            timestep,
                            self.gaussians,
                            gs_grad=False,
                            cam_grad=True)

            image = render_pkg["render"]
            gt_image = self.poses.record_data['colors'][timestep].cuda()
            mask = render_pkg["render_dep"] > 0
            mask = mask  * rigid_mask
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

            rgb_loss = self.loss_weight_tracking['rgb'] * rgb_loss_func(image, gt_image, mask=mask)
            
            flow_mask = rigid_mask
            flow_loss = self.loss_weight_tracking['flow'] * projection_flow_loss(timestep, self.poses.record_data['pred_depths'][timestep-1:timestep, ...], \
                self.poses.record_data['pred_w2c'][timestep-1], render_pkg['render_w2c'], self.poses.record_data, flow_mask)

            loss = flow_loss + rgb_loss

            loss.backward()
            self.poses.scheduler_eval_pose.step()
            self.iter_end.record()
            rgb_losses.append(rgb_loss.item())
            flow_losses.append(flow_loss.item())
            with torch.no_grad():
                self.poses.optimizer.step()
                self.poses.optimizer.zero_grad(set_to_none=True)
            progress_bar.set_postfix({
                "Time-Step": timestep,
                "Loss": f"{loss:.7f}"
            })
            progress_bar.update(1)
        progress_bar.close()
        return {
            'render_depth': render_pkg['render_dep'],
            'rgb_loss': rgb_losses,
            'flow_loss': flow_losses,
            'sampson_dist': sampson_dist.detach().cpu(),
            'rigid_mask': rigid_mask.detach().cpu().numpy(),
            'flow_mask': flow_mask,
            'render_pkg': render_pkg
        }


    def mapping(self, cur_timestep, mapping_iter, progressive=False, iteration=None):
        if progressive == True and cur_timestep != 0:
            view = 2
        else:
            view = 1
        if progressive == True:
            progress_bar = tqdm(range(mapping_iter), desc=f"Mapping Time Step: {cur_timestep}")
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        for iter in range(mapping_iter):
            viewspace_point_tensor_all = []
            visibility_filter_all = []
            radii_acm_all = []
            
            loss = 0
            if self.viewer is not None:
                while self.viewer.state.status == "paused":
                    time.sleep(0.1)
                    print(self.viewer.state.status)
                self.viewer.lock.acquire()
            _tic = time.time()

            self.iter_start.record()
            self.iteration += 1
            for i in range(view):
                
                if view == 2:
                    if i == 0:
                        timestep = random.choice(self.poses.keyframe_list)
                    else:
                        timestep = cur_timestep
                else:
                    timestep = cur_timestep
                render_pkg = render(self.poses,
                            timestep,
                            self.gaussians,
                            gs_grad=True,
                            cam_grad=False)
                image = render_pkg["render"]
                
                # Loss
                gt_image = self.poses.record_data['colors'][timestep].cuda()
                rgb_loss = rgb_loss_func(image, gt_image) * self.loss_weight_mapping['rgb']
                mono_dep = self.poses.record_data['monodeps'][timestep:timestep+1].float().cuda()
                pearson_dep_loss = pearson_depth_loss(mono_dep[0], render_pkg["render_dep"])
                lp_loss = local_pearson_loss(mono_dep[0], render_pkg["render_dep"], 128, 0.5)
                dep_loss = (pearson_dep_loss * 0.05 + lp_loss * 0.15) 
                loss += rgb_loss + dep_loss
                if i == 0:
                    viewspace_point_tensor_all.append(render_pkg["viewspace_points"])
                    visibility_filter_all.append(render_pkg["visibility_filter"])
                    radii_acm_all.append(render_pkg["radii"])
            
            loss.backward()
            self.iter_end.record()
            num_rays_per_sec = self.num_rays_per_step / (time.time() - _tic)
            with torch.no_grad():
                self.densification(timestep, self.dataset, self.opt, self.iteration, viewspace_point_tensor_all, \
                        visibility_filter_all, radii_acm_all)
                
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                
                if args.log:
                    wandb.log({
                        'mapping/rgb_loss': rgb_loss,
                        'mapping/dep_loss': dep_loss,
                        'mapping/loss': loss
                    })
                if self.viewer is not None:
                    self.viewer.lock.release()
                    self.viewer.state.num_train_rays_per_sec = num_rays_per_sec * 4
                    if self.viewer.mode == "training":
                        self.viewer.update(self.iteration, self.num_rays_per_step)
                if progressive == True:
                    progress_bar.set_postfix({
                        "Time-Step": timestep,
                        "rgb_loss": f"{rgb_loss:.7f}",
                        "dep_loss": f"{dep_loss:.7f}"
                    })
                    progress_bar.update(1)
        if progressive == True:
            progress_bar.close()
        return render_pkg

    def densification(self, timstep, dataset, opt, iteration, viewspace_point_tensor, visibility_filter, radii_acm_all, densify=False, grow=True, gs_mask=None):
        for i in range(len(viewspace_point_tensor)):
            self.gaussians.variables['max_radii2D'][visibility_filter[i]] = torch.max(
                            self.gaussians.variables['max_radii2D'][visibility_filter[i]],
                            radii_acm_all[i][visibility_filter[i]],
                        )
            self.gaussians.add_densification_stats(viewspace_point_tensor[i], visibility_filter[i])

        if iteration % 300 == 0 and iteration < 15000:
            pre_num = len(self.gaussians.params['_xyz'])
            size_threshold = 20 if self.iteration > 4000 else None
            max_grad = opt.densify_grad_threshold
            min_opacity = 0.05
            max_screen_size = size_threshold
            self.gaussians.densify_and_prune(max_grad, min_opacity, max_screen_size, gs_mask)
            print(f"At {timstep} Densification from {pre_num} to {len(self.gaussians.params['_xyz'])}!!!")
            
        if iteration % 3000 == 0:
            print("reset opacity!!")
            self.gaussians.reset_opacity()

    def progressive_run(self):
        self.poses.pose_param_net.train()
        self.poses.initialize_tracking_optimizer(self.tracking_iter)
        
        for i in range(self.poses.pose_param_net.num_cams):
            self.timestep = i
            self.gaussians.update_learning_rate(self.iteration)

            if self.timestep > 0:
                if self.timestep > 1:
                    self.poses.initialize_pose(self.timestep, pnp=False)
                self.poses.initialize_tracking_optimizer(
                    self.tracking_iter)
                self.poses.optimizer.zero_grad(set_to_none=True)
                tracking_pkg = self.tracking(timestep=self.timestep)

            if i % self.mapping_interval == 0 and i in self.poses.i_train:
                if self.iteration % 1000 == 0:
                    self.gaussians.oneupSHdegree()
                    self.gaussians.cam._replace(
                        sh_degree=self.gaussians.active_sh_degree)
                mapping_iter = 200 if i == 0 else self.mapping_iter
                render_pkg = self.mapping(self.timestep, mapping_iter=mapping_iter, progressive=True)
                render_color = render_pkg['render'].detach().cpu()
                gt_color = self.poses.record_data['colors'][self.timestep]
                self.poses.record_data['pred_depths'][self.timestep,...] = render_pkg['render_dep'][0].detach().cpu().float()
                self.poses.record_data['pred_colors'][self.timestep,...] = render_color.float()
                self.poses.keyframe_list.append(self.timestep)
                
            if self.timestep % self.logging_interval == 0:

                if self.args.log and self.timestep > 0:
                    
                    vis_gt_dep = visualize_depth(self.poses.record_data['monodeps'][self.timestep].detach().cpu())
                    vis_render_dep = visualize_depth(render_pkg['render_dep'][0].detach().cpu())
                    comparison = hcat(
                        add_label(gt_color, "GT rgb"),
                        add_label(render_color, "Rendered rgb"),
                        add_label(vis_gt_dep, "GT depth"),
                        add_label(vis_render_dep, "Rendered depth"),
                        )
                    metrics = log_image(
                        "comparison",
                        [prep_image(add_border(comparison))],
                        step=self.iteration,
                        caption=["surgical"],
                    )
                    wandb.log(dict(metrics, **{"progressive_train/global_step": self.iteration}))
                    
        
            

        # Saving Model
        torch.save(
            (self.gaussians.capture(), self.iteration),
            self.dataset.model_path + "/chkpnt" + str(self.iteration) + ".pth")
        torch.save(
            (self.poses.capture(), self.iteration),
            self.dataset.model_path + "/poses" + str(self.iteration) + ".pth")
                    
    def global_run(self):
        self.gaussians.initialize_optimizer()
        
        for iter in range(self.first_iter, self.opt.iterations + 1):
            timestep = random.choice(self.poses.i_train)#self.poses.num_cams)))
            if iter % 1000 == 0:
                self.gaussians.oneupSHdegree()
                self.gaussians.cam._replace(
                    sh_degree=self.gaussians.active_sh_degree)
            self.gaussians.update_learning_rate(iter)
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            render_pkg = self.mapping(timestep, mapping_iter=1, progressive=False, iteration=iter)
            render_color = render_pkg['render'].detach().cpu()
            gt_color = self.poses.record_data['colors'][timestep]
            self.poses.record_data['pred_depths'][timestep,...] = render_pkg['render_dep'][0].detach().cpu().float()
            self.poses.record_data['pred_colors'][timestep,...] = render_pkg['render'].detach().cpu()
            self.progress_bar.set_postfix(
                {"iter": iter, "gs pts: ": len(self.gaussians.params['_xyz'])})
            self.progress_bar.update(1)
            if iter == self.opt.iterations:
                self.progress_bar.close()

            # Evaluation of rendering performance on test set
            if iter % 5000 == 0:
                gt_rgb = []
                pred_rgb = []
                for index in tqdm(self.poses.i_test):
                    render_pkg = render(self.poses,
                                    index,
                                    self.gaussians,
                                    gs_grad=False,
                                    cam_grad=False)
                    render_color = render_pkg['render'].detach().cpu()
                    gt_color = self.poses.record_data['colors'][index]
                    vis_gt_dep = visualize_depth(self.poses.record_data['monodeps'][index].detach().cpu())
                    vis_render_dep = visualize_depth(render_pkg['render_dep'][0].detach().cpu())
                    comparison = hcat(
                        add_label(gt_color, "GT rgb"),
                        add_label(render_color, "Rendered rgb"),
                        add_label(vis_gt_dep, "GT depth"),
                        add_label(vis_render_dep, "Rendered depth"),
                       )
                    save_image(comparison, osp.join(self.dataset.model_path,f'test_{iter}_{index}.png'))
                    pred_rgb.append(render_color.cpu())
                    gt_rgb.append(gt_color.cpu())
                pred_rgb = torch.clamp(torch.stack(pred_rgb), 0.0, 1.0)
                gt_rgb = torch.clamp(torch.stack(gt_rgb), 0.0, 1.0)
                psnr, ssim, lpips_ = rgb_evaluation(pred_rgb.numpy(), gt_rgb.numpy())
                
                if self.args.log:
                    wandb.log({
                        'global_train/psnr': psnr,
                        'global_train/ssim': ssim,
                        'global_train/lpips_': lpips_
                    })
                    
   
        

            if iter % 5000 == 4999:
                torch.save(
                    (self.gaussians.capture(), iter),
                    self.dataset.model_path + "/chkpnt" + str(iter) + ".pth")
                torch.save(
                    (self.poses.capture(), iter),
                    self.dataset.model_path + "/poses" + str(iter) + ".pth")


    def validation(self):
        
        test_folder = osp.join(self.dataset.model_path, "test_results")
        os.makedirs(test_folder, exist_ok=True)
        eval_pose(self.poses)
        pred_rgb = []
        gt_rgb = []
        for index in tqdm(self.poses.i_test):
            render_pkg = render(self.poses,
                            index,
                            self.gaussians,
                            gs_grad=False,
                            cam_grad=False)
            render_color = render_pkg['render'].detach().cpu()
            gt_color = self.poses.record_data['colors'][index]
            vis_gt_dep = visualize_depth(self.poses.record_data['monodeps'][index].detach().cpu())
            vis_render_dep = visualize_depth(render_pkg['render_dep'][0].detach().cpu())
            gt_color = torch.clamp(gt_color, 0.0, 1.0).detach()
            render_color = torch.clamp(render_color, 0.0, 1.0).detach()
            
            gt_color = gt_color.cpu()
            image_vis = render_color.cpu()
            pred_rgb.append(image_vis)
            gt_rgb.append(gt_color)
            comparison = hcat(
                add_label(gt_color, "GT rgb"),
                add_label(render_color, "Rendered rgb"),
                add_label(vis_gt_dep, "GT depth"),
                add_label(vis_render_dep, "Rendered depth"),
            )
            save_image(comparison, osp.join(test_folder, f"validation_{index}.png"))

            

        gt_rgb = np.stack(gt_rgb, axis=0)
        pred_rgb = np.stack(pred_rgb, axis=0)
        psnr, ssim, lpips_ = rgb_evaluation(gt_rgb,
                                            pred_rgb)
        if args.log:
            wandb.log({
                'validation/psnr': psnr,
                'validation/ssim': ssim,
                'validation/lpips_': lpips_,
            })


def eval_pose(poses):
    data_ind = poses.record_data['data_ind']
    i = 0
    all_metric = np.array([0.0, 0.0, 0.0])
    traj = []
    gt_traj = []
    for key, value in poses.record_data['gt_poses'].items():
        pred_w2c = torch.from_numpy(
            poses.record_data['pred_w2c'])[data_ind[i]:data_ind[i + 1]]
        gt_poses = value
        c2ws_est_to_draw_align2cmp, metrics = align_pose(pred_w2c, gt_poses)
        all_metric += np.array(metrics) * poses.record_data['weights'][i]
        traj.append(c2ws_est_to_draw_align2cmp)
        gt_traj.append(gt_poses)
        i += 1

    print("all metrics: ", "rpe_trans: {0:.3f}".format(all_metric[0]),'&' "rpe_rot: {0:.3f}".format(all_metric[1]), \
            '&', "ape: {0:.3f}".format(all_metric[2]))
    if args.log:
        wandb.log({
            'validation/rpe_trans': all_metric[0],
            'validation/rpe_rot': all_metric[1],
            'validation/ape': all_metric[2],
        })


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def backup_code(work_dir):
    root_dir = osp.abspath(osp.join(osp.dirname(__file__)))
    tracked_dirs = [osp.join(root_dir, dirname) for dirname in ["flow3d", "scripts"]]
    dst_dir = osp.join(work_dir, "code", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    for tracked_dir in tracked_dirs:
        if osp.exists(tracked_dir):
            shutil.copytree(tracked_dir, osp.join(dst_dir, osp.basename(tracked_dir)))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations",
                        nargs="+",
                        type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--save_iterations",
                        nargs="+",
                        type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations",
                        nargs="+",
                        type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--runner_name", type=str, default="free-surgs")
    parser.add_argument('--render_only',
                        action='store_true',
                        help='use small model')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr',
                        action='store_true',
                        help='use efficent correlation implementation')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path, "data type: ", args.data_type)
    # Initialize Wandb Logging
    if args.log == True:
        run = wandb.init(project="3DGS", group=args.runner_name)
        wandb.config.update(args)
        wandb_path = osp.join(args.model_path, "wandbs")
        os.makedirs(wandb_path, exist_ok=True)
        run.config.data = wandb_path
        run.name = args.runner_name
    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    slam = FreeSurGS(args, lp.extract(args), op.extract(args),
                     pp.extract(args))
    
    print("\nAll complete.")
