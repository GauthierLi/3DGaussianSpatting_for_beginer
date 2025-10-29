"""
    combine train, validate, interactive visualization
"""
import os
import torch
import argparse
import subprocess
import numpy as np

from tqdm import tqdm
from typing import List
from random import randint
from configs.base import CFG
from common.render import render
from losses.ssim_loss import l1_loss, ssim
from dataset.scene_dataset import SceneDataset
from model.gaussian_models import GaussianModels, SPARSE_ADAM_AVAILABLE

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


def install_requirements():
    def _excute(cmd: List[str], exec_dir: str = None):
        try:
            print(cmd)
            proc = subprocess.run(cmd,
                        stderr=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        text=True,
                        cwd=exec_dir,
                        bufsize=1)
            for line in proc.stdout.splitlines():
                print(line)

        except Exception as e:
            print(f"Error installing requirements: {e}")
    install_requirements_cmd = [
        "pip", "install", "-r", "requirements.txt"
    ]
    _excute(install_requirements_cmd)

    modules = os.listdir("submodules")
    for module in modules:
        exec_dir = os.path.join("submodules", module)
        module_install_cmd = ["python", "setup.py", "develop", "--user"]
        _excute(module_install_cmd, exec_dir=exec_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Train, validate, and visualize 3dGS models.")
    parser.add_argument("--install", action="store_true", default=False, help="Install requirements.")
    parser.add_argument("--mode", type=str, choices=["train", "validate"], default="train", help="Operation mode.")
    return parser.parse_args()

def init():
    save_dir = CFG.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

def train():
    dataset = SceneDataset()
    gaussian_model = GaussianModels(sh_degrees=CFG.sh_degrees, optimizer_type=CFG.optimizer_type)
    gaussian_model.create_from_pcd(dataset.scene_info.point_cloud,
                                    dataset.scene_info.train_cameras,
                                    dataset.cameras_extent)
    gaussian_model.training_setup()
    bg_color = [1, 1, 1] if CFG.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    use_sparse_adam = CFG.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    for iteration in tqdm(range(0, CFG.iterations + 1)):
        viewpoint_cam = dataset[iteration]
        gaussian_model.update_learning_rates(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussian_model.oneupSHdegree()

        # Render
        bg = torch.rand((3), device="cuda") if CFG.random_background else background
        render_pkg = render(viewpoint_cam, gaussian_model, bg, use_trained_exp=CFG.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - CFG.lambda_dssim) * Ll1 + CFG.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        gaussian_model.exposure_optimizer.step()
        gaussian_model.exposure_optimizer.zero_grad(set_to_none=True)
        if use_sparse_adam:
            visible = radii > 0
            gaussian_model.optimizer.step(visible, radii.shape[0])
            gaussian_model.optimizer.zero_grad(set_to_none=True)
        else:
            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none=True)

        if iteration % 100 == 0:
            import cv2 
            im_np = render_pkg['render'].permute(1, 2, 0).detach().cpu().numpy()
            depth_np = render_pkg['depth'].permute(1, 2, 0).detach().cpu().numpy()
            im_np = (im_np * 255).astype("uint8")[:,:,::-1]
            depth_np = (depth_np * 255).astype("uint8")[:,:,::-1]
            depth_np = np.repeat(depth_np, 3, axis=2)
            im_show = np.hstack([im_np, depth_np])
            h, w, c = im_show.shape
            scale = 2.
            im_show = cv2.resize(im_show, (int(w//scale), int(h//scale)))
            cv2.imshow("frame", im_show)
            cv2.waitKey(1)

def main(args: argparse.Namespace):
    if args.install:
        install_requirements()
    init()
    if args.mode == "train":
        train()

if __name__ == "__main__":
    args = parse_args()
    main(args)
