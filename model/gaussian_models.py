import torch
import numpy as np

from torch import nn
from configs.base import CFG
from simple_knn._C import distCUDA2
from common.scene_tools import BasicPointCloud, build_covariance_from_scaling_rotation, inverse_sigmoid, RGB2SH, get_expon_lr_func

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception as e:
    print(e)
    SPARSE_ADAM_AVAILABLE = False


class GaussianModels:
    def __init__(self, sh_degrees: int, optimizer_type: str = "default"):
        self.optimizer_type = optimizer_type
        self.activate_sh_dedgrees = 0
        self.max_sh_degrees = sh_degrees
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.spatial_lr_scale = 0

        self.setup_functions()

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def training_setup(self):
        self.percent_dense = CFG.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": CFG.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [self._features_dc], "lr": CFG.feature_lr, "name": "f_dc"},
            {
                "params": [self._features_rest],
                "lr": CFG.feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self._opacity], "lr": CFG.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": CFG.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": CFG.rotation_lr, "name": "rotation"},
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=CFG.position_lr_init,
            lr_final=CFG.position_lr_final,
            lr_delay_mult=CFG.position_lr_delay_mult,
            max_steps=CFG.position_lr_max_steps,
        )

        self.exposure_scheduler_args = get_expon_lr_func(
            lr_init=CFG.exposure_lr_init,
            lr_final=CFG.exposure_lr_final,
            lr_delay_mult=CFG.exposure_lr_delay_mult,
            lr_delay_steps=CFG.exposure_lr_delay_steps,
            max_steps=CFG.iterations,
        )

    def update_learning_rates(self, iteration: int):
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group["lr"] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def oneupSHdegree(self):
        if self.activate_sh_dedgrees < self.max_sh_degrees:
            self.activate_sh_dedgrees += 1

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def create_from_pcd(
        self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float
    ):
        self.sparse_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.Tensor(np.asarray(pcd.points)).float().cuda()
        # pcd.colors => N, 3 ; fused_color => N, 3
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # 第一个分量表示本征颜色，其他分量初始化为0
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degrees + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {
            cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)
        }
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
