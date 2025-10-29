import torch
import numpy as np

from common.scene_tools import eval_sh
from configs.base import CFG
from model.gaussian_models import GaussianModels
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render(viewpoint_camera,
            pc : GaussianModels,
            bg_color : torch.Tensor,
            scaling_modifier = 1.0,
            separate_sh = False,
            override_color = None,
            use_trained_exp=False):
    """
    Render the scene 
    """
    # 此处加0是为了产生梯度？
    screenspace_points = torch.zeros_like(pc.get_xyz,
                                        device=pc.get_xyz.device,
                                        requires_grad=True) + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        pass 
    tanfovx = np.tan(viewpoint_camera.FoVx / 2)
    tanfovy = np.tan(viewpoint_camera.FoVy / 2)
    # 光栅化
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.activate_sh_dedgrees,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=CFG.debug,
        antialiasing=CFG.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # 这里的3d和2d means都是N,3的shape，也许means2D是齐次z的
    means3D = pc.get_xyz 
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if CFG.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        if CFG.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degrees+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeate(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.activate_sh_dedgrees, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        # [3, 1280, 720],半径？ [36579], [1, 1280, 720]
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

    # Apply exposure to rendered image (training only) 这里的exposure是什么？
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
        
    # 在渲染的时候会将视锥外或者半径为0的点剔除掉
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image
    }
    return out
    
