import torch

from torch import nn
from PIL import Image
from typing import List, Tuple, Union
from configs.base import CFG
from common.scene_tools import *

# ===================================== camera =====================================
class Camera(nn.Module):
    def __init__(self,
                 resolution: Union[List[int], Tuple[int]],
                 colmap_id: int,
                 R, T, FoVx, FoVy,
                 image: Image.Image, 
                 image_name: str,
                 uid: int,
                 trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.data_device = torch.device(data_device)
        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform @ self.projection_matrix).cuda()
        self.world_view_transform = self.world_view_transform.cuda()
        self.projection_matrix = self.projection_matrix.cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3]
            
def loadCam(id, cam_info, resolution_scale, is_test_dataset):
    image = Image.open(cam_info.image_path)
    invdepthmap = None
    
    orig_w, orig_h = image.size
    if CFG.resolution in [1,2,4,8]:
        resolution = round(orig_w/(resolution_scale * CFG.resolution)), round(orig_h/(resolution_scale * CFG.resolution))
    else:
        if CFG.resolution == -1: # 如果不指定分辨率，则将最长边缩放到1600，如果小于1600则不做处理
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / CFG.resolution
        
        
        scale = float(global_down) * float(resolution_scale) # 降采样率
        resolution = (int(orig_w/scale)), int((orig_h/scale))
    return Camera(resolution, colmap_id=cam_info.uid,
                  R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=image, image_name=cam_info.image_path,
                  uid=id,data_device=CFG.data_device,
                  train_test_exp=CFG.train_test_exp,
                  is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

def cameraList_from_camInfos(cam_infos, resolution_scale, is_test_dataset):
    cameras = []
    for i, cam_info in enumerate(cam_infos):
        cam = loadCam(i, cam_info, resolution_scale, is_test_dataset)
        cameras.append(cam)
    return cameras
