import os 

from random import randint
from configs.base import CFG
from common.scene_tools import *
from torch.utils.data import Dataset
from dataset.camera import cameraList_from_camInfos

class SceneDataset(Dataset):
    def __init__(self, train: bool = True):
        super().__init__()
        assert os.path.exists(os.path.join(CFG.data_dir, "sparse")), f"sparse dir is not found in {CFG.data_dir}"
        # type SceneInfo
        self.scene_info = readColmapSceneInfo(os.path.join(CFG.data_dir))
        self.cameras = cameraList_from_camInfos(self.scene_info.train_cameras,
                    CFG.resolution_scale, not train)
        self.cameras_extent = self.scene_info.nerf_normalization["radius"]
        self.viewpoint_stack = self.cameras.copy()
        self.viewpoint_indices = list(range(len(self.viewpoint_stack)))

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        # idx not used
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.cameras.copy()
            self.viewpoint_indices = list(range(len(self.viewpoint_stack)))
        rand_idx = randint(0, len(self.viewpoint_indices) - 1)
        viewpoint_cam = self.viewpoint_stack.pop(rand_idx)
        vind = self.viewpoint_indices.pop(rand_idx)

        return viewpoint_cam
