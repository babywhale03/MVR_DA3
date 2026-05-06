# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RealBlur_R video deblurring dataset implementation.
"""

import glob
import os
from typing import Dict as TDict

import numpy as np
from addict import Dict
from PIL import Image

from depth_anything_3.bench.dataset import Dataset
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY


REALBLUR_R_DATA_ROOT = "/mnt/dataset1/MV_Restoration/restormer_benchmark/RealBlur_R"


@MV_REGISTRY.register(name="realblur_r")
@MONO_REGISTRY.register(name="realblur_r")
class RealBlurR(Dataset):
    """
    RealBlur_R video deblurring dataset wrapper.
    
    Each scene contains sequential frames (blurred input + sharp target).
    """

    data_root = REALBLUR_R_DATA_ROOT
    
    # Extract unique scene names from input folder
    @property
    def SCENES(self):
        input_dir = os.path.join(self.data_root, "input")
        if not os.path.exists(input_dir):
            return []
        files = os.listdir(input_dir)
        # Extract scene names by removing frame numbers
        scenes = set()
        for f in files:
            if f.endswith('.png'):
                # Remove trailing "-XXXXXX.png"
                scene_name = f.rsplit('-', 1)[0]
                scenes.add(scene_name)
        return sorted(list(scenes))

    def get_data(self, scene: str) -> Dict:
        """
        Collect frame sequences for a scene.

        Args:
            scene: Scene identifier

        Returns:
            Dict with:
                - image_files: List[str] - paths to blurred frames (input/)
                - target_files: List[str] - paths to sharp frames (target/)
        """
        input_dir = os.path.join(self.data_root, "input")
        target_dir = os.path.join(self.data_root, "target")
        
        # Find all frames for this scene
        input_files = sorted(glob.glob(os.path.join(input_dir, f"{scene}-*.png")))
        target_files = sorted(glob.glob(os.path.join(target_dir, f"{scene}-*.png")))
        
        out = Dict({
            "image_files": input_files,
            "target_files": target_files,
            "scene": scene,
        })
        
        return out

    def __len__(self):
        return len(self.SCENES)

    def eval_pose(self, scene: str, result_path: str) -> TDict[str, float]:
        """
        Camera pose estimation not applicable for video deblurring.
        """
        return {}

    def eval3d(self, scene: str, fuse_path: str) -> dict:
        """
        3D reconstruction evaluation not applicable for video deblurring.
        """
        return {}
