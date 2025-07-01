from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import numpy as np

class SpriteDataLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 角度映射表
        self.direction_to_angle = {
            "down": 0,
            "right": 90,
            "left": -90,
            "up": 180
        }
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        
        # 获取所有角色文件夹
        character_folders = [d for d in os.listdir(self.root_dir) 
                            if os.path.isdir(os.path.join(self.root_dir, d))]
        
        # 为每个角色生成样本
        for character_folder in character_folders:
            character_path = os.path.join(self.root_dir, character_folder)
            
            # 按方向分组图像
            direction_images = {}
            for img_name in os.listdir(character_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    # 从文件名中提取方向信息
                    for direction in self.direction_to_angle.keys():
                        if f"_{direction}." in img_name:
                            if direction not in direction_images:
                                direction_images[direction] = []
                            direction_images[direction].append(os.path.join(character_path, img_name))
                            break
            
            # 为每个基础方向生成样本
            for base_dir, base_imgs in direction_images.items():
                base_angle = self.direction_to_angle[base_dir]
                
                # 为每个基础图像创建样本
                for base_img_path in base_imgs:
                    # 为每个目标方向创建样本
                    for target_dir, target_imgs in direction_images.items():
                        if target_dir == base_dir:  # 跳过相同方向
                            continue
                            
                        target_angle = self.direction_to_angle[target_dir]
                        
                        # 为每个目标方向图像创建样本
                        for target_img_path in target_imgs:
                            # 确保基础图像和目标图像是同一角色的不同方向
                            if self._is_same_character(base_img_path, target_img_path):
                                samples.append({
                                    'base_img_path': base_img_path,
                                    'target_img_path': target_img_path,
                                    'base_angle': base_angle,
                                    'target_angle': target_angle
                                })
        
        print(f"从 {self.root_dir} 加载了 {len(samples)} 个图像对")
        return samples
    
    def _is_same_character(self, img_path1, img_path2):
        """检查两个图像是否属于同一角色"""
        name1 = os.path.basename(img_path1).split('_')[0]
        name2 = os.path.basename(img_path2).split('_')[0]
        return name1 == name2
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载基础图像和目标图像
        base_img = Image.open(sample['base_img_path']).convert('RGBA')
        target_img = Image.open(sample['target_img_path']).convert('RGBA')
        
        # 应用图像变换
        if self.transform:
            base_img = self.transform(base_img)
            target_img = self.transform(target_img)
        
        return {
            'base_image': base_img,
            'target_image': target_img,
            'base_angle': torch.tensor(sample['base_angle']).float(),
            'target_angle': torch.tensor(sample['target_angle']).float()
        }