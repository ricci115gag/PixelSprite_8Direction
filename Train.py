import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image as PILImage
from scripts.sprite_rotator import SpriteRotator
from sprite_dataloader import SpriteDataLoader

# ===== 工具函数 =====

def binarize_alpha(alpha_tensor, threshold=0.5):
    """使用阶跃函数将alpha通道二值化为0或1"""
    return (alpha_tensor >= threshold).float()

# ===== 损失函数 =====

class AlphaBinarizationLoss(nn.Module):
    """强制Alpha通道为0或1的损失函数"""
    def __init__(self, alpha_weight=5.0, epsilon=1e-6):
        super(AlphaBinarizationLoss, self).__init__()
        self.alpha_weight = alpha_weight
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        """计算损失时强制alpha通道接近0或1"""
        pred_rgb = pred[:, :3, :, :]
        pred_alpha = pred[:, 3:, :, :]
        target_rgb = target[:, :3, :, :]
        target_alpha = target[:, 3:, :, :]
        
        # 1. 计算RGB损失（仅在目标alpha>0的区域）
        alpha_mask = (target_alpha > 0.5).float()
        rgb_loss = nn.MSELoss()(pred_rgb * alpha_mask, target_rgb * alpha_mask)
        
        # 2. 计算Alpha损失（使用BCE强制接近0或1）
        alpha_loss = nn.BCELoss()(pred_alpha, target_alpha)
        
        # 3. 计算Alpha二值化惩罚（鼓励严格0或1）
        alpha_penalty = torch.mean(pred_alpha * (1 - pred_alpha))
        
        # 4. 组合损失
        total_loss = rgb_loss + self.alpha_weight * (alpha_loss + 0.1 * alpha_penalty)
        
        return total_loss

# ===== 训练函数 =====

def train_model(config_path='config.yaml'):
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型和数据加载器
    model = SpriteRotator(
        in_channels=config['model']['in_channels'],
        base_channels=config['model']['base_channels'],
        embed_dim=config['model']['embed_dim']
    ).to(device)
    
    dataset = SpriteDataLoader(
        root_dir=config['train']['dataset_path'],
        transform=transforms.Compose([
            transforms.Resize((config['train']['h'], config['train']['w'])),
            transforms.ToTensor()
        ])
    )
    
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)
    
    # 模型路径设置
    model_dir = config['misc']['model_path']
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, 'model.pth')
    csv_file = os.path.join(model_dir, 'loss.csv')
    history_plot_file = os.path.join(model_dir, 'loss_curve_history.png')
    current_plot_file = os.path.join(model_dir, 'loss_curve_current.png')
    
    # 加载已有模型（如果存在）
    start_epoch = 0
    all_losses = []
    
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f"已加载模型: {model_file}")
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            start_epoch = len(df)
            all_losses = df['loss'].tolist()
            print(f"继续训练: 从第 {start_epoch} 轮开始")
    
    # 定义损失函数和优化器
    criterion = AlphaBinarizationLoss(alpha_weight=5.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    
    # 训练循环
    end_epoch = start_epoch + config['train']['epochs']
    
    for epoch in range(start_epoch, end_epoch):
        running_loss = 0.0
        model.train()
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), 
                           desc=f"Epoch {epoch+1}/{end_epoch}")
        
        for i, batch in progress_bar:
            # 准备数据
            base_images = batch['base_image'].to(device)
            target_images = batch['target_image'].to(device)
            base_angles = batch['base_angle'].to(device)
            target_angles = batch['target_angle'].to(device)
            
            # 前向传播
            outputs = model(base_images, base_angles, target_angles)
            
            # 计算损失（注意：不要在计算梯度前修改outputs）
            loss = criterion(outputs, target_images)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'batch_loss': loss.item()})
        
        # 计算平均损失
        avg_loss = running_loss / len(dataloader)
        all_losses.append(avg_loss)
        
        # 打印本轮损失
        print(f"Epoch [{epoch+1}/{end_epoch}], Loss: {avg_loss:.6f}")
        
        # 保存模型
        torch.save(model.state_dict(), model_file)
        print(f"模型已保存至: {model_file}")
        
        # 更新CSV日志
        df = pd.DataFrame({
            'epoch': list(range(1, len(all_losses) + 1)),
            'loss': all_losses
        })
        df.to_csv(csv_file, index=False)
        print(f"Loss 日志已更新至: {csv_file}")
    
    # 训练结束后绘制损失图
    if len(all_losses) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(all_losses) + 1), all_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        plt.savefig(history_plot_file)
        print(f"历史损失曲线已保存至: {history_plot_file}")
        
        if start_epoch > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(start_epoch + 1, end_epoch + 1), all_losses[start_epoch:])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'This Training Loss (Epoch {start_epoch+1}-{end_epoch})')
            plt.grid(True)
            plt.savefig(current_plot_file)
            print(f"当前训练损失曲线已保存至: {current_plot_file}")
    
    print("训练完成!")

# ===== 主函数 =====

if __name__ == "__main__":
    # 运行训练
    train_model()