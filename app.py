import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import gradio as gr
from scripts.sprite_rotator import SpriteRotator
from palette import apply_palette_and_binarize

def load_config():
    """加载配置文件"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}

def load_model(config, device):
    """从配置中加载模型"""
    model_path = config.get('misc', {}).get('model_path', '')
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    model_file = os.path.join(model_path, 'model.pth')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    # 创建模型实例
    model = SpriteRotator(
        in_channels=4,  # RGBA图像
        base_channels=32,
        embed_dim=128
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    
    print(f"已加载模型: {model_file}")
    return model


def process_image(image_path, source_direction, target_direction, model, device):
    """处理图像并保存结果到原路径"""
    if not image_path or not os.path.exists(image_path):
        return f"错误: 文件不存在 - {image_path}"
    
    # 方向角度映射
    direction_to_angle = {
        "向下": 0,
        "向右": 90,
        "向左": -90,
        "向上": 180
    }
    
    try:
        # 打开推理使用图像
        image = Image.open(image_path).convert('RGBA')
        # 变成tensor，并添加批次维度
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]
        
        # 获取角度
        source_angle = direction_to_angle[source_direction]
        target_angle = direction_to_angle[target_direction]
        source_angle = torch.tensor([source_angle]).float().to(device)
        target_angle = torch.tensor([target_angle]).float().to(device)
        
        # 模型推理
        with torch.no_grad():
            output_tensor = model(image_tensor, source_angle, target_angle)
        
        # 将预测结果转换为PIL图像
        output_img = output_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        output_img = (output_img * 255).astype(np.uint8)
        output_img = Image.fromarray(output_img, mode='RGBA')
        
        # 加工，限制色盘和二值化alpha通道
        output_img = apply_palette_and_binarize(output_img, image)

        # 生成输出路径（在原文件同级目录）
        dirname = os.path.dirname(image_path)
        filename, ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(dirname, f"{filename}_{target_direction}{ext}")
        
        # 保存结果（确保保留透明度）
        output_img.save(output_path, format='PNG')
        
        return f"成功! 已保存至: {output_path}"
    
    except Exception as e:
        return f"处理失败: {str(e)}"

def create_app():
    # 加载配置和模型
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    
    # 创建Gradio界面
    with gr.Blocks(title="像素角色方向转换") as app:
        gr.Markdown("# 像素角色方向转换")
        gr.Markdown("输入本地图像路径，选择源方向和目标方向，然后点击转换。结果将保存在原图像同级目录。")
        
        with gr.Row():
            with gr.Column():
                image_path = gr.Textbox(
                    label="图像路径",
                    placeholder="输入本地图像的完整路径，例如: C:/images/sprite_down.png"
                )
                
                with gr.Row():
                    source_direction = gr.Dropdown(
                        choices=["向下", "向右", "向左", "向上"],
                        value="向下",
                        label="源方向"
                    )
                    target_direction = gr.Dropdown(
                        choices=["向下", "向右", "向左", "向上"],
                        value="向左",
                        label="目标方向"
                    )
                
                convert_btn = gr.Button("转换", variant="primary")
            
            with gr.Column():
                status_message = gr.Textbox(label="处理状态")
    
        # 设置事件处理
        convert_btn.click(
            fn=lambda path, src, tgt: process_image(path, src, tgt, model, device),
            inputs=[image_path, source_direction, target_direction],
            outputs=[status_message]
        )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()