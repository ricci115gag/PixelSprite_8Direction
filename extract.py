import os
import re
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

def crop_image_by_vertices(img, vertices):
    """
    根据四个顶点坐标裁剪图像
    
    参数:
    img: PIL.Image对象
    vertices: 四个顶点坐标 [x1, y1, x2, y2, x3, y3, x4, y4]
    
    返回:
    裁剪后的PIL.Image对象
    """
    # 确保顶点是8个值
    if len(vertices) != 8:
        raise ValueError("顶点参数必须包含8个值: [x1, y1, x2, y2, x3, y3, x4, y4]")
    
    # 提取四个顶点
    points = [(vertices[i*2], vertices[i*2+1]) for i in range(4)]
    
    # 计算包围盒
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    
    # 裁剪图像
    cropped = img.crop((min_x, min_y, max_x, max_y))
    
    return cropped

def extract_sprites(root_dir, output_dir, vertex_params, position_flag, dry_run=False):
    """
    从指定目录结构中提取精灵图像，并按角色分组保存
    
    参数:
    root_dir: 根目录，包含多个以Evc开头的文件夹
    output_dir: 输出目录，用于保存提取的精灵图像
    vertex_params: 裁剪区域的四个顶点坐标 [x1, y1, x2, y2, x3, y3, x4, y4]
    position_flag: 位置标记，用于筛选特定位置的图像
    dry_run: 是否进行试运行（不实际保存文件）
    """
    # 用于存储找到的文件
    sprite_files = []
    
    # 遍历根目录下的所有角色文件夹
    for dir_name in os.listdir(root_dir):
        if dir_name.startswith(("Evc", "Npc", "Ply")) and os.path.isdir(os.path.join(root_dir, dir_name)):
            textures_dir = os.path.join(root_dir, dir_name, "Textures")
            
            # 检查Textures文件夹是否存在
            if os.path.exists(textures_dir) and os.path.isdir(textures_dir):
                # 遍历Textures文件夹中的所有文件
                for file_name in os.listdir(textures_dir):
                    # 筛选符合格式的PNG文件
                    if file_name.endswith(".png") and re.search(r"UI_cl\.png$", file_name):
                        sprite_files.append((os.path.join(textures_dir, file_name), dir_name))
    
    print(f"找到 {len(sprite_files)} 个符合条件的文件")
    
    # 处理每个找到的文件
    for file_path, character_name in tqdm(sprite_files, desc="提取精灵图像"):
        try:
            # 打开图像
            img = Image.open(file_path).convert("RGBA")
            
            # 提取文件名（不含扩展名）用于保存
            file_base_name = os.path.basename(file_path).replace(".png", "")
            
            # 根据顶点参数裁剪图像
            cropped_img = crop_image_by_vertices(img, vertex_params)
            
            # 确定角色输出文件夹
            character_output_dir = os.path.join(output_dir, character_name)
            
            # 创建角色文件夹（如果不存在）
            if not dry_run:
                os.makedirs(character_output_dir, exist_ok=True)
                
                # 保存裁剪后的图像，文件名包含方向信息
                output_path = os.path.join(character_output_dir, f"{file_base_name}_{position_flag}.png")
                cropped_img.save(output_path)
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    print(f"提取完成，文件已保存至 {output_dir}")
    return sprite_files

def visualize_crop(img_path, vertices):
    """可视化裁剪效果（用于调试）"""
    img = Image.open(img_path).convert("RGBA")
    
    # 创建画布
    plt.figure(figsize=(12, 6))
    
    # 显示原图
    plt.subplot(1, 2, 1)
    plt.title("原始图像")
    plt.imshow(img)
    
    # 显示裁剪区域
    plt.subplot(1, 2, 2)
    plt.title("裁剪区域")
    cropped = crop_image_by_vertices(img, vertices)
    plt.imshow(cropped)
    
    plt.tight_layout()
    plt.show()

def main():
    # 配置参数
    root_dir = r"D:\资料\project\AI_Project\PixelSprite_8Direction\PaperAssets\Octopath\Game\Character\Resource"
    output_dir = r"D:\资料\project\AI_Project\PixelSprite_8Direction\TrainningData\Octopath"
    
    # 裁剪区域的四个顶点坐标，按方向分组
    vertex_params = {
        "down": [208, 64, 240, 64, 208, 112, 240, 112],
        "left": [208, 128, 240, 128, 208, 174, 240, 176],
    }
    
    # 为每个方向提取精灵图像
    for position_flag, params in vertex_params.items():
        print(f"\n提取 {position_flag} 方向的精灵图像...")
        extract_sprites(root_dir, output_dir, params, position_flag, dry_run=False)
    
    # 可选：可视化特定方向的裁剪效果（用于调试）
    # visualize_crop("./path_to_sample.png", vertex_params["down"])

if __name__ == "__main__":
    main()