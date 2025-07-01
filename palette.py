from PIL import Image
import numpy as np

def get_palette(img):

    # 打开并加载图片
    img = img.convert('RGBA')  # 确保是RGBA模式
    img_array = np.array(img)
    
    # 获取图片尺寸和通道数
    height, width, _ = img_array.shape[:]
    
    # 创建RGB元组列表
    rgb_set = set()
    
    # 提取非透明像素的RGB值
    alpha = img_array[:, :, 3]
    for y in range(height):
        for x in range(width):
            if alpha[y, x] != 0:  # 只提取不透明的颜色
                rgb = tuple(int(c) for c in img_array[y, x, :3].tolist())
                rgb_set.add(rgb)
    
    return tuple(rgb_set)

def apply_palette_and_binarize(source_img, target_img):
    """
    1. 计算待处理图source_img每个像素的 RGB，与参考图target_img每个元素的 RGB 计算差的平方和，取最小的色盘元素替换
    2. 检测预测图所有像素的 alpha 通道，等于255的设为1，其余设为0
    """

    img_array = np.array(source_img)  # (H, W, 4)
    source_palette = get_palette(source_img)
    target_palette = get_palette(target_img)
    color_mapping = {}
    
    # 遍历待处理色盘的每个颜色
    for color_s in source_palette:
        min_dist = float('inf')  # 初始化最小距离为无穷大
        nearest_color = None     # 初始化最近颜色为None
        
        # 遍历目标色盘的每个颜色，计算距离
        for color_t in target_palette:
            # 计算RGB平方差之和
            dist = (color_s[0] - color_t[0])**2 + \
                   (color_s[1] - color_t[1])**2 + \
                   (color_s[2] - color_t[2])**2
            
            # 更新最小距离和最近颜色
            if dist < min_dist:
                min_dist = dist
                nearest_color = color_t
        
        # 建立映射关系
        color_mapping[color_s] = nearest_color

    # 替换图像中的颜色
    h, w = img_array.shape[:2]
    for y in range(h):
        for x in range(w):
            if img_array[y, x, 3] != 0:  # 只处理不透明像素
                rgb = tuple(int(c) for c in img_array[y, x, :3])
                new_rgb = color_mapping[rgb]
                img_array[y, x, 0] = new_rgb[0]  # 单独设置R
                img_array[y, x, 1] = new_rgb[1]  # 单独设置G
                img_array[y, x, 2] = new_rgb[2]  # 单独设置B
                img_array[y, x, 3] = 0 if img_array[y, x, 3] < 127 else 255  # 透明度设为0或255

    # 创建PIL图像并返回
    return Image.fromarray(img_array, mode='RGBA')