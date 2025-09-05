#!/usr/bin/env python3
"""
语义标签格式转换工具
将语义标签转换为调色板模式的PNG图像

调色板模式说明:
- 输出为P模式(调色板模式)的PNG图像
- 每个像素值直接对应类别ID
- 通过调色板定义每个ID的显示颜色
- 类别0: RGB(0,0,0) 黑色
- 类别1: RGB(255,255,255) 白色

使用方法:
python convert_label.py --input_path /path/to/labels --output_path /path/to/output
"""

import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob

def create_color_palette():
    """创建调色板，返回256*3的颜色列表"""
    # 创建256种颜色的调色板
    palette = []
    
    # 定义类别颜色
    class_colors = {
        0: [0, 0, 0],        # 背景: 黑色
        1: [255, 255, 255]   # 前景: 白色
    }
    
    # 为每个类别ID设置颜色
    for i in range(256):
        if i in class_colors:
            palette.extend(class_colors[i])
        else:
            # 未定义的类别使用随机颜色或默认颜色
            palette.extend([i, i, i])  # 灰度色
    
    return palette

def convert_label_to_palette(label_path, palette):
    """
    将单个标签图像转换为调色板模式的PNG图像
    
    Args:
        label_path: 输入标签图像路径
        palette: 调色板颜色列表
    
    Returns:
        PIL.Image: 转换后的调色板模式图像
    """
    # 读取标签图像
    label_img = Image.open(label_path)
    
    # 转换为灰度图(L模式)，确保像素值就是类别ID
    if label_img.mode != 'L':
        label_img = label_img.convert('L')
    
    # 转换为调色板模式(P模式)
    palette_img = label_img.convert('P')
    
    # 设置调色板
    palette_img.putpalette(palette)
    
    return palette_img

def process_labels(input_path, output_path, palette):
    """
    批量处理标签图像
    
    Args:
        input_path: 输入标签文件夹路径
        output_path: 输出文件夹路径
        palette: 调色板颜色列表
    """
    # 支持的图像格式
    supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    
    # 获取所有标签文件
    label_files = []
    for fmt in supported_formats:
        label_files.extend(glob.glob(os.path.join(input_path, fmt)))
        label_files.extend(glob.glob(os.path.join(input_path, fmt.upper())))
    
    if not label_files:
        print(f"在路径 {input_path} 中未找到支持的图像文件")
        return
    
    print(f"找到 {len(label_files)} 个标签文件")
    print(f"将转换为调色板模式PNG图像:")
    print(f"  类别0 -> RGB(0,0,0) 黑色")
    print(f"  类别1 -> RGB(255,255,255) 白色")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 处理每个标签文件
    success_count = 0
    for label_file in tqdm(label_files, desc="转换标签"):
        try:
            # 转换标签到调色板模式
            palette_img = convert_label_to_palette(label_file, palette)
            
            # 生成输出文件名
            base_name = os.path.basename(label_file)
            name, ext = os.path.splitext(base_name)
            output_file = os.path.join(output_path, f"{name}.png")
            
            # 保存调色板模式的PNG图像
            palette_img.save(output_file)
            success_count += 1
            
        except Exception as e:
            print(f"处理文件 {label_file} 时出错: {e}")
    
    print(f"转换完成! 成功处理 {success_count}/{len(label_files)} 个文件")
    print(f"输出保存在: {output_path}")
    print(f"输出格式: 调色板模式PNG (每个像素值即为类别ID)")

def main():
    parser = argparse.ArgumentParser(description="语义标签格式转换工具")
    parser.add_argument("--input_path", type=str, required=True,
                        help="输入标签文件夹路径")
    parser.add_argument("--output_path", type=str, default=None,
                        help="输出文件夹路径(可选)")
    parser.add_argument("--suffix", type=str, default="_palette",
                        help="当未指定输出路径时，输出文件夹的后缀名")
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input_path):
        print(f"错误: 输入路径 {args.input_path} 不存在")
        return
    
    if not os.path.isdir(args.input_path):
        print(f"错误: 输入路径 {args.input_path} 不是文件夹")
        return
    
    # 确定输出路径
    if args.output_path is None:
        # 在输入文件夹同一层创建同名加后缀的文件夹
        parent_dir = os.path.dirname(args.input_path.rstrip('/'))
        folder_name = os.path.basename(args.input_path.rstrip('/'))
        output_path = os.path.join(parent_dir, folder_name + args.suffix)
    else:
        output_path = args.output_path
    
    print(f"输入路径: {args.input_path}")
    print(f"输出路径: {output_path}")
    
    # 创建调色板
    palette = create_color_palette()
    
    # 处理标签
    process_labels(args.input_path, output_path, palette)

if __name__ == "__main__":
    main()