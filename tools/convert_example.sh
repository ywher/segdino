#!/bin/bash
# 语义标签转换示例脚本

echo "=== 语义标签调色板转换示例 ==="

# 示例1: 转换到指定输出路径
echo "示例1: 转换到指定路径"
python convert_label.py \
    --input_path ../segdata/xh_kidney/test/label \
    --output_path ../segdata/xh_kidney/test/label_palette

# 示例2: 自动生成输出路径(在同级目录)
# echo "示例2: 自动生成输出路径"
# python tools/convert_label.py \
#     --input_path ./segdata/tn3k/test/mask \
#     --suffix _palette

echo "=== 转换完成 ==="
echo ""
echo "输出说明:"
echo "- 调色板模式PNG图像"
echo "- 每个像素值 = 类别ID"
echo "- 类别0 -> 黑色 RGB(0,0,0)"
echo "- 类别1 -> 白色 RGB(255,255,255)"
echo ""
echo "验证方法:"
echo "使用图像查看器打开生成的PNG文件即可看到彩色的语义分割标签"
