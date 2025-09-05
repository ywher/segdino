# 语义标签格式转换工具使用说明

## 功能描述

将语义标签转换为ID格式的RGB图像，其中：

- 0 (背景) → RGB(0, 0, 0) - 黑色
- 1 (前景) → RGB(255, 255, 255) - 白色

## 使用方法

### 基本用法

```bash
# 指定输入和输出路径
python tools/convert_label.py --input_path /path/to/labels --output_path /path/to/output

# 只指定输入路径，输出路径自动生成
python tools/convert_label.py --input_path /path/to/labels

# 自定义输出文件夹后缀
python tools/convert_label.py --input_path /path/to/labels --suffix "_converted"
```

### 参数说明

- `--input_path`: 输入标签文件夹路径 (必需)
- `--output_path`: 输出文件夹路径 (可选)
- `--suffix`: 当未指定输出路径时的文件夹后缀 (默认: "_rgb")

### 示例

1. **基本转换**

```bash
python tools/convert_label.py --input_path ./segdata/tn3k/train/mask
```

输出将保存在: `./segdata/tn3k/train/mask_rgb/`

2. **指定输出路径**

```bash
python tools/convert_label.py \
    --input_path ./segdata/tn3k/train/mask \
    --output_path ./segdata/tn3k/train/mask_converted
```

3. **自定义后缀**

```bash
python tools/convert_label.py \
    --input_path ./segdata/tn3k/train/mask \
    --suffix "_id_format"
```

输出将保存在: `./segdata/tn3k/train/mask_id_format/`

## 支持的图像格式

- PNG, JPG, JPEG, BMP, TIFF, TIF

## 颜色映射

```
ID 0 (背景): RGB(0, 0, 0)   - 黑色
ID 1 (前景): RGB(255, 255, 255) - 白色
```

## 输出

- 转换后的图像保存为PNG格式
- 保持原始文件名，扩展名统一为.png
- 显示处理进度和成功率
