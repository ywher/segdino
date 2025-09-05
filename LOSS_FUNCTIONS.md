# 损失函数使用说明

## 新增的损失函数

### 1. Focal Loss
- **用途**: 解决类别不平衡问题，关注难分类样本
- **参数**:
  - `alpha`: 类别权重 (默认1.0)
  - `gamma`: 聚焦参数 (默认2.0)
- **特点**: 减少简单样本的损失贡献，增加难样本的权重

### 2. Dice Loss  
- **用途**: 优化重叠度，适合分割任务
- **参数**:
  - `smooth`: 平滑参数 (默认1.0)
- **特点**: 直接优化Dice系数，有助于提高分割精度

### 3. Combined Loss
- **用途**: 结合多种损失函数的优势
- **组成**: BCE Loss + Focal Loss + Dice Loss
- **权重参数**:
  - `--bce_weight`: BCE损失权重 (默认1.0)
  - `--focal_weight`: Focal损失权重 (默认1.0) 
  - `--dice_weight`: Dice损失权重 (默认1.0)
  - `--focal_alpha`: Focal Loss的alpha参数 (默认1.0)
  - `--focal_gamma`: Focal Loss的gamma参数 (默认2.0)
  - `--dice_smooth`: Dice Loss的smooth参数 (默认1.0)

## 使用方法

### 基础BCE损失 (默认)
```bash
python train_segdino.py --dino_ckpt path/to/ckpt.pth
```

### 使用组合损失
```bash
python train_segdino.py \
    --dino_ckpt path/to/ckpt.pth \
    --bce_weight 1.0 \
    --focal_weight 1.0 \
    --dice_weight 1.0
```

### 自定义权重示例
```bash
# 强调Focal Loss，减少BCE权重
python train_segdino.py \
    --dino_ckpt path/to/ckpt.pth \
    --bce_weight 0.5 \
    --focal_weight 2.0 \
    --dice_weight 1.0 \
    --focal_alpha 0.25 \
    --focal_gamma 2.0

# 只使用Dice Loss
python train_segdino.py \
    --dino_ckpt path/to/ckpt.pth \
    --bce_weight 0.0 \
    --focal_weight 0.0 \
    --dice_weight 1.0

# 平衡的组合损失
python train_segdino.py \
    --dino_ckpt path/to/ckpt.pth \
    --bce_weight 0.4 \
    --focal_weight 0.4 \
    --dice_weight 0.2
```

## TensorBoard监控

使用组合损失时，会在TensorBoard中记录：
- `Train/Loss_Step`: 总损失
- `Train/BCE_Loss_Step`: BCE损失分量
- `Train/Focal_Loss_Step`: Focal损失分量  
- `Train/Dice_Loss_Step`: Dice损失分量
- 对应的Epoch级别损失

## 进度条显示

训练时进度条会显示：
- `loss`: 总损失
- `bce`: BCE损失平均值
- `focal`: Focal损失平均值
- `dloss`: Dice损失平均值
- `dice`: Dice系数
- `iou`: IoU分数
- `lr`: 学习率

## 推荐配置

### 类别平衡的数据集
```bash
--bce_weight 1.0 --focal_weight 0.0 --dice_weight 1.0
```

### 类别不平衡的数据集
```bash
--bce_weight 0.5 --focal_weight 2.0 --dice_weight 1.0 --focal_alpha 0.25 --focal_gamma 2.0
```

### 追求高精度分割
```bash
--bce_weight 0.2 --focal_weight 0.3 --dice_weight 0.5
```

## 注意事项

1. **权重调节**: 建议从默认权重开始，根据验证集表现调整
2. **Focal参数**: gamma=2.0适合大多数情况，alpha根据类别比例调整
3. **监控损失**: 通过TensorBoard观察各损失分量的变化趋势
4. **实验对比**: 可以尝试不同的损失组合，找到最适合数据集的配置
