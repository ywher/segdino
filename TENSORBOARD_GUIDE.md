# TensorBoard 使用指南

## 训练时自动记录的指标

### 训练指标 (Train/)

- `Train/Loss_Step`: 每个batch的训练损失
- `Train/Dice_Step`: 每个batch的Dice系数
- `Train/IoU_Step`: 每个batch的IoU值
- `Train/Loss_Epoch`: 每个epoch的平均训练损失
- `Train/Dice_Epoch`: 每个epoch的平均Dice系数
- `Train/IoU_Epoch`: 每个epoch的平均IoU值
- `Train/Learning_Rate`: 学习率变化

### 验证指标 (Val/)

- `Val/Loss`: 每个epoch的验证损失
- `Val/Dice`: 每个epoch的验证Dice系数
- `Val/IoU`: 每个epoch的验证IoU值

### 可视化图像

- `Train/Input_Images`: 训练输入图像
- `Train/Predicted_Masks`: 训练预测结果
- `Train/Ground_Truth_Masks`: 训练真实标签
- `Val/Input_Images`: 验证输入图像
- `Val/Predicted_Masks`: 验证预测结果
- `Val/Ground_Truth_Masks`: 验证真实标签

### 超参数和最终结果 (Summary/)

- `Summary/Best_Val_Dice`: 最佳验证Dice系数
- `Summary/Best_Val_IoU`: 最佳验证IoU值
- 超参数对比分析

## 如何查看TensorBoard

1. 训练开始后，在终端运行：

```bash
tensorboard --logdir ./runs/segdino_*/tensorboard --port 6006
```

2. 打开浏览器访问：

```
http://localhost:6006
```

3. 可以查看的标签页：
   - **SCALARS**: 查看损失和精度曲线
   - **IMAGES**: 查看训练和验证的图像对比
   - **HPARAMS**: 查看不同超参数的对比结果

## 常用功能

### 平滑曲线

在SCALARS页面，可以调整Smoothing滑块来平滑显示曲线，便于观察趋势。

### 对比不同实验

如果有多个实验运行，TensorBoard会自动在同一图表中显示不同实验的结果，便于对比。

### 下载数据

可以点击左下角的下载按钮，将数据导出为CSV格式进行进一步分析。

## 小贴士

1. 训练过程中可以实时查看TensorBoard，无需等待训练结束
2. 可以通过调整时间范围来查看特定时期的训练情况
3. 图像标签页可以查看模型预测效果的改善过程
4. 建议定期保存TensorBoard的截图作为实验记录
