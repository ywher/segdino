import os
import cv2
import numpy as np
import torch
import torch.utils.data
from glob import glob
from sklearn.model_selection import train_test_split
from dataset import Dataset


data_dir = '/data02/users/ysc/segdino/seg_data'
c_dataset = "busi"
img_ext = '.png'
mask_ext = '_mask.png'
num_classes = 1
batch_size = 2
seed = 1029
input_h,input_w = 256, 256
img_ids = sorted(glob(os.path.join(data_dir, c_dataset,'images', '*' + img_ext)))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=seed)

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from albumentations import RandomRotate90, Resize
from albumentations.core.composition import Compose
train_transform = Compose([
    RandomRotate90(),
    # geometric.transforms.Flip(),
    Resize(input_h,input_w),
    transforms.Normalize(),
])

val_transform = Compose([
    Resize(input_h,input_w),
    transforms.Normalize(),
])

train_dataset = Dataset(
    img_ids=train_img_ids,
    img_dir=os.path.join(data_dir, c_dataset, 'images'),
    mask_dir=os.path.join(data_dir, c_dataset, 'masks'),
    img_ext=img_ext,
    mask_ext=mask_ext,
    num_classes=num_classes,
    transform=train_transform)
val_dataset = Dataset(
    img_ids=val_img_ids,
    img_dir=os.path.join(data_dir, c_dataset, 'images'),
    mask_dir=os.path.join(data_dir, c_dataset, 'masks'),
    img_ext=img_ext,
    mask_ext=mask_ext,
    num_classes=num_classes,
    transform=val_transform)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=16,
    drop_last=False)


def dice_binary(pred_logits, target, eps=1e-6, thresh=0.5):
    """
    Binary Dice，pred_logits: [B,1,H,W]；target: [B,1,H,W] (float 0/1 或 [0,1])
    """
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / union
    return dice.mean().item()

from tqdm import tqdm
from model import DinoV3SegModel, DinoLastUNetDecoder
import torch
REPO_DIR = "/data02/users/ysc/segdino/dinov3"
CKPT = "/data02/users/ysc/segdino/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
backbone = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=CKPT)

# model = DinoV3SegModel(
#     backbone=backbone,
#     num_classes=num_classes,         
#     in_ch=1,               
#     dec_chs=(256,192,128,96),
#     use_intermediate_layers=True,   
#     last_layer_idx=-1              # 只取最后一层
# ).cuda()
from unet_model import UNet
model = UNet(n_channels=3, n_classes=num_classes).cuda()
# pbar = tqdm(total=len(train_loader))
# for input, target, _ in train_loader:
#     input = input.cuda()
#     target = target.cuda()
#     output = model(input)
import torch.nn as nn
def train_one_epoch(model, train_loader, optimizer, device, num_classes=1):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    for input, target, _ in train_loader:
        input  = input.to(device)   # (B,3,H,W)
        target = target.to(device) # (B,1,H,W)

        optimizer.zero_grad()
        logits = model(input)      # (B,num_classes,H,W)

        if num_classes == 1:
            loss = criterion(logits, target)  # target: float, same shape
        else:
            loss = criterion(logits, target.squeeze(1).long())  # target: long, (B,H,W)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)