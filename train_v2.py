import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ------- 反归一化版：支持 (1,H,W) / (3,H,W) -------
def tensor_to_rgb(img_t: torch.Tensor, mean=None, std=None) -> np.ndarray:
    img = img_t.detach().cpu().float()
    img = img.clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)          # (C,H,W)
    img = np.transpose(img, (1, 2, 0))                    # (H,W,C) RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)            # OpenCV 用 BGR
    return img


def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    """
    将 logits 或 概率 或 0/1 mask 转 0-255 的单通道图 (H,W) uint8
    输入 shape: (1,H,W) 或 (H,W)
    """
    m = mask_t.detach().cpu().float()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected mask tensor shape: {m.shape}")

    # 如果值范围明显不是0/1，当作logits走sigmoid
    if m.max() > 1.0 or m.min() < 0.0:
        m = torch.sigmoid(m)

    m_bin = (m > thr).float()
    m_img = (m_bin * 255.0).round().clamp(0, 255).byte().numpy()  # (H,W)
    return m_img

def save_train_visuals(epoch, inputs, logits, targets, out_dir, max_save=8, thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
    b = min(inputs.size(0), max_save)
    for i in range(b):
        # 原图（BGR uint8）
        img_bgr = tensor_to_rgb(inputs[i])
        # 预测 & GT（灰度 0-255，阈值化后二值）
        pred_gray = mask_to_gray(logits[i], thr)
        gt_gray   = mask_to_gray(targets[i], thr)

        base = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}")
        cv2.imwrite(base + "_img.png",  img_bgr)
        cv2.imwrite(base + "_pred.png", pred_gray)
        cv2.imwrite(base + "_gt.png",   gt_gray)

@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, thr=0.5, fname_prefix="val"):
    os.makedirs(out_dir, exist_ok=True)
    # 原图（BGR uint8）
    img_bgr = tensor_to_rgb(inputs)
    # 预测 & GT（灰度 0-255，阈值化后二值）
    pred_gray = mask_to_gray(logits, thr)
    gt_gray   = mask_to_gray(targets, thr)

    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png",  img_bgr)
    cv2.imwrite(base + "_pred.png", pred_gray)
    cv2.imwrite(base + "_gt.png",   gt_gray)


def iou_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    """
    pred_logits: (B,1,H,W) 未经Sigmoid
    target:      (B,1,H,W) 0/1 或 概率
    return:      (B,) 每样本 IoU
    """
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    iou = (inter + eps) / union
    return iou.view(-1)

def dice_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / union
    return dice  # 逐样本 Dice，[B,1] -> [B,1] 或 [B]

# ====== 训练/验证函数 ======
def train_one_epoch(model, train_loader, optimizer, device, num_classes=1, dice_thr=0.5, vis_dir=None, epoch=0):
    model.train()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []

    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    first_batch_logged = False

    pbar = tqdm(train_loader, desc=f"[Train e{epoch}]")
    for step, (inputs, targets, _) in enumerate(pbar):
        inputs  = inputs.to(device)   # (B,3,H,W)
        targets = targets.to(device)  # (B,1,H,W) float
        optimizer.zero_grad()

        logits = model(inputs)        # (B,1,H,W) for binary
        if num_classes == 1:
            loss = criterion(logits, targets)
        else:
            loss = criterion(logits, targets.squeeze(1).long())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算 batch Dice
        with torch.no_grad():
            dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            iou  = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            dice_scores.append(dice)
            iou_scores.append(iou)

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}", iou=f"{iou:.4f}")

        # 只在第一个 batch 可视化若干样本
        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals(epoch, inputs, logits, targets, out_dir=vis_dir, max_save=8, thr=dice_thr)
            first_batch_logged = True

    avg_loss = total_loss / max(1, len(train_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou  = float(np.mean(iou_scores))  if len(iou_scores)  > 0 else 0.0
    print(f"[Train Epoch {epoch}] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}")
    return avg_loss, avg_dice  # 保持原样，便于与你的主循环兼容

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=1, dice_thr=0.5, vis_dir=None):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    iou_scores  = [] 

    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    idx_global = 0
    pbar = tqdm(val_loader, desc="[Eval]")
    for (inputs, targets, _) in pbar:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        if num_classes == 1:
            loss = criterion(logits, targets)
            iou  = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        else:
            loss = criterion(logits, targets.squeeze(1).long())

        total_loss += loss.item()
        dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        iou  = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        dice_scores.append(dice)
        iou_scores.append(iou)

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}", iou=f"{iou:.4f}")
        # pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

        # 保存所有样本可视化
        if vis_dir is not None:
            os.makedirs(vis_dir, exist_ok=True)
            B = inputs.size(0)
            for b in range(B):
                save_eval_visuals(
                    idx_global,
                    inputs[b],
                    logits[b],
                    targets[b],
                    out_dir=vis_dir,
                    thr=dice_thr,
                    fname_prefix="val"
                )
                idx_global += 1

    avg_loss = total_loss / max(1, len(val_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou  = float(np.mean(iou_scores))  if len(iou_scores)  > 0 else 0.0
    print(f"[Eval] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}")
    return avg_loss, avg_dice

# ====== 训练主循环 ======
# save_root = "/data02/users/ysc/segdino/runs/segdinov3_busi"
save_root = "/data02/users/ysc/segdino/runs/segdinov3_cvc"
os.makedirs(save_root, exist_ok=True)
train_vis_dir = os.path.join(save_root, "train_vis")
val_vis_dir   = os.path.join(save_root, "val_vis")
ckpt_dir      = os.path.join(save_root, "ckpts")
os.makedirs(train_vis_dir, exist_ok=True)
os.makedirs(val_vis_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
from tqdm import tqdm
from model import DinoV3SegModel, DinoLastUNetDecoder
import torch
REPO_DIR = "/data02/users/ysc/segdino/dinov3"
CKPT = "/data02/users/ysc/segdino/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
backbone = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=CKPT)

model = DinoV3SegModel(
    backbone=backbone,
    num_classes=1,         
    in_ch=1,               
    dec_chs=(256,192,128,96),
    use_intermediate_layers=True,   
    last_layer_idx=-1              
).cuda()
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 50
lr = 1e-4
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

from glob import glob
from sklearn.model_selection import train_test_split
from dataset import Dataset
data_dir = '/data02/users/ysc/segdino/seg_data'
c_dataset = "cvc"
img_ext = '.png'
# mask_ext = '_mask.png'
mask_ext = '.png'
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
    # transforms.Normalize(),
])

val_transform = Compose([
    Resize(input_h,input_w),
    # transforms.Normalize(),
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


best_val_dice = -1.0
for epoch in range(1, epochs + 1):
    train_loss, train_dice = train_one_epoch(
        model, train_loader, optimizer, device,
        num_classes=num_classes, dice_thr=0.5,
        vis_dir=train_vis_dir, epoch=epoch
    )

    val_loss, val_dice = evaluate(
        model, val_loader, device,
        num_classes=num_classes, dice_thr=0.5,
        vis_dir=val_vis_dir
    )

    # 保存最新和最优权重
    latest_path = os.path.join(ckpt_dir, "latest.pth")
    torch.save(
        {"epoch": epoch, "state_dict": model.state_dict(),
         "optimizer": optimizer.state_dict()},
        latest_path
    )
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        best_path = os.path.join(ckpt_dir, f"best_ep{epoch:03d}_dice{val_dice:.4f}.pth")
        torch.save(model.state_dict(), best_path)
        print(f"[Save] New best ckpt: {best_path}")
