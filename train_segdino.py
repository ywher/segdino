# train_segdinov3.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dpt import DPT
from dataset import FolderDataset, ResizeAndNormalize

class FocalLoss(nn.Module):
    """Focal Loss implementation for binary segmentation"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss implementation for binary segmentation"""
    def __init__(self, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (probs * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        # Dice loss is 1 - dice coefficient
        dice_loss = 1 - dice_coeff
        
        return dice_loss

class CombinedLoss(nn.Module):
    """Combined loss function with BCE, Focal, and Dice losses"""
    def __init__(self, bce_weight=1.0, focal_weight=1.0, dice_weight=1.0, 
                 focal_alpha=1.0, focal_gamma=2.0, dice_smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
    
    def forward(self, inputs, targets):
        losses = {}
        total_loss = 0
        
        if self.bce_weight > 0:
            bce = self.bce_loss(inputs, targets)
            losses['bce'] = bce
            total_loss += self.bce_weight * bce
        
        if self.focal_weight > 0:
            focal = self.focal_loss(inputs, targets)
            losses['focal'] = focal
            total_loss += self.focal_weight * focal
        
        if self.dice_weight > 0:
            dice = self.dice_loss(inputs, targets)
            losses['dice'] = dice
            total_loss += self.dice_weight * dice
        
        losses['total'] = total_loss
        # 添加一个属性来存储详细损失，但返回总损失用于反向传播
        self.last_losses = losses
        return total_loss

def count_parameters(model):
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }

def print_model_parameters(model, logger=None):
    """打印模型参数统计信息"""
    param_stats = count_parameters(model)
    
    def format_number(num):
        """格式化数字，添加千位分隔符"""
        return f"{num:,}"
    
    info_lines = [
        "=" * 60,
        "Model Parameter Statistics:",
        f"  Total parameters:     {format_number(param_stats['total'])}",
        f"  Trainable parameters: {format_number(param_stats['trainable'])}",
        f"  Frozen parameters:    {format_number(param_stats['frozen'])}",
        f"  Trainable ratio:      {param_stats['trainable']/param_stats['total']*100:.2f}%",
        "=" * 60
    ]
    
    for line in info_lines:
        if logger:
            logger.info(line)
        else:
            print(line)
    
    return param_stats

def tensor_to_rgb(img_t: torch.Tensor, mean=None, std=None) -> np.ndarray:
    img = img_t.detach().cpu().float()
    img = img.clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    m = mask_t.detach().cpu().float()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected mask tensor shape: {m.shape}")
    if m.max() > 1.0 or m.min() < 0.0:
        m = torch.sigmoid(m)
    m_bin = (m > thr).float()
    m_img = (m_bin * 255.0).round().clamp(0, 255).byte().numpy()
    return m_img

def save_train_visuals(epoch, inputs, logits, targets, out_dir, max_save=8, thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
    b = min(inputs.size(0), max_save)
    for i in range(b):
        img_bgr = tensor_to_rgb(inputs[i])
        pred_gray = mask_to_gray(logits[i], thr)
        gt_gray   = mask_to_gray(targets[i], thr)
        base = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}")
        cv2.imwrite(base + "_img.png",  img_bgr)
        cv2.imwrite(base + "_pred.png", pred_gray)
        cv2.imwrite(base + "_gt.png",   gt_gray)

def log_images_to_tensorboard(writer, inputs, logits, targets, epoch, tag, max_images=4, thr=0.5):
    """Log images to tensorboard"""
    import torchvision.utils as vutils
    
    # 限制图像数量
    batch_size = min(inputs.size(0), max_images)
    
    # 准备输入图像 (归一化到0-1)
    input_imgs = inputs[:batch_size].detach().cpu()
    if input_imgs.shape[1] == 1:  # 灰度图像
        input_imgs = input_imgs.repeat(1, 3, 1, 1)  # 转换为3通道以便可视化
    
    # 准备预测结果
    pred_probs = torch.sigmoid(logits[:batch_size]).detach().cpu()
    pred_masks = (pred_probs > thr).float()
    if pred_masks.shape[1] == 1:
        pred_masks = pred_masks.repeat(1, 3, 1, 1)
    
    # 准备真实标签
    gt_masks = targets[:batch_size].detach().cpu()
    if gt_masks.shape[1] == 1:
        gt_masks = gt_masks.repeat(1, 3, 1, 1)
    
    # 创建图像网格
    img_grid = vutils.make_grid(input_imgs, nrow=batch_size, normalize=True, padding=2)
    pred_grid = vutils.make_grid(pred_masks, nrow=batch_size, normalize=False, padding=2)
    gt_grid = vutils.make_grid(gt_masks, nrow=batch_size, normalize=False, padding=2)
    
    # 记录到tensorboard
    writer.add_image(f'{tag}/Input_Images', img_grid, epoch)
    writer.add_image(f'{tag}/Predicted_Masks', pred_grid, epoch)
    writer.add_image(f'{tag}/Ground_Truth_Masks', gt_grid, epoch)

@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, thr=0.5, fname_prefix="val"):
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = tensor_to_rgb(inputs)
    pred_gray = mask_to_gray(logits, thr)
    gt_gray   = mask_to_gray(targets, thr)
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png",  img_bgr)
    cv2.imwrite(base + "_pred.png", pred_gray)
    cv2.imwrite(base + "_gt.png",   gt_gray)

def iou_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
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
    return dice

def train_one_epoch(model, train_loader, optimizer, device, num_classes=1, dice_thr=0.5, vis_dir=None, epoch=0, writer=None, criterion=None):
    model.train()
    total_loss = 0.0
    bce_loss_sum = 0.0
    focal_loss_sum = 0.0
    dice_loss_sum = 0.0
    dice_scores = []
    iou_scores = []
    
    # Use default BCE loss if no criterion provided
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        use_combined_loss = False
    else:
        use_combined_loss = isinstance(criterion, CombinedLoss)
    
    first_batch_logged = False
    
    # 配置进度条显示ETA
    pbar = tqdm(train_loader, 
                desc=f"[Train e{epoch}]",
                unit="batch",
                leave=True,
                ncols=140,  # 增加宽度以显示更多信息
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for step, (inputs, targets, _) in enumerate(pbar):
        inputs  = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        
        if num_classes == 1:
            if use_combined_loss:
                loss = criterion(logits, targets)
                # Get detailed losses from the last_losses attribute
                losses = criterion.last_losses
                
                # Record individual losses
                if 'bce' in losses:
                    bce_loss_sum += losses['bce'].item()
                if 'focal' in losses:
                    focal_loss_sum += losses['focal'].item()
                if 'dice' in losses:
                    dice_loss_sum += losses['dice'].item()
            else:
                loss = criterion(logits, targets)
        else:
            loss = criterion(logits, targets.squeeze(1).long())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        with torch.no_grad():
            dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            iou  = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            dice_scores.append(dice)
            iou_scores.append(iou)
        
        # Update progress bar with loss information
        postfix_dict = {
            'loss': f"{loss.item():.4f}",
            'dice': f"{dice:.4f}",
            'iou': f"{iou:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        }
        
        if use_combined_loss and step > 0:
            avg_bce = bce_loss_sum / (step + 1) if bce_loss_sum > 0 else 0
            avg_focal = focal_loss_sum / (step + 1) if focal_loss_sum > 0 else 0
            avg_dice_loss = dice_loss_sum / (step + 1) if dice_loss_sum > 0 else 0
            postfix_dict.update({
                'bce': f"{avg_bce:.3f}",
                'focal': f"{avg_focal:.3f}",
                'dloss': f"{avg_dice_loss:.3f}"
            })
        
        pbar.set_postfix(postfix_dict)
        
        # 记录每个batch的指标到TensorBoard
        if writer is not None:
            global_step = (epoch - 1) * len(train_loader) + step
            writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            writer.add_scalar('Train/Dice_Step', dice, global_step)
            writer.add_scalar('Train/IoU_Step', iou, global_step)
            
            # Record individual loss components if using combined loss
            if use_combined_loss and hasattr(criterion, 'last_losses'):
                losses = criterion.last_losses
                if 'bce' in losses:
                    writer.add_scalar('Train/BCE_Loss_Step', losses['bce'].item(), global_step)
                if 'focal' in losses:
                    writer.add_scalar('Train/Focal_Loss_Step', losses['focal'].item(), global_step)
                if 'dice' in losses:
                    writer.add_scalar('Train/Dice_Loss_Step', losses['dice'].item(), global_step)
        
        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals(epoch, inputs, logits, targets, out_dir=vis_dir, max_save=8, thr=dice_thr)
            # 记录训练图像到TensorBoard
            if writer is not None:
                log_images_to_tensorboard(writer, inputs, logits, targets, epoch, 'Train', max_images=4, thr=dice_thr)
            first_batch_logged = True
    
    avg_loss = total_loss / max(1, len(train_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou  = float(np.mean(iou_scores))  if len(iou_scores)  > 0 else 0.0
    
    # Calculate average individual losses
    avg_bce_loss = bce_loss_sum / max(1, len(train_loader)) if bce_loss_sum > 0 else 0.0
    avg_focal_loss = focal_loss_sum / max(1, len(train_loader)) if focal_loss_sum > 0 else 0.0
    avg_dice_loss = dice_loss_sum / max(1, len(train_loader)) if dice_loss_sum > 0 else 0.0
    
    # 记录epoch平均指标到TensorBoard
    if writer is not None:
        writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch)
        writer.add_scalar('Train/Dice_Epoch', avg_dice, epoch)
        writer.add_scalar('Train/IoU_Epoch', avg_iou, epoch)
        
        if use_combined_loss:
            if avg_bce_loss > 0:
                writer.add_scalar('Train/BCE_Loss_Epoch', avg_bce_loss, epoch)
            if avg_focal_loss > 0:
                writer.add_scalar('Train/Focal_Loss_Epoch', avg_focal_loss, epoch)
            if avg_dice_loss > 0:
                writer.add_scalar('Train/Dice_Loss_Epoch', avg_dice_loss, epoch)
    
    # Enhanced logging with individual losses
    if use_combined_loss:
        loss_components = []
        if avg_bce_loss > 0:
            loss_components.append(f"bce={avg_bce_loss:.4f}")
        if avg_focal_loss > 0:
            loss_components.append(f"focal={avg_focal_loss:.4f}")
        if avg_dice_loss > 0:
            loss_components.append(f"dice_loss={avg_dice_loss:.4f}")
        
        loss_info = f"total={avg_loss:.4f} ({', '.join(loss_components)})" if loss_components else f"total={avg_loss:.4f}"
        logging.info(f"[Train Epoch {epoch}] loss={loss_info}  dice={avg_dice:.4f}  iou={avg_iou:.4f}")
    else:
        logging.info(f"[Train Epoch {epoch}] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}")
    
    return avg_loss, avg_dice

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=1, dice_thr=0.5, vis_dir=None, epoch=0, writer=None, criterion=None):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    iou_scores  = []
    
    # Use default BCE loss if no criterion provided
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        use_combined_loss = False
    else:
        use_combined_loss = isinstance(criterion, CombinedLoss)
    
    idx_global = 0
    first_batch_logged = False
    
    # 配置验证进度条显示ETA
    pbar = tqdm(val_loader, 
                desc="[Eval]",
                unit="batch",
                leave=True,
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for (inputs, targets, _) in pbar:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        
        if num_classes == 1:
            if use_combined_loss:
                loss = criterion(logits, targets)
            else:
                loss = criterion(logits, targets)
        else:
            loss = criterion(logits, targets.squeeze(1).long())
            
        total_loss += loss.item()
        dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        iou  = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        dice_scores.append(dice)
        iou_scores.append(iou)
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{dice:.4f}",
            'iou': f"{iou:.4f}"
        })
        
        # 记录第一个batch的验证图像到TensorBoard
        if (not first_batch_logged) and writer is not None:
            log_images_to_tensorboard(writer, inputs, logits, targets, epoch, 'Val', max_images=4, thr=dice_thr)
            first_batch_logged = True
            
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
    
    # 记录验证指标到TensorBoard
    if writer is not None:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/Dice', avg_dice, epoch)
        writer.add_scalar('Val/IoU', avg_iou, epoch)
    
    logging.info(f"[Eval] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}")
    return avg_loss, avg_dice, avg_iou

def main():
    import argparse
    import random
    from glob import glob
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="tn3k")
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--in_ch", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=1)
    # model args
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_size", type=str, default="b", choices=["b", "s"],
                        help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--dino_ckpt", type=str, required=True,help="Path to the pretrained DINO checkpoint (.pth). "
                         "Use ViT-B/16 checkpoint for --dino_size b, "
                         "or ViT-S/16 checkpoint for --dino_size s.")
    # training args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    # loss args
    parser.add_argument("--bce_weight", type=float, default=1.0,
                        help="Weight for BCE loss component")
    parser.add_argument("--focal_weight", type=float, default=0.0,
                        help="Weight for Focal loss component (0 to disable)")
    parser.add_argument("--dice_weight", type=float, default=0.0,
                        help="Weight for Dice loss component (0 to disable)")
    parser.add_argument("--focal_alpha", type=float, default=1.0,
                        help="Alpha parameter for Focal loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for Focal loss")
    parser.add_argument("--dice_smooth", type=float, default=1.0,
                        help="Smoothing parameter for Dice loss")
    # other args 
    parser.add_argument("--last_layer_idx", type=int, default=-1)
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--freeze_backbone", action="store_true", 
                        help="Freeze the DINO backbone during training")
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.freeze_backbone:
        save_root = f"./runs/segdino_{args.dino_size}_{args.input_h}_{args.dataset}_freeze"
    else:
        save_root = f"./runs/segdino_{args.dino_size}_{args.input_h}_{args.dataset}"
    os.makedirs(save_root, exist_ok=True)
    
    # 设置logging配置
    log_file = os.path.join(save_root, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Training started. Logs will be saved to {log_file}")
    logger.info(f"Arguments: {vars(args)}")
    
    train_vis_dir = os.path.join(save_root, "train_vis")
    val_vis_dir   = os.path.join(save_root, "val_vis")
    ckpt_dir      = os.path.join(save_root, "ckpts")
    tensorboard_dir = os.path.join(save_root, "tensorboard")
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(val_vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 清理旧的best模型文件（可选）
    import glob
    old_best_files = glob.glob(os.path.join(ckpt_dir, "best_ep*.pth"))
    if old_best_files:
        logging.info(f"Found {len(old_best_files)} old best model files, cleaning up...")
        for old_file in old_best_files:
            try:
                os.remove(old_file)
                logging.info(f"[Cleanup] Removed old best model: {os.path.basename(old_file)}")
            except Exception as e:
                logging.warning(f"[Warning] Failed to remove {old_file}: {e}")
    
    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    logger.info("Run 'tensorboard --logdir {} --port 6006' to view in browser".format(tensorboard_dir))

    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)
    
    logging.info(f"Loaded DINO backbone: {args.dino_size} from {args.dino_ckpt}")

    
    model = DPT(nclass=1, backbone=backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logging.info(f"Model loaded on device: {device}")
    
    # 可选择冻结backbone
    if args.freeze_backbone:
        model.lock_backbone()
        logging.info("DINO backbone has been frozen")
    else:
        logging.info("DINO backbone will be fine-tuned")
    
    # 统计和输出模型参数信息
    param_stats = print_model_parameters(model, logging)
    
    # Create loss function based on parameters
    if args.focal_weight > 0 or args.dice_weight > 0:
        criterion = CombinedLoss(
            bce_weight=args.bce_weight,
            focal_weight=args.focal_weight,
            dice_weight=args.dice_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            dice_smooth=args.dice_smooth
        )
        loss_info = []
        if args.bce_weight > 0:
            loss_info.append(f"BCE(w={args.bce_weight})")
        if args.focal_weight > 0:
            loss_info.append(f"Focal(w={args.focal_weight}, α={args.focal_alpha}, γ={args.focal_gamma})")
        if args.dice_weight > 0:
            loss_info.append(f"Dice(w={args.dice_weight}, smooth={args.dice_smooth})")
        logging.info(f"Using Combined Loss: {' + '.join(loss_info)}")
    else:
        criterion = nn.BCEWithLogitsLoss() if args.num_classes == 1 else nn.CrossEntropyLoss()
        logging.info("Using default BCE loss")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    
    root = os.path.join(args.data_dir, args.dataset)

    train_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    val_transform   = ResizeAndNormalize(size=(args.input_h, args.input_w))

    train_dataset = FolderDataset(
        root=root,
        split="train",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=train_transform,
    )
    val_dataset = FolderDataset(
        root=root,
        split="test",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=val_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    logging.info(f"Dataset loaded: {args.dataset}")
    logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logging.info(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")

    best_val_dice = -1.0
    best_val_dice_epoch = -1
    best_val_iou  = -1.0
    best_val_iou_epoch  = -1
    best_model_path = None  # 记录当前最佳模型路径，用于删除旧模型

    # 记录训练开始时间，用于计算总训练时间的ETA
    import time
    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, device,
            num_classes=args.num_classes, dice_thr=0.5,
            vis_dir=train_vis_dir, epoch=epoch, writer=writer,
            criterion=criterion
        )
        val_loss, val_dice, val_iou = evaluate(
            model, val_loader, device,
            num_classes=args.num_classes, dice_thr=0.5,
            vis_dir=val_vis_dir, epoch=epoch, writer=writer,
            criterion=criterion
        )
        
        # 记录学习率到TensorBoard
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        latest_path = os.path.join(ckpt_dir, "latest.pth")
        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()},
            latest_path
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_dice_epoch = epoch
            
            # 删除之前的最佳模型文件
            if best_model_path is not None and os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                    logging.info(f"[Delete] Removed previous best model: {os.path.basename(best_model_path)}")
                except Exception as e:
                    logging.warning(f"[Warning] Failed to remove previous best model: {e}")
            
            # 保存新的最佳模型
            best_path = os.path.join(ckpt_dir, f"best_ep{epoch:03d}_dice{val_dice:.4f}_{val_iou:.4f}.pth")
            torch.save(model.state_dict(), best_path)
            best_model_path = best_path  # 更新最佳模型路径
            logging.info(f"[Save] New best ckpt: {best_path}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_iou_epoch = epoch
        
        # 计算并显示训练进度和ETA
        current_time = time.time()
        elapsed_time = current_time - training_start_time
        avg_time_per_epoch = elapsed_time / epoch
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_time_per_epoch * remaining_epochs
        
        # 格式化时间显示
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds//60:.0f}m{seconds%60:.0f}s"
            else:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{hours:.0f}h{minutes:.0f}m"
        
        progress_info = (
            f"Epoch {epoch}/{args.epochs} | "
            f"Progress: {epoch/args.epochs*100:.1f}% | "
            f"Elapsed: {format_time(elapsed_time)} | "
            f"ETA: {format_time(eta_seconds)} | "
            f"Best Dice: {best_val_dice:.4f} | "
            f"Current Dice: {val_dice:.4f}"
        )
        
        logging.info(f"[Training Progress] {progress_info}")
            
    # 保存最终模型
    final_path = os.path.join(ckpt_dir, f"final_ep{args.epochs:03d}_dice{val_dice:.4f}_{val_iou:.4f}.pth")
    torch.save(model.state_dict(), final_path)
    logging.info(f"[Save] Final model ckpt: {final_path}")
    
    # 计算总训练时间
    total_training_time = time.time() - training_start_time
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m{seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h{minutes:.0f}m"
    
    logging.info(f"[Training Completed] Total training time: {format_time(total_training_time)}")
    logging.info('\n')

    # 记录最佳指标到TensorBoard
    writer.add_scalar('Summary/Best_Val_Dice', best_val_dice, best_val_dice_epoch)
    writer.add_scalar('Summary/Best_Val_IoU', best_val_iou, best_val_iou_epoch)
    
    # 记录超参数和最终结果
    writer.add_hparams(
        {
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'weight_decay': args.weight_decay,
            'dino_size': args.dino_size,
            'input_size': args.input_h,
        },
        {
            'best_val_dice': best_val_dice,
            'best_val_iou': best_val_iou,
            'best_dice_epoch': best_val_dice_epoch,
            'best_iou_epoch': best_val_iou_epoch,
        }
    )
    
    writer.close()

    logging.info("=" * 60)
    logging.info(f"[Summary] Best Val Dice = {best_val_dice:.4f} @ epoch {best_val_dice_epoch}")
    logging.info(f"[Summary] Best Val IoU  = {best_val_iou:.4f}  @ epoch {best_val_iou_epoch}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
