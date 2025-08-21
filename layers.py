# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 基础模块 =====
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),
        )
    def forward(self, x): return self.block(x)

class UpNoSkip(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, out_hw):
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return self.conv(x)

class DinoLastUNetDecoder(nn.Module):
    def __init__(self, C_in=384, dec_chs=(256, 192, 128, 96), num_classes=2):
        super().__init__()
        d4, d3, d2, d1 = dec_chs

        self.in_proj = ConvBNReLU(C_in, d4, k=1, s=1, p=0)

        # H/16 -> H/8 -> H/4 -> H/2 -> H
        self.up4 = UpNoSkip(d4, d4)   # 到 H/8
        self.up3 = UpNoSkip(d4, d3)   # 到 H/4
        self.up2 = UpNoSkip(d3, d2)   # 到 H/2
        self.up1 = UpNoSkip(d2, d1)   # 到 H

        self.head = nn.Conv2d(d1, num_classes, kernel_size=1)

    def forward(self, feat_last: torch.Tensor, out_hw):
        H, W = out_hw

        # 逐级目标尺寸
        hw8 = (H // 8,  W // 8)
        hw4 = (H // 4,  W // 4)
        hw2 = (H // 2,  W // 2)

        x = self.in_proj(feat_last)      # [B, d4, H/16, W/16]
        x = self.up4(x, hw8)             # [B, d4, H/8,  W/8]
        x = self.up3(x, hw4)             # [B, d3, H/4,  W/4]
        x = self.up2(x, hw2)             # [B, d2, H/2,  W/2]
        x = self.up1(x, (H, W))          # [B, d1, H,    W]
        return self.head(x)              # [B, C, H, W]


