import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): 
        return self.act(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),
        )
    def forward(self, x): 
        return self.block(x)

class UpNoSkip(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, out_hw):
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return self.conv(x)

# ===== 解码头：仅用 feats[-1] =====
class DinoLastUNetDecoder(nn.Module):
    def __init__(self, C_in=384, dec_chs=(256, 192, 128, 96), num_classes=2):
        super().__init__()
        d4, d3, d2, d1 = dec_chs
        self.in_proj = ConvBNReLU(C_in, d4, k=1, s=1, p=0)
        self.up4 = UpNoSkip(d4, d4)   # H/16 -> H/8
        self.up3 = UpNoSkip(d4, d3)   # H/8  -> H/4
        self.up2 = UpNoSkip(d3, d2)   # H/4  -> H/2
        self.up1 = UpNoSkip(d2, d1)   # H/2  -> H
        self.head = nn.Conv2d(d1, num_classes, kernel_size=1)

    def forward(self, feat_last, out_hw):
        H, W = out_hw
        x = self.in_proj(feat_last)
        x = self.up4(x, (H//8,  W//8))
        x = self.up3(x, (H//4,  W//4))
        x = self.up2(x, (H//2,  W//2))
        x = self.up1(x, (H, W))
        return self.head(x)

class DinoV3SegModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int,
                 in_ch: int = 3, dec_chs=(256,192,128,96),
                 use_intermediate_layers: bool = True, last_layer_idx: int = -1):
        super().__init__()
        # 1) 冻结 backbone
        self.backbone = backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.in_ch = in_ch
        self.use_intermediate_layers = use_intermediate_layers
        self.last_layer_idx = last_layer_idx

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256) 
            if self.use_intermediate_layers and hasattr(self.backbone, "get_intermediate_layers"):
                feats = self.backbone.get_intermediate_layers(dummy, n=range(12), reshape=True, norm=True)
                C_in = feats[self.last_layer_idx].shape[1]
            else:
                out = self.backbone.forward_features(dummy)
                if isinstance(out, dict):
                    tokens = out.get("x_norm_patchtokens", None) or out.get("x_patchtokens", None) or out.get("x", None)
                else:
                    tokens = out[:, 1:, :]  
                C_in = tokens.shape[-1]

        self.decoder = DinoLastUNetDecoder(C_in=C_in, dec_chs=dec_chs, num_classes=num_classes)

    def forward(self, x):
        if self.in_ch == 1 and x.shape[1] == 1:
            x_in = x.repeat(1, 3, 1, 1)
        else:
            x_in = x

        H, W = x_in.shape[-2:]
        assert H % 16 == 0 and W % 16 == 0, "输入尺寸需是16的倍数（如256/512）"

        with torch.no_grad():
            feats = self.backbone.get_intermediate_layers(x_in, n=range(12), reshape=True, norm=True)
            feat_last = feats[self.last_layer_idx]  # [B, C_in, H/16, W/16]

        logits = self.decoder(feat_last, out_hw=(H, W))  # [B, num_classes, H, W]
        return logits

    def freeze_backbone(self, freeze: bool = True):
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = not (freeze)
