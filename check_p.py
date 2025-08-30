import torch
import torch.nn as nn
from collections import OrderedDict

# === your model defs here (DPTHead, DPT, backbone 构建等) ===
# from your_file import DPT, DPTHead, build_backbone ...
# backbone = build_backbone(...)
# model = DPT(encoder_size='base', nclass=21, features=128,
#             out_channels=[96,192,384,768], backbone=backbone)
# from dpt import DPT
backbone = torch.hub.load('/vip_media/sicheng/DataShare/tmi_re/segdino_good/dinov3', 'dinov3_vits16', source='local', weights='/vip_media/sicheng/DataShare/tmi_re/segdino_good/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')

# backbone = torch.hub.load('/vip_media/sicheng/DataShare/tmi_re/segdino_good/dinov3', 'dinov3_vitb16', source='local', weights='/vip_media/sicheng/DataShare/tmi_re/segdino_good/web_pth/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')

def extract_state_dict(obj):
    """Return a state_dict no matter the checkpoint format."""
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "net", "weights", "ema_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj  # already a flat state_dict

def strip_common_prefixes(state_dict):
    """Remove common wrappers like 'module.' / 'model.' from keys."""
    def rm_pref(k):
        for p in ("module.", "model."):
            if k.startswith(p):
                return k[len(p):]
        return k
    return {rm_pref(k): v for k, v in state_dict.items()}

def count_params(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train

def human(n: int):
    return f"{n:,}  ({n/1e6:.2f}M)"


# ---------- load checkpoint ----------
# from dpt import DPT
# model = DPT(nclass=1, backbone=backbone)
# ckpt_path = "/vip_media/sicheng/DataShare/tmi_re/segdinov3/segdino_b/runs/upsegdinov9_256_kvasir/ckpts/best_ep050_dice0.8791_0.8090.pth"


from dpt import DPT
model = DPT(nclass=1, backbone=backbone)
# ckpt_path = "/vip_media/sicheng/DataShare/tmi_re/segdino/runs/segdino_s_256_kvasir/ckpts/best_ep002_dice0.8070_0.7116.pth"
ckpt_path = '/vip_media/sicheng/DataShare/tmi_re/segdino/runs/segdino_s_256_ISIC/ckpts/best_ep021_dice0.8621_0.7814.pth'
ckpt = torch.load(ckpt_path, map_location="cpu")
sd = extract_state_dict(ckpt)
sd = strip_common_prefixes(sd)

# 先尝试整模型加载（strict=False 以便忽略不匹配的键）
missing, unexpected = model.load_state_dict(sd, strict=False)

# 如果权重里只包含 head（或以 "head." 开头），再单独喂给 head
head_sd = {k.replace("head.", "", 1): v for k, v in sd.items() if k.startswith("head.")}
if head_sd:
    _missing, _unexpected = model.head.load_state_dict(head_sd, strict=False)
    # 可按需打印 _missing / _unexpected

# ---------- print param counts ----------
# 如果你的属性名是 dpt_head，请用 getattr(model, "dpt_head", model.head)
head_module = getattr(model, "dpt_head", model.head)
print(head_module)

t_all, tr_all = count_params(model)
t_head, tr_head = count_params(head_module)

print(f"[Whole Model]     total={human(t_all)}, trainable={human(tr_all)}")
print(f"[DPT Head only]   total={human(t_head)}, trainable={human(tr_head)}")

# (可选) 冻结backbone后再看可训练参数量
# model.lock_backbone()
# _, tr_after_freeze = count_params(model)
# print(f"[After freezing backbone] trainable={human(tr_after_freeze)}")
