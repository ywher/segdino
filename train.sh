# tn3k
# dataset="tn3k"
# img_dir_name="img"
# label_dir_name="label"
# img_ext=".jpg"
# mask_ext=".jpg"

# xh_kidney
dataset="xh_kidney"
img_dir_name="image"
label_dir_name="label"
img_ext=".jpg"
mask_ext=".png"


input_h=256
input_w=256
dino_size="b"
dino_ckpt="./web_pth/dinov3_vitb16.pth"

CUDA_VISIBLE_DEVICES=0 python train_segdino.py \
  --data_dir ./segdata \
  --dataset $dataset \
  --img_dir_name $img_dir_name \
  --label_dir_name $label_dir_name \
  --img_ext $img_ext \
  --mask_ext $mask_ext \
  --input_h $input_h \
  --input_w $input_w \
  --in_ch 1 \
  --num_classes 1 \
  --repo_dir ./dinov3 \
  --dino_size $dino_size \
  --dino_ckpt $dino_ckpt \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4
#   --freeze_backbone
