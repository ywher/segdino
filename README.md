# SegDINO

SegDINO is a segmentation framework built on top of DINOv3 features.

---
## Environment Setup

```bash
conda create -n segdino python=3.10.16
conda activate segdino

pip install -r requirements.txt
````

Clone the DINOv3 repository:

```bash
git clone https://github.com/facebookresearch/dinov3.git
```

Download DINO pretrained weights and place them in:

```
./web_pth
```

---

## Dataset Preparation

Datasets can be downloaded here: [Baidu](#) | [Google](#)

Organize datasets in the following structure:

```
./segdata/tn3k
./segdata/kvasir
./segdata/isic
```

Each dataset folder should contain an `image` directory and a `mask` directory.

---

## Pretrained Weights

We provide trained SegDINO model weights: [Baidu](#) | [Google](#)

Place the downloaded `.pth` files in a proper directory (e.g., `./checkpoints/`).

---

## Training

Example training command:

```bash
python train_segdino_b.py \
  --data_dir ./segdata \
  --dataset tn3k \
  --img_dir_name image \
  --label_dir_name mask \
  --mask_ext '.jpg' \
  --input_h 256 --input_w 256 \
  --repo_dir ./dinov3 \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4
```

---

## Testing

Example testing command:

```bash
python test_segdino.py \
  --data_dir ./segdata \
  --dataset tn3k \
  --input_h 256 --input_w 256 \
  --dino_size s \
  --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --ckpt ./tn3k.pth \
  --repo_dir ./dinov3 \
  --img_dir_name image \
  --label_dir_name mask \
  --mask_ext '.jpg'
```

---

## Notes

* Make sure the pretrained DINO weights (`.pth` file) are correctly downloaded and placed under `./web_pth`.
* Modify paths as needed for your environment.
* Training and testing configurations (e.g., image size, batch size, learning rate) can be adjusted via command-line arguments.

---

## Acknowledgements

We would like to thank Facebook Research for open-sourcing [DINOv3](https://github.com/facebookresearch/dinov3) and all contributors to the datasets used in this project.

