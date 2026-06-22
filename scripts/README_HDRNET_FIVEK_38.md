# HDRNet + MIT-Adobe FiveK small 38-image experiment

This folder contains the scripts for:

- 30 train images
- 5 validation/eval images
- 3 test images
- Expert C target

## 1. Put the scripts in the correct places

Copy:

```txt
download_fivek_38.py
```

to:

```txt
E:\henrique\mit-adobe-fivek-dataset\download_fivek_38.py
```

Copy:

```txt
export_fivek_38_to_hdrnet.py
check_hdrnet_38_pairs.py
```

to:

```txt
E:\henrique\HDRnet-PyTorch\scripts\
```

## 2. Download the selected FiveK images

```powershell
cd E:\henrique\mit-adobe-fivek-dataset
.\.venv\Scripts\activate
python download_fivek_38.py
```

Expected output:

```txt
Manifest saved to:
E:/henrique/data/MITAboveFiveK/small_30_5_3_manifest.json
```

## 3. Export to HDRNet format

```powershell
cd E:\henrique\HDRnet-PyTorch
.\.venv\Scripts\activate
python scripts\export_fivek_38_to_hdrnet.py
```

Expected folder:

```txt
E:\henrique\data\hdrnet_fivek_38
├── train\input   30 images
├── train\output  30 images
├── eval\input    5 images
├── eval\output   5 images
├── test\input    3 images
└── test\output   3 images
```

## 4. Check the image pairs

```powershell
python scripts\check_hdrnet_38_pairs.py
```

Expected final message:

```txt
All pairs exist and have matching sizes.
```

## 5. Train HDRNet

Small test:

```powershell
python train.py --epochs=5 --train_data_dir=E:/henrique/data/hdrnet_fivek_38/train --eval_data_dir=E:/henrique/data/hdrnet_fivek_38/eval --output_res 256 256 --ckpt_interval 10 --summary_interval 2 --cuda
```

Longer small experiment:

```powershell
python train.py --epochs=100 --train_data_dir=E:/henrique/data/hdrnet_fivek_38/train --eval_data_dir=E:/henrique/data/hdrnet_fivek_38/eval --output_res 256 256 --ckpt_interval 20 --summary_interval 5 --cuda
```

## 6. Test inference

Use your latest checkpoint, for example:

```powershell
python test.py --ckpt_path=ckpts/epoch_99_iter_800.pt --test_img_path=E:/henrique/data/hdrnet_fivek_38/test/input/0001.jpg --cuda
```
