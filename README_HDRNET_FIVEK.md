# HDRNet + MIT-Adobe FiveK train+val+test usage

This folder uses expert C target

## 1. Export to HDRNet format

Expected folder for the downloaded images:
E:/henrique/data/MITAboveFiveK
└── data_split_manifest.json


```powershell
cd E:\henrique\HDRnet-PyTorch
.\.venv\Scripts\activate
python scripts\export_fivek_to_hdrnet.py
```

Expected folder:

```txt
E:\henrique\data\hdrnet_fivek
├── train\input
├── train\output
├── eval\input
├── eval\output
├── test\input 
└── test\output
```

## 2. Check the image pairs

```powershell
python scripts\check_hdrnet_pairs.py
```

Expected final message:

```txt
All pairs exist and have matching sizes.
```

## 3. Train HDRNet

Example train+val experiment:

```powershell
python train.py --epochs=30 --train_data_dir=E:/henrique/data/hdrnet_fivek/train --eval_data_dir=E:/henrique/data/hdrnet_fivek/eval --output_res 256 256 --ckpt_interval 20 --summary_interval 5 --cuda
```

## 4. Test inference

Use your latest checkpoint, for example:

```powershell
python test.py --ckpt_path=ckpts/epoch_XX_iter_XX.pt --test_img_path=E:/henrique/data/hdrnet_fivek/test/input/0000.jpg --cuda
```
