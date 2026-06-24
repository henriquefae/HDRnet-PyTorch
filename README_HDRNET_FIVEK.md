# HDRNet + MIT-Adobe FiveK train+val+test usage

This folder uses expert C target

## 1. Export to HDRNet format

Expected folder for the downloaded images:

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
python train.py --epochs=1000 --train_data_dir=E:/henrique/data/hdrnet_fivek/train --eval_data_dir=E:/henrique/data/hdrnet_fivek/eval --output_res 1024 1024 --ckpt_interval 1000 --summary_interval 100 --run_name experimento1 --cuda
```

## 4. Test inference

Use your latest checkpoint, for example:

```powershell
python test.py --ckpt_path=ckpts/ckpts_experiment2/epoch_99_iter_2500.pt --test_img_path=E:/henrique/data/hdrnet_fivek/test/input/0000.tif --cuda
```
