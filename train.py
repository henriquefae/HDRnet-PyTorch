import numpy as np
import os
import time
import torch
from argparse import ArgumentParser
from datasets import Train_Dataset, Eval_Dataset
from models import HDRnetModel
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import psnr, print_params, load_train_ckpt, save_model_stats, plot_per_check, AvgMeter
from loss import build_loss, to_lpips_range
import re
import json
import lpips
from skimage.color import rgb2lab


def lab_error_metrics(output, target):
    """
    Compute validation errors in CIE-L*a*b* space.

    Inputs:
    - output: torch tensor [N, 3, H, W], range approximately [0, 1]
    - target: torch tensor [N, 3, H, W], range [0, 1]

    Returns:
    - mean Lab L2 error over pixels
    - mean absolute L* error over pixels
    """

    output_np = output.detach().clamp(0.0, 1.0).cpu().numpy()
    target_np = target.detach().clamp(0.0, 1.0).cpu().numpy()

    # Convert NCHW -> NHWC
    output_np = np.transpose(output_np, (0, 2, 3, 1))
    target_np = np.transpose(target_np, (0, 2, 3, 1))

    lab_l2_values = []
    lab_l_values = []

    for i in range(output_np.shape[0]):
        output_lab = rgb2lab(output_np[i])
        target_lab = rgb2lab(target_np[i])

        diff = output_lab - target_lab

        lab_l2 = np.sqrt(np.sum(diff ** 2, axis=2)).mean()
        lab_l = np.abs(diff[:, :, 0]).mean()

        lab_l2_values.append(lab_l2)
        lab_l_values.append(lab_l)

    return float(np.mean(lab_l2_values)), float(np.mean(lab_l_values))


def parse_epoch_iter_from_ckpt_name(ckpt_path):
    """
    Extract epoch and iteration from checkpoint filename.

    Example:
        epoch_15_iter_17000.pt
        -> epoch = 15
        -> iteration = 17000
    """
    fname = os.path.basename(ckpt_path)

    match = re.search(r"epoch_(\d+)_iter_(\d+)\.pt", fname)

    if match is None:
        raise ValueError(
            f"Could not parse epoch/iteration from checkpoint name: {fname}. "
            "Expected format: epoch_X_iter_Y.pt"
        )

    epoch = int(match.group(1))
    iteration = int(match.group(2))

    return epoch, iteration


def resolve_resume_ckpt_path(params):
    """
    Allows:
        --resume_ckpt epoch_15_iter_17000.pt

    or:
        --resume_ckpt ./ckpts/ckpts_run/epoch_15_iter_17000.pt

    If only a filename is given, it searches inside params['ckpt_dir'].
    """
    resume_ckpt = params["resume_ckpt"]

    if resume_ckpt is None:
        return None

    if os.path.isabs(resume_ckpt) or os.path.dirname(resume_ckpt) != "":
        ckpt_path = resume_ckpt
    else:
        ckpt_path = os.path.join(params["ckpt_dir"], resume_ckpt)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    return ckpt_path


def load_resume_ckpt(model, ckpt_path, device):
    """
    Load model weights from a checkpoint.
    """
    print(f"\nLoading resume checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        raise ValueError("Unsupported checkpoint format.")

    model.load_state_dict(state_dict)
    print("Checkpoint weights loaded successfully.")


def load_existing_stats(stats_dir):
    """
    Continue stats.json if it already exists.
    If not found, start new stats.
    """
    stats_path = os.path.join(stats_dir, "stats.json")

    if os.path.exists(stats_path):
        print(f"Loading existing stats from: {stats_path}")
        with open(stats_path, "r") as f:
            stats = json.load(f)

        stats.setdefault("train_loss", [])
        stats.setdefault("train_psnr", [])
        stats.setdefault("valid_psnr", [])
        stats.setdefault("valid_lpips", [])
        stats.setdefault("valid_lab_l2", [])
        stats.setdefault("valid_lab_l", [])

        return stats

    return {
        "train_loss": [],
        "train_psnr": [],
        "valid_psnr": [],
        "valid_lpips": [],
        "valid_lab_l2": [],
        "valid_lab_l": []
    }


def print_image_tensor_stats(name, x):
    x_float = x.float()

    print(
        f"{name}: "
        f"shape={tuple(x.shape)}, "
        f"dtype={x.dtype}, "
        f"min={x_float.min().item():.6f}, "
        f"max={x_float.max().item():.6f}, "
        f"mean={x_float.mean().item():.6f}"
    )


def train(params, train_loader, valid_loader, model, device, start_epoch=0, start_iteration=0):
    # Optimization
    optimizer = Adam(model.parameters(), params["learning_rate"], weight_decay=1e-8)

    # # Learning rate adjustment
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     patience=params["epochs"] / 4,
    #     factor=0.5,
    #     verbose=True
    # )

    # Training loss
    criterion = build_loss(params["loss"]).to(device)
    criterion.eval()

    print(f"\nTraining loss: {params['loss']}")

    # Validation metric: LPIPS with AlexNet.
    # This is only used for validation statistics, not for training.
    lpips_alex = lpips.LPIPS(net="alex").to(device)

    for p in lpips_alex.parameters():
        p.requires_grad = False

    lpips_alex.eval()

    # Training meters
    train_loss_meter = AvgMeter()
    train_psnr_meter = AvgMeter()

    stats = load_existing_stats(params["stats_dir"])

    iteration = start_iteration
    old_time = time.time()

    print(f"\nStarting training from epoch={start_epoch}, iteration={iteration}")

    for epoch in range(start_epoch, params["epochs"]):
        for batch_idx, (low, full, target) in enumerate(train_loader):
            iteration += 1
            model.train()

            low = low.to(device)
            full = full.to(device)
            target = target.to(device)

            # Normalize to [0, 1] on GPU.
            # Expecting uint16 images. Change to 255.0 if using uint8 images.
            if params["hdr"]:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)
            else:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)

            target = torch.div(target, 65535.0)

            if epoch == start_epoch and batch_idx == 0:
                print("\nNormalized tensor sanity check:")
                print_image_tensor_stats("low normalized", low)
                print_image_tensor_stats("full normalized", full)
                print_image_tensor_stats("target normalized", target)

            output = model(low, full)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % params["summary_interval"] == 0:
                train_loss_meter.update(loss.item())

                train_psnr = psnr(output, target).item()
                train_psnr_meter.update(train_psnr)

                new_time = time.time()

                print(
                    "[%d/%d] Iteration: %d | Loss: %.4f | PSNR: %.4f | Time: %.2fs"
                    % (
                        epoch + 1,
                        params["epochs"],
                        iteration,
                        loss.item(),
                        train_psnr,
                        new_time - old_time
                    )
                )

                old_time = new_time

            if iteration % params["ckpt_interval"] == 0:
                stats["train_loss"].append(train_loss_meter.avg)
                train_loss_meter.reset()

                stats["train_psnr"].append(train_psnr_meter.avg)
                train_psnr_meter.reset()

                valid_psnr, valid_lpips, valid_lab_l2, valid_lab_l = eval(
                    params,
                    valid_loader,
                    model,
                    device,
                    lpips_alex
                )

                stats['valid_psnr'].append(valid_psnr)
                stats['valid_lpips'].append(valid_lpips)
                stats['valid_lab_l2'].append(valid_lab_l2)
                stats['valid_lab_l'].append(valid_lab_l)

                plot_per_check(
                    params["stats_dir"],
                    "Train loss",
                    stats["train_loss"],
                    "Training loss"
                )

                plot_per_check(
                    params["stats_dir"],
                    "Train PSNR",
                    stats["train_psnr"],
                    "PSNR (dB)"
                )

                plot_per_check(
                    params["stats_dir"],
                    "Valid PSNR",
                    stats["valid_psnr"],
                    "PSNR (dB)"
                )

                plot_per_check(
                    params["stats_dir"],
                    "Valid LPIPS",
                    stats["valid_lpips"],
                    "LPIPS distance"
                )

                plot_per_check(params['stats_dir'], 'Valid Lab L2', stats['valid_lab_l2'], 'Mean Lab L2 error')

                plot_per_check(params['stats_dir'], 'Valid L error', stats['valid_lab_l'], 'Mean L* error')

                ckpt_fname = "epoch_" + str(epoch) + "_iter_" + str(iteration) + ".pt"
                save_model_stats(model, params, ckpt_fname, stats)


def eval(params, valid_loader, model, device, lpips_metric):
    model.eval()

    psnr_meter = AvgMeter()
    lpips_meter = AvgMeter()
    lab_l2_meter = AvgMeter()
    lab_l_meter = AvgMeter()

    with torch.no_grad():
        for batch_idx, (low, full, target) in enumerate(valid_loader):
            low = low.to(device)
            full = full.to(device)
            target = target.to(device)

            # Normalize to [0, 1] on GPU.
            # Expecting uint16 images. Change to 255.0 if using uint8 images.
            if params["hdr"]:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)
            else:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)

            target = torch.div(target, 65535.0)

            output = model(low, full)

            save_image(
                output.clamp(0.0, 1.0),
                os.path.join(params["eval_out"], str(batch_idx) + ".png")
            )

            eval_psnr = psnr(output, target).item()
            psnr_meter.update(eval_psnr)

            output_lpips = to_lpips_range(output.clamp(0.0, 1.0))
            target_lpips = to_lpips_range(target.clamp(0.0, 1.0))

            eval_lpips = lpips_metric(output_lpips, target_lpips).mean().item()
            lpips_meter.update(eval_lpips)

            eval_lab_l2, eval_lab_l = lab_error_metrics(output, target)

            lab_l2_meter.update(eval_lab_l2)
            lab_l_meter.update(eval_lab_l)

    print("Validation PSNR:   ", psnr_meter.avg)
    print("Validation LPIPS:  ", lpips_meter.avg)
    print("Validation Lab L2: ", lab_l2_meter.avg)
    print("Validation L*:     ", lab_l_meter.avg)

    return psnr_meter.avg, lpips_meter.avg, lab_l2_meter.avg, lab_l_meter.avg


def parse_args():
    parser = ArgumentParser(description="HDRnet training")

    # Training, logging and checkpointing parameters
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--ckpt_interval", default=600, type=int, help="Interval for saving checkpoints, unit is iteration")
    parser.add_argument("--ckpt_dir", default="./ckpts", type=str, help="Checkpoint directory")
    parser.add_argument("--stats_dir", default="./stats", type=str, help="Statistics directory")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--summary_interval", default=10, type=int)

    # Loss
    parser.add_argument(
        "--loss",
        default="mse",
        choices=["mse", "l1_vgg_r12"],
        help="Training loss to use."
    )

    # Data pipeline and data augmentation
    parser.add_argument("--batch_size", default=4, type=int, help="Size of a mini-batch")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Dataset path")
    parser.add_argument("--eval_data_dir", default=None, type=str, help="Directory with the validation data.")
    parser.add_argument("--eval_out", default="./outputs", type=str, help="Validation output path")
    parser.add_argument("--hdr", action="store_true", help="Handle HDR image")

    # Model parameters
    parser.add_argument("--batch_norm", action="store_true", help="Use batch normalization")
    parser.add_argument("--input_res", default=256, type=int, help="Resolution of the down-sampled input")
    parser.add_argument("--output_res", default=(1024, 1024), type=int, nargs=2, help="Resolution of the guidemap/final output")

    # Run name
    parser.add_argument("--run_name", type=str, default=None)

    # Resume checkpoint
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Checkpoint file to resume from, e.g. epoch_15_iter_17000.pt"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Random seeds
    seed = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Parse training parameters
    params = vars(parse_args())

    if params["run_name"] is not None:
        run_name = params["run_name"]

        params["ckpt_dir"] = f"./ckpts/ckpts_{run_name}"
        params["stats_dir"] = f"./stats/stats_{run_name}"
        params["eval_out"] = f"./outputs/outputs_{run_name}"

    print_params(params)

    # Folders
    os.makedirs(params["ckpt_dir"], exist_ok=True)
    os.makedirs(params["stats_dir"], exist_ok=True)
    os.makedirs(params["eval_out"], exist_ok=True)

    # Dataloader for training
    train_dataset = Train_Dataset(params)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True
    )

    # Dataloader for validation
    valid_dataset = Eval_Dataset(params)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1
    )

    # Model for training
    model = HDRnetModel(params)

    if params["cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    start_epoch = 0
    start_iteration = 0

    if params["resume_ckpt"] is not None:
        ckpt_path = resolve_resume_ckpt_path(params)

        resume_epoch, resume_iteration = parse_epoch_iter_from_ckpt_name(ckpt_path)

        load_resume_ckpt(model, ckpt_path, device)

        start_epoch = resume_epoch
        start_iteration = resume_iteration
    else:
        # Original behavior: automatically load latest checkpoint from ckpt_dir
        load_train_ckpt(model, params["ckpt_dir"])

    train(
        params,
        train_loader,
        valid_loader,
        model,
        device,
        start_epoch=start_epoch,
        start_iteration=start_iteration
    )