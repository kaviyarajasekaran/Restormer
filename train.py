import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import config as cfg
from data.dataset import PairedGrayDataset
from models.restormer import Restormer
from losses.loss import LineRestoreLoss
from utils.seed import set_seed
from utils.metrics import psnr, ssim_approx
from utils.checkpoint import save_checkpoint

@torch.no_grad()
def validate(model, loader, device, loss_fn):
    model.eval()
    total_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        loss = loss_fn(pred, clean)

        total_loss += loss.item()
        total_psnr += psnr(pred, clean)
        total_ssim += ssim_approx(pred, clean)

    n = len(loader)
    return total_loss / n, total_psnr / n, total_ssim / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=cfg.DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--dim", type=int, default=cfg.MODEL_DIM)
    args = parser.parse_args()

    set_seed(cfg.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    noisy_dir = os.path.join(args.data_root, cfg.NOISY_SUBDIR)
    clean_dir = os.path.join(args.data_root, cfg.CLEAN_SUBDIR)

    full_ds = PairedGrayDataset(noisy_dir, clean_dir, train=True)

    n = len(full_ds)
    split = int(cfg.TRAIN_SPLIT * n)
    train_paths = full_ds.noisy_paths[:split]
    val_paths = full_ds.noisy_paths[split:]

    train_ds = PairedGrayDataset(noisy_dir, clean_dir, train=True)
    val_ds   = PairedGrayDataset(noisy_dir, clean_dir, train=False)

    train_ds.noisy_paths = train_paths
    train_ds.clean_paths = [os.path.join(clean_dir, os.path.basename(p)) for p in train_paths]

    val_ds.noisy_paths = val_paths
    val_ds.clean_paths = [os.path.join(clean_dir, os.path.basename(p)) for p in val_paths]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = Restormer(inp_ch=1, out_ch=1, dim=args.dim).to(device)

    loss_fn = LineRestoreLoss(cfg.W_L1, cfg.W_SSIM, cfg.W_EDGE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for noisy, clean in pbar:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                pred = model(noisy)
                loss = loss_fn(pred, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        train_loss = running / len(train_loader)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, device, loss_fn)

        print(f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | PSNR {val_psnr:.2f} | SSIM {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(cfg.SAVE_PATH, model, epoch, best_psnr)
            print(f"Saved best model: {cfg.SAVE_PATH} (PSNR {best_psnr:.2f})")

if __name__ == "__main__":
    main()
