import torch

def save_checkpoint(path, model, epoch, best_psnr):
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr
    }, path)

def load_checkpoint(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt
