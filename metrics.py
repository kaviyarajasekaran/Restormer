import torch
import torch.nn.functional as F

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return 99.0
    return (10.0 * torch.log10(1.0 / mse)).item()

@torch.no_grad()
def ssim_approx(x, y, window_size=11):
    # lightweight SSIM approx (same as in loss)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, 1, window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, window_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, window_size, 1, window_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, window_size // 2) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + 1e-8
    )
    return ssim.mean().item()
