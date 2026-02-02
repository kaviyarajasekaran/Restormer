import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, self.window_size, 1, self.window_size // 2)
        mu_y = F.avg_pool2d(y, self.window_size, 1, self.window_size // 2)

        sigma_x = F.avg_pool2d(x * x, self.window_size, 1, self.window_size // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, self.window_size, 1, self.window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, self.window_size, 1, self.window_size // 2) - mu_x * mu_y

        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + 1e-8
        )
        return 1 - ssim.mean()

def sobel_edges(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-8)

class LineRestoreLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.5, w_edge=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(window_size=11)
        self.w_l1, self.w_ssim, self.w_edge = w_l1, w_ssim, w_edge

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        ssim = self.ssim(pred, target)
        edge = self.l1(sobel_edges(pred), sobel_edges(target))
        return self.w_l1*l1 + self.w_ssim*ssim + self.w_edge*edge
