import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        var = x.var(dim=(2,3), keepdim=True, unbiased=False)
        mean = x.mean(dim=(2,3), keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion=2.66):
        super().__init__()
        hidden = int(dim * ffn_expansion)
        self.project_in = nn.Conv2d(dim, hidden*2, 1, 1, 0)
        self.dwconv = nn.Conv2d(hidden*2, hidden*2, 3, 1, 1, groups=hidden*2)
        self.project_out = nn.Conv2d(hidden, dim, 1, 1, 0)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, 1, 1, 0)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, 3, 1, 1, groups=dim*3)
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, c//self.num_heads, h*w)
        k = k.view(b, self.num_heads, c//self.num_heads, h*w)
        v = v.view(b, self.num_heads, c//self.num_heads, h*w)

        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        attn = (q.transpose(2,3) @ k) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v.transpose(2,3)
        out = out.transpose(2,3).contiguous().view(b, c, h, w)

        out = self.project_out(out)
        return out

class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn  = MDTA(dim, num_heads=num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = GDFN(dim, ffn_expansion=ffn_expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim*2, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim//2, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)

class Restormer(nn.Module):
    def __init__(self, inp_ch=1, out_ch=1, dim=48,
                 num_blocks=(2,3,4,3), num_refine=2,
                 heads=(1,2,4,8), ffn_expansion=2.66):
        super().__init__()
        self.patch_embed = nn.Conv2d(inp_ch, dim, 3, 1, 1)

        self.enc1 = nn.Sequential(*[RestormerBlock(dim, heads[0], ffn_expansion) for _ in range(num_blocks[0])])
        self.down1 = Downsample(dim)

        self.enc2 = nn.Sequential(*[RestormerBlock(dim*2, heads[1], ffn_expansion) for _ in range(num_blocks[1])])
        self.down2 = Downsample(dim*2)

        self.enc3 = nn.Sequential(*[RestormerBlock(dim*4, heads[2], ffn_expansion) for _ in range(num_blocks[2])])
        self.down3 = Downsample(dim*4)

        self.latent = nn.Sequential(*[RestormerBlock(dim*8, heads[3], ffn_expansion) for _ in range(num_blocks[3])])

        self.up3 = Upsample(dim*8)
        self.dec3 = nn.Sequential(*[RestormerBlock(dim*4, heads[2], ffn_expansion) for _ in range(2)])

        self.up2 = Upsample(dim*4)
        self.dec2 = nn.Sequential(*[RestormerBlock(dim*2, heads[1], ffn_expansion) for _ in range(2)])

        self.up1 = Upsample(dim*2)
        self.dec1 = nn.Sequential(*[RestormerBlock(dim, heads[0], ffn_expansion) for _ in range(2)])

        self.refine = nn.Sequential(*[RestormerBlock(dim, heads[0], ffn_expansion) for _ in range(num_refine)])
        self.output = nn.Conv2d(dim, out_ch, 3, 1, 1)

    def forward(self, x):
        inp = x
        x = self.patch_embed(x)

        e1 = self.enc1(x)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        e3 = self.enc3(x)
        x = self.down3(e3)

        x = self.latent(x)

        x = self.up3(x) + e3
        x = self.dec3(x)

        x = self.up2(x) + e2
        x = self.dec2(x)

        x = self.up1(x) + e1
        x = self.dec1(x)

        x = self.refine(x)
        out = self.output(x)

        out = torch.clamp(out + inp, 0.0, 1.0)
        return out
