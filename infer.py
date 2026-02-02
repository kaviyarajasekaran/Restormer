import os
import argparse
import cv2
import numpy as np
import torch

from models.restormer import Restormer

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    x = img.astype(np.float32) / 255.0
    return x

def save_gray(path, x01):
    x01 = np.clip(x01, 0, 1)
    out = (x01 * 255.0).astype(np.uint8)
    cv2.imwrite(path, out)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="restored.png")
    parser.add_argument("--dim", type=int, default=48)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Restormer(inp_ch=1, out_ch=1, dim=args.dim).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x = read_gray(args.input)
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    pred = model(t).squeeze().detach().cpu().numpy()
    save_gray(args.output, pred)

    print("Saved:", args.output)

if __name__ == "__main__":
    main()
