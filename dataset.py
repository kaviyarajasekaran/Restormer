import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import get_train_transforms, get_val_transforms

class PairedGrayDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, train=True):
        self.noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, "*")))
        self.clean_paths = [os.path.join(clean_dir, os.path.basename(p)) for p in self.noisy_paths]

        if len(self.noisy_paths) == 0:
            raise RuntimeError(f"No files found in {noisy_dir}")
        if not all(os.path.exists(p) for p in self.clean_paths):
            missing = [p for p in self.clean_paths if not os.path.exists(p)]
            raise RuntimeError(f"Missing clean pairs for some files. Example missing: {missing[0]}")

        self.tf = get_train_transforms() if train else get_val_transforms()

    def __len__(self):
        return len(self.noisy_paths)

    def _read_gray(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = img.astype(np.float32) / 255.0  # [0,1]
        return img

    def __getitem__(self, idx):
        noisy = self._read_gray(self.noisy_paths[idx])
        clean = self._read_gray(self.clean_paths[idx])

        aug = self.tf(image=noisy, mask=clean)
        noisy, clean = aug["image"], aug["mask"]

        noisy = torch.from_numpy(noisy).unsqueeze(0)  # [1,H,W]
        clean = torch.from_numpy(clean).unsqueeze(0)
        return noisy, clean
