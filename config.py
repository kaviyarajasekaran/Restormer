DATA_ROOT = "/kaggle/input/YOUR_DATASET_NAME" 
NOISY_SUBDIR = "Noisy"
CLEAN_SUBDIR = "Clean"

IMG_SIZE = 384
BATCH_SIZE = 8
NUM_WORKERS = 2

EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 1e-4

MODEL_DIM = 48  # try 64 if GPU is strong
SAVE_PATH = "best_restormer_gray.pth"

W_L1 = 1.0
W_SSIM = 0.5
W_EDGE = 0.5

# Train/val split
TRAIN_SPLIT = 0.8
SEED = 42
