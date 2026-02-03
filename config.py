DATA_ROOT = "/kaggle/input/input-data2k/2kdata/Dataset-1k/New_Data100" 
NOISY_SUBDIR = "Noisy"
CLEAN_SUBDIR = "Clean"

IMG_SIZE = 384
BATCH_SIZE = 8
NUM_WORKERS = 2

EPOCHS = 30
LR = 2e-4
WEIGHT_DECAY = 1e-4

MODEL_DIM = 48 
SAVE_PATH = "best_restormer_gray.pth"

W_L1 = 1.0
W_SSIM = 0.5
W_EDGE = 0.5

# Train/val split
TRAIN_SPLIT = 0.8
SEED = 42
