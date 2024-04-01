import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

#First Train using all loss func until Epoch 176

TRAIN_PATH = "../datasets/DIV2K_train_HR/" #data train path
VAL_PATH = "../datasets//DIV2K_valid_LR_x8/" # data validation path
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar" #filename model generator yang kita simpan
CHECKPOINT_DISC = "disc.pth.tar" #filename model discriminator yang kita simpan
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
START_EPOCHS = 1
NUM_EPOCHS = 500
BATCH_SIZE = 3
NUM_WORKERS = 4
HIGH_RES = 96
RATIO = 4
LOW_RES = HIGH_RES // RATIO
IMG_CHANNELS = 3

# transformasi yang digunakan untuk gambar high resolution
highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)
# transformasi yang digunakan untuk gambar low resolution
lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
# transformasi yang digunakan untuk kedua gambar
both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)
#trainsformasi yang digunakan untuk melakukan prediction
test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
