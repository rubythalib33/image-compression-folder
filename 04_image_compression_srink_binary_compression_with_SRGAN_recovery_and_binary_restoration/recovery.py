from utils import *
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import Generator
from torchvision.utils import make_grid

CKPT_PATH = "ckpt_weights/gen.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator()
gen.load_state_dict(torch.load(CKPT_PATH)["state_dict"])
gen.eval().to(DEVICE)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

def recovery_srgan(img):
    with torch.no_grad():
        upscaled_img = gen(
            test_transform(image=np.asarray(img))["image"]
            .unsqueeze(0)
            .to(DEVICE)
        )
        upscaled_img = upscaled_img * 0.5 + 0.5
        upscaled_img = make_grid(upscaled_img)
        upscaled_img = upscaled_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        
        return upscaled_img


def split(img):
    bottom_img = bit8to4(img)
    top_img = img - bottom_img*16
    return np.vstack((top_img, bottom_img))


def recovery_binary(img):
    split_img = split(img)
    img_8_bit = bit4to8(split_img)

    return img_8_bit


if __name__ == '__main__':
    from PIL import Image

    IMAGE_PATH = "../03_image_compression_srink_binary_compression_with_SRGAN_recovery/asset/Compressed/result.png"
    RESULT_PATH = "asset/Restored/result.png"

    image = np.asarray(Image.open(IMAGE_PATH))

    image = recovery_binary(image)
    image = recovery_srgan(image)

    image = Image.fromarray(image)
    image.save(RESULT_PATH)