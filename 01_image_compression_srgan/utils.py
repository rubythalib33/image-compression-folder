import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import numpy as np
from albumentations import Resize

def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(low_res_folder, gen):
    os.system("rm saved/*")
    os.makedirs('saved', exist_ok=True)
    files = os.listdir(low_res_folder)
    np.random.shuffle(files)
    gen.eval()
    for file in files[:10]:
        image = Image.open(low_res_folder + file)
        try:
            with torch.no_grad():
                upscaled_img = gen(
                    config.test_transform(image=np.asarray(image))["image"]
                    .unsqueeze(0)
                    .to(config.DEVICE)
                )
            save_image(upscaled_img * 0.5 + 0.5, f"saved/{file}")
        except Exception as e: print('Memory insufficient for that image',e)
    gen.train()

def compress(img: np.ndarray, ratio:int):
    resized = Resize(width=img.shape[1]//ratio, height=img.shape[0]//ratio, interpolation=Image.BICUBIC)(image=img)["image"]
    return resized