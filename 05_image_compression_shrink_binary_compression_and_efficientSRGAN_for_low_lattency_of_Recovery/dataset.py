import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import bit8to4, bit4to8


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.data = [os.path.join(root_dir, fl) for fl in os.listdir(root_dir)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        image = np.array(Image.open(img_file))
        image = config.both_transforms(image=image)["image"]
        high_res = config.highres_transform(image=image)["image"]
        image = bit4to8(bit8to4(image))
        low_res = config.lowres_transform(image=image)["image"]
        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir="Datasets/train_val_images/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()