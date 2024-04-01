import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyImageFolder(Dataset):
    def __init__(self, root_dir): #constructor -> fungsi yang akan dijalankan ketika kita ngecreate kelas MyImageFolder
        super(MyImageFolder, self).__init__() # kalau kita mengimplementasikan inheritance di python kita perlu memanggil super classnya
        self.data = [] #menginisiasi list kosong untuk menyimpan keseluruhan data (data path per image)
        self.root_dir = root_dir
        #menjalankan operasi untuk mengambil keluruhan path image yang ada di dalam folder root_dir
        self.data = [os.path.join(root_dir, fl) for fl in os.listdir(root_dir)]
    #fungsi wajib yang dimiliki data module untuk melakukan return len()
    # data = MyImageFolder('.')
    # len(data)
    def __len__(self):
        return len(self.data)
    # fungsi -> data[0] -> low_res, high_res
    def __getitem__(self, index):
        # dia akan mengambil gambar pada index tersebut
        img_file = self.data[index]
        # dia akan melakukan load image menggunakan Pillow (library untuk image operation) -> convert menjadi numpy array
        image = np.array(Image.open(img_file))
        # image kita transformasikan menggunakan both transform
        image = config.both_transforms(image=image)["image"]
        # kita mentransformasikan untuk high resolution transform
        high_res = config.highres_transform(image=image)["image"]
        # mentransformasikan untuk low resolution trasnform
        low_res = config.lowres_transform(image=image)["image"]
        # melakukan return 2 data yaitu low res dan high res
        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir="Datasets/train_val_images/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()