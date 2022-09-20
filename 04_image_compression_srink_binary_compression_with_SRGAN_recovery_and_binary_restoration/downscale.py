from PIL import Image
import os

ROOT = 'Datasets/test_images'
OUTPUT = 'Datasets/low_test_images'

files = os.listdir(ROOT)

for file in files:
    path = os.path.join(ROOT, file)
    image = Image.open(path)
    wsize = image.size[0]//4
    hsize = image.size[1]//4
    image = image.resize((wsize,hsize), Image.BICUBIC)
    image.save(os.path.join(OUTPUT,file))