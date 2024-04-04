import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder

torch.backends.cudnn.benchmark = True

#train_fn adalah sebuah fungsi yang akan dijalankan di setiap epochnya
def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    # loader: dataloader -> object data yang kita buat, dan sudah dimasukan kedalam PyTorchDataLoader
    # disc: discriminator model kita yang sudah di definisikan
    # gen: generative model kita yang sudah di definisikan
    # opt_gen: optimizer yang digunakan untuk update weight si generator model
    # opt_disc: optimizer yang digunakan untuk update weight si discriminator model
    # mse: loss_function mse
    # bce: binary cross entropy loss, yaitu loss function yang digunakan oleh diskriminator
    # vgg_loss: feature loss yang kita bahas kemarin

    #tqdm itu adalah sebuah python package yang digunakan untuk memvisualisasikan sebuah looping yang dengan itu dia bisa return value
    loop = tqdm(loader, leave=True)

    #disetiap iterasi dari idx -> urutan training step, get low res image, high res image
    for idx, (low_res, high_res) in enumerate(loop):
        #kita pindah device high res image ke config device
        high_res = high_res.to(config.DEVICE)
        #kita pindah device low res image ke config device
        low_res = low_res.to(config.DEVICE)
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        #kita ngegenerate high res image dari model generator
        fake = gen(low_res)
        #kita menjalankan prediksi apakah dia fake or real dari data asli
        disc_real = disc(high_res)
        #kita menjalankan prediksi apakah dia fake or real dari data generated(fake)
        disc_fake = disc(fake.detach())
        #kita akan menghitung lossnya dengan binary cross entropy terhadap data real dengan hasil prediksi discriminator terhadap data real
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        #kita akan menghitung lossnya dengan binary cross entropy terhadap data fake dengan hasil prediksi discriminator terhadap data fake
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        #lossnya kita jadikan satu
        loss_disc = disc_loss_fake + disc_loss_real
        # print(f'discrimantor loss:{loss_disc}')
        #proses inisiasi optimizer
        opt_disc.zero_grad()
        #backpropagation
        loss_disc.backward()
        #update weight terhadap model discriminator
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # kita melakukan prediksi hasil discriminator terhadap data fake, 
        disc_fake = disc(fake)
        # mengkalkulasikan l2_loss mse terhadap data fake dan data real
        l2_loss = mse(fake, high_res)
        # adversarial_loss, menghitung binary cross entropy dari output disc_fake, terhadap sebuah matriks yang kita buat berisi 1 semua
        adversarial_loss = bce(disc_fake, torch.ones_like(disc_fake))
        # menjalankan feature loss terhadap data fake dan data real
        loss_for_vgg = vgg_loss(fake, high_res)
        # gen_loss = loss_for_vgg + adversarial_loss + l2_loss
        #setelah itu kita jumlahkan
        gen_loss = 6e-2 * loss_for_vgg + 1e-2 * adversarial_loss + 0.92*l2_loss
        # print(f'Generative loss:{gen_loss}')

        #inisiasi weight si optimizer generator
        opt_gen.zero_grad()
        #bakcpropagation
        gen_loss.backward()
        #update weight generator
        opt_gen.step()

        if idx % 200 == 0:
            plot_examples(config.VAL_PATH, gen)
            print(f'discrimantor loss:{loss_disc}')
            print(f'Generative loss:{gen_loss}')

def main():
    #kita load datasetnya dengan MyImageFolder modul
    dataset = MyImageFolder(root_dir=config.TRAIN_PATH)
    #kita load DataLoader dengan input dataset yang sudah kita definisikan
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    # kita mendeklarasikan model generator kita
    gen = Generator(in_channels=3, ratio=config.RATIO).to(config.DEVICE)
    # kita mendeklarasikan model discriminator kita
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    # kita mendeklarasikan optimizer generator yang akan kita gunakan
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    #kita definisikan loss loss kita
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()
    #jika config.LOAD_MODEL kita true maka dia akan load checkpoint
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
           config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )
    #pada setiap epoch akan dilakukan training menggunakan function train_fn kita yang sudah di deklarasikan
    for epoch in range(config.START_EPOCHS-1,config.NUM_EPOCHS):
        print(f'======================EPOCH: {epoch+1}=====================')
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()