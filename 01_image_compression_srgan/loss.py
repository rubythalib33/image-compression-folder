import torch.nn as nn
from torchvision.models import vgg19
import config

# phi_5,4 5th conv layer before maxpooling but after activation
# setiap modul network (layer, blok network, loss function) pada pytorch akan melakukan inheritance ke nn.Module
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # melakukan pendefinisian vgg19 dengan menggunakan pretrained imagenet dan menggunakan
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        # kita mendefinisikan MSELoss untuk melakukan mengkalkulasi perbedaan antara input feature dan target feature
        self.loss = nn.MSELoss()

        #ini adalah sebuah operasi untuk melakukan freezing parameters terhadap vgg jadi ketika training parameternya tidak terupdate
        for param in self.vgg.parameters():
            param.requires_grad = False

    # __call__ -> loss_fn = VGGLoss() -> loss_fn(input, target)
    def forward(self, input, target):
        #memasukan input kedalam vgg untuk menghasilkan input feature
        vgg_input_features = self.vgg(input)
        #memasukan target kedalam vgg untuk menghasilkan target feature
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)