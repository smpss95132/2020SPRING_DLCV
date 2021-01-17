import numpy as np
import random
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
import os
import sys
from PIL import Image

class args:
    image_size = 64
    nc = 3
    nz = 100
    ngf = 64

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            # 1*1
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 4*4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 8*8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16*16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 32*32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 64*64
        )

    def forward(self, x):
        return self.layer(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, ):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            # 64*64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32*32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16*16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8*8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4*4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

def main():
    # model define and load model
    G = Generator(args.nz, args.ngf, args.nc)
    checkpointG = torch.load('GAN_G.bin?dl=1')
    G.load_state_dict(checkpointG['model_state_dict'])
    G.cuda()
    G.eval()

    # reconstruct image
    invTrans = transforms.Compose([ 
                                transforms.Normalize(mean = [ 0., 0., 0. ],std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ], std = [ 1., 1., 1. ]),
                               ])

    # generate image
    img_list = []
    torch.manual_seed(18)
    for i in range(32):
        noise = torch.randn(1, args.nz, 1, 1, device=torch.device('cuda:0'))
        fake = G(noise).detach().cpu()
        img_list.append(invTrans(fake.squeeze(0)))
    
    img_list = torch.stack(img_list)
    grid_img = make_grid(img_list, 8, 2)
    save_image(grid_img, sys.argv[1])

if __name__=="__main__":
    main()