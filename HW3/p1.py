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
    ec = 128
    dc = 128
    latent_size = 1024
    color = True
    use_GPU = True

class encode_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(encode_block, self).__init__()
        self.module = nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding), 
                        nn.BatchNorm2d(out_channel),
                        nn.LeakyReLU(0.2),
                        )
        
    def forward(self, x):
        return self.module(x)

class decode_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, last_layer):
        super(decode_block, self).__init__()
        if last_layer:
            self.module = nn.Sequential(
                            nn.UpsamplingNearest2d(scale_factor=2), 
                            nn.ReplicationPad2d(1),
                            nn.Conv2d(in_channel, out_channel, kernel_size),
                            nn.Tanh()
                            )
        else:
            self.module = nn.Sequential(
                            nn.UpsamplingNearest2d(scale_factor=2), 
                            nn.ReplicationPad2d(1),
                            nn.Conv2d(in_channel, out_channel, kernel_size),
                            nn.BatchNorm2d(out_channel),
                            nn.LeakyReLU(0.2),
                            )
            
    def forward(self, x):
        return self.module(x)

class VAE(nn.Module):
    def __init__(self, ec=128, dc=128, latent_size=512, color=True, use_GPU=True):
        super(VAE, self).__init__()
        
        # basic parameter
        self.latent_size = latent_size
        self.ec = ec
        self.dc = dc
        self.in_channel = 3 if color else 1
        self.use_GPU = use_GPU
        
        # model definition
        # encoder
        self.encoder = nn.Sequential(
                        # 64
                        encode_block(self.in_channel, ec, 3, 2, 1),
            
                        # 32
                        encode_block(ec, ec*2, 3, 2, 1),
            
                        # 16
                        encode_block(ec*2, ec*4, 3, 2, 1),
            
                        # 8
                        encode_block(ec*4, ec*8, 3, 2, 1),
            
                        # 4
                        encode_block(ec*8, ec*16, 3, 2, 1)
            
                        # 2 
                        )
        self.mean_fc = nn.Linear(ec*16*2*2, self.latent_size)
        self.logvar_fc = nn.Linear(ec*16*2*2, self.latent_size)
        
        # decoder
        self.decode_fc = nn.Linear(self.latent_size, dc*16*2*2)
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
                        # 2
                        decode_block(dc*16, dc*8, 3, False),
            
                        #4
                        decode_block(dc*8, dc*4, 3, False),
                        
                        # 8
                        decode_block(dc*4, dc*2, 3, False),
            
                        #16
                        decode_block(dc*2, dc, 3, False),
                        
                        # 32
                        decode_block(dc, self.in_channel, 3, True)
            
                        #64
                        )
        
    def sampling(self, mean, logvar):
        std = torch.exp((logvar*0.5))
        nor_dis = torch.normal(0, 1, size=mean.shape).cuda() if self.use_GPU else torch.normal(0, 1, size=mean.shape)
        
        return nor_dis*std + mean
        
        
    def forward(self, x):
        # x: B, 3, 64, 64
        x = self.encoder(x).view(x.shape[0], -1)
        # x: B, 2048*2*2
        mean, logvar = self.mean_fc(x), self.logvar_fc(x)
        # mean=logvar: B, 1024
        latent = self.sampling(mean, logvar)
        # latent: B, 1024
        x = self.relu(self.decode_fc(latent)).view(x.shape[0] ,self.dc*16 ,2 ,2)
        x = self.decoder(x)
        
        return x, mean, logvar

def main():
    # define and load model
    model = VAE(args.ec, args.dc, args.latent_size, args.color, args.use_GPU)
    checkpoint = torch.load('VAE.bin?dl=1')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    decoder = model.decoder.cuda()
    fc = model.decode_fc.cuda()
    relu = model.relu.cuda()
    
    # reconstruct image
    invTrans = transforms.Compose([ 
                                transforms.Normalize(mean = [ 0., 0., 0. ],std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ], std = [ 1., 1., 1. ]),
                               ])
    
    # generate random image
    img_list = []
    torch.manual_seed(4)
    for i in range(32):
        with torch.no_grad():
            latent = torch.normal(0, 1, size=(1, 1024)).cuda()

            pre = decoder(relu(fc(latent)).view(1 ,128*16 ,2 ,2))
            img_list.append(invTrans(pre[0]).cpu())


    img_list = torch.stack(img_list)
    grid_img = make_grid(img_list, 8, 2)
    save_image(grid_img, sys.argv[1])
            
if __name__=='__main__':
    main()