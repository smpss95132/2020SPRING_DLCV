import numpy as np
import glob 
from torch.utils.data import Dataset, DataLoader
import random
import os
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import os
from datetime import datetime
import time
import torchvision.models as models
import matplotlib.pyplot as plt

# hyper parameter

class args:
    
    # hyper-parameter
    # batchsize = 128
    batchsize = 64
    n_epochs = 200
    lr = 0.0002
    
    # model structure
    image_size = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    beta = 0.5
    
    # dataloader
    num_workers = 8
    use_GPU = True
    
    # wether to print out information
    verbose = True          
    verbose_step = 1  
                   
    # save model weight
    # path to save trained model
    folder = '../weights/GAN_re'  
    
    # data root
    DATA_ROOT_PATH = '../hw3_data/face/train'
    TEST_ROOT_PATH = '../hw3_data/face/test'

def get_train_transforms():     
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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

class DatasetRetriever(Dataset):

    def __init__(self, root_path, img_names, transforms):
        super().__init__()
        
        self.root_path = root_path                      
        self.img_names = img_names
        self.transforms = transforms

    def __getitem__(self, idx: int):
        # load image
        with open(os.path.join(self.root_path, self.img_names[idx]), 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
        
        return self.transforms(img)
    
    def __len__(self) -> int:
        return len(self.img_names)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        
        self.G_sum = 0
        self.G_avg = 0
        self.D_sum = 0
        self.D_avg = 0
        self.D_correct = 0
        self.G_correct = 0
        self.GD_correct = 0
        self.D_acc = 0
        self.G_acc = 0
        self.GD_acc = 0
        

    def update(self, D_loss, G_loss, batch_size, D_correct, G_correct, GD_correct):
        self.G_sum += G_loss * batch_size
        self.D_sum += D_loss * batch_size
        self.D_correct += D_correct * batch_size
        self.G_correct += G_correct * batch_size
        self.GD_correct += GD_correct * batch_size
        
        self.count += batch_size
        
        # loss
        if self.count: self.G_avg = self.G_sum / self.count
        else: self.G_avg=0
            
        if self.count: self.D_avg = self.D_sum / self.count
        else: self.D_avg=0
        
        # accuracy
        if self.count: self.D_acc = self.D_correct / self.count
        else: self.D_acc = 0
            
        if self.count: self.G_acc = self.G_correct / self.count
        else: self.G_acc = 0
            
        if self.count: self.GD_acc = self.GD_correct / self.count
        else: self.GD_acc = 0

class Fitter:
    def __init__(self, G, D, device, config):
        # assign parameter
        self.G = G
        self.D = D
        
        self.device = device
        self.config = config
        
        self.epoch = 0
        
        # the folder saving the trained model, if the folder is not exist, create it
        self.base_dir = config.folder
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_loss_G = 10**5   
        
        self.criterion = nn.BCELoss()
        self.real_label, self.fake_label = 1., 0.
        
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.beta, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.beta, 0.999))
        
        self.log(f'Fitter prepared. Device is {self.device}') 

    def fit(self, train_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer_G.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
            
            # train
            t = time.time()
            loss = self.train_one_epoch(train_loader)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, G_loss: {loss.G_avg:.5f}, ' +\
                     f'D_loss: {loss.D_avg:.5f}, D_acc: {loss.D_acc:.5f}, G_acc: {loss.G_acc:.5f}, ' +\
                     f'GD_acc: {loss.GD_acc:.5f}, time: {(time.time() - t):.5f}')
            
            
            # if the new model is better, save it, otherwise abandant
            if loss.G_avg < self.best_loss_G:
                self.best_loss_G = loss.G_avg
                self.G.eval()
                self.save(f'{self.base_dir}/best-checkpoint_G.bin', 'G')
                
                self.best_loss_D = loss.D_avg
                self.D.eval()
                self.save(f'{self.base_dir}/best-checkpoint_D.bin', 'D')
            
            # save the last check point 
            self.save(f'{self.base_dir}/last-checkpoint_G.bin', 'G') 
            self.save(f'{self.base_dir}/last-checkpoint_D.bin', 'D') 
                
            self.epoch += 1

    def train_one_epoch(self, train_loader):
        
        loss_recorder = AverageMeter()
        t = time.time()
        
        self.G.train()
        self.D.train()
        for step, (data) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'D_loss: {loss_recorder.D_avg:.5f}, ' + \
                        f'G_loss: {loss_recorder.G_avg:.5f},' + \
                        f'D_acc: {loss_recorder.D_acc:.5f},' +\
                        f'G_acc: {loss_recorder.G_acc:.5f},' + \
                        f'GD_acc: {loss_recorder.GD_acc:.5f},' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
                    
            batch_size = data.shape[0]
            
            """                            discriminetor                                      """
            # ============================= Real image ==========================================
            self.D.zero_grad()
            
            # img and label
            real_img = data.to(self.device)
            batch_size = real_img.shape[0]
            label = torch.full((batch_size,), self.real_label, dtype=torch.float, device=self.device)
            
            # forward
            output = self.D(real_img).view(-1)
            
            # calculate loss and backward propagation
            errD_real = self.criterion(output, label)
            errD_real.backward()
            
            # calculate accuracy
            D_x = output.mean().item()

            ## ============================= Fake image ==========================================
            
            # img and label
            noise = torch.randn(batch_size, args.nz, 1, 1, device=self.device)
            fake = self.G(noise)
            label.fill_(self.fake_label)
            
            # forward
            output = self.D(fake.detach()).view(-1)
            
            # calculate loss and backward propagation
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            
            # calculate accuracy
            D_G_z1 = output.mean().item()
            
            # D weight update
            errD = errD_real + errD_fake
            self.optimizer_D.step()

            """ generator """
            self.G.zero_grad()
            
            # fake label
            label.fill_(self.real_label)  
            
            # forward
            output = self.D(fake).view(-1)
            
            # calculate loss and backward propagation
            errG = self.criterion(output, label)
            errG.backward()
            
            # calculate accuracy
            D_G_z2 = output.mean().item()
            
            # G weight update
            self.optimizer_G.step()

            loss_recorder.update(errD_fake.detach().item(), errG.detach().item(), batch_size, D_x, 1-D_G_z1, 1-D_G_z2)
     
        return loss_recorder
    
    def save(self, path, model_type):
        if model_type=='G':
            self.G.eval()
            torch.save({
                'model_state_dict': self.G.state_dict(),
                'optimizer_state_dict': self.optimizer_G.state_dict(),
                'best_loss': self.best_loss_G,
                'epoch': self.epoch,
            }, path)
        else:
            self.D.eval()
            torch.save({
                'model_state_dict': self.D.state_dict(),
                'optimizer_state_dict': self.optimizer_D.state_dict(),
                'best_loss': self.best_loss_D,
                'epoch': self.epoch,
            }, path)

    def load(self, path_G, path_D):
        checkpoint = torch.load(path_G)
        self.G.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss_G = checkpoint['best_loss']
        self.epoch = checkpoint['epoch'] + 1
        
        checkpoint = torch.load(path)
        self.D.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
    
    # print and write information in log
    def log(self, message):
        if self.config.verbose:print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

def run_training(Gen, Dis, args):
    
    device = torch.device('cuda:0')
    Gen.to(device)
    Dis.to(device)
    
    # construct dataset
    img_names = sorted(os.listdir(os.path.join(args.DATA_ROOT_PATH)))
    train_dataset = DatasetRetriever(args.DATA_ROOT_PATH, img_names, get_train_transforms())
   
    # construct dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                                     drop_last=False)


    fitter = Fitter(G=Gen, D=Dis, device=device, config=args)
    fitter.fit(train_loader)

def main():
    G = Generator(args.nz, args.ngf, args.nc)
    G.apply(weights_init)

    D = Discriminator(args.nc, args.ndf)
    D.apply(weights_init)

    run_training(G, D, args)

if __name__=="__main__":
    main()


