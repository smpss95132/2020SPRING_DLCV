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
    
    # basic parameter
    ec = 128
    dc = 128
    latent_size = 1024
    color = True
    
    # dataloader
    num_workers = 8
    use_GPU = True
    
    # basic setting
    batchsize = 15
    n_epochs = 60
    lr = 0.001
    KL_lambda = 1e-5
    
    # wether to print out information
    verbose = True          
    verbose_step = 1  
                   
    # save model weight
    # path to save trained model
    folder = '../weights/VAE_data_kl'  
    
    # data root
    DATA_ROOT_PATH = '../hw3_data/face/train'
    TEST_ROOT_PATH = '../hw3_data/face/test'

def get_train_transforms():     
    return transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=None, shear=None, resample=False, fillcolor=0),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def get_valid_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

class DatasetRetriever(Dataset):

    def __init__(self, root_path, img_names, transforms, mode):
        super().__init__()
        
        self.root_path = root_path                      
        self.img_names = img_names
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, idx: int):
        # load image
        with open(os.path.join(self.root_path, self.img_names[idx]), 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
        
        return self.transforms(img)
    
    def __len__(self) -> int:
        return len(self.img_names)

class encode_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(encode_block, self).__init__()
        self.module = nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding), 
                        nn.BatchNorm2d(out_channel),
                        nn.LeakyReLU(0.2),
            
#                         nn.Conv2d(out_channel, out_channel, kernel_size, 1, padding), 
#                         nn.BatchNorm2d(out_channel),
#                         nn.LeakyReLU(0.2),
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

#                             nn.UpsamplingNearest2d(scale_factor=2), 
#                             nn.ReplicationPad2d(1),
#                             nn.Conv2d(out_channel, out_channel, 3, 1),
#                             nn.BatchNorm2d(out_channel),
#                             nn.LeakyReLU(0.2),
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        
        self.sum = 0
        self.avg = 0
        self.mse_sum = 0
        self.mse_avg = 0

    def update(self, loss, mse_loss, batch_size):
        self.sum += loss * batch_size
        self.mse_sum += mse_loss * batch_size
        self.count += batch_size
        
        if self.count: self.avg = self.sum / self.count
        else: self.avg=0
            
        if self.count: self.mse_avg = self.mse_sum / self.count
        else: self.mse_avg=0

def loss_function(pre, target, mean, logvar, KL_lambda):
    
    MSE = nn.MSELoss()(pre, target)
    LKL = torch.sum((1 + logvar - mean*mean - torch.exp(logvar)))*(-0.5)

    return MSE + KL_lambda * LKL

class Fitter:
    def __init__(self, model, device, config):
        # assign parameter
        self.model = model
        self.device = device
        self.config = config
        self.KL_lambda = config.KL_lambda
        
        self.epoch = 0
        
        # the folder saving the trained model, if the folder is not exist, create it
        self.base_dir = config.folder
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_loss = 10**5   
        
        self.criterion = loss_function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        
        self.log(f'Fitter prepared. Device is {self.device}') 

    def fit(self, train_loader, valid_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
            
            # train
            t = time.time()
            loss = self.train_one_epoch(train_loader)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, loss: {loss.avg:.5f}, mse_loss: {loss.mse_avg:.5f},time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin') 
            
            # validation
            t = time.time()
            loss = self.validation(valid_loader)
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, loss: {loss.avg:.5f}, mse_loss: {loss.mse_avg:.5f},time: {(time.time() - t):.5f}')
            
            # if the new model is better, save it, otherwise abandant
            if loss.avg < self.best_loss:
                self.best_loss = loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint.bin')
            
            #self.save(f'{self.base_dir}/checkpoint-{e}.bin')
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        
        # record information
        loss_recorder = AverageMeter()   
        t = time.time()
        
        for step, (data) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'loss: {loss_recorder.avg:.5f}, ' + \
                        f'mse_loss: {loss_recorder.mse_avg:.5f},' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                batch_size = data.shape[0]
                data = data.to(self.device)

                output, mean, logvar = self.model(data)
                loss = self.criterion(output, data, mean, logvar, self.KL_lambda)
                mse_loss = nn.MSELoss()(output, data)
            loss_recorder.update(loss.detach().item(), mse_loss.detach().item(), batch_size)


        return loss_recorder

    def train_one_epoch(self, train_loader):
        self.model.train()
        
        loss_recorder = AverageMeter()
        t = time.time()
        
        for step, (data) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'loss: {loss_recorder.avg:.5f}, ' + \
                        f'mse_loss: {loss_recorder.mse_avg:.5f},' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            batch_size = data.shape[0]
            data = data.to(self.device)
            output, mean, logvar = self.model(data)
            loss = self.criterion(output, data, mean, logvar, self.KL_lambda)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            mse_loss = nn.MSELoss()(output, data)
            loss_recorder.update(loss.detach().item(), mse_loss.detach().item(), batch_size)

                
        return loss_recorder
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_summary_loss': self.best_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_summary_loss = checkpoint['best_loss']
        self.epoch = checkpoint['epoch'] + 1
    
    # print and write information in log
    def log(self, message):
        if self.config.verbose:print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

def run_training(net, args):
    
    device = torch.device('cuda:0')
    net.to(device)
    
    # construct dataset
    img_names = sorted(os.listdir(os.path.join(args.DATA_ROOT_PATH)))
    train_names, valid_names = img_names[:-(len(img_names)//8)], img_names[-(len(img_names)//8):]
    train_dataset = DatasetRetriever(args.DATA_ROOT_PATH, train_names, get_train_transforms(), mode='train')
    valid_dataset = DatasetRetriever(args.DATA_ROOT_PATH, valid_names, get_valid_transforms(), mode='valid')
    
    # construct dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                                     drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers,
                                     drop_last=False)

    fitter = Fitter(model=net, device=device, config=args)
    fitter.fit(train_loader, valid_loader)

def main():
    model = VAE(args.ec, args.dc, args.latent_size, args.color, args.use_GPU)
    run_training(model, args)

if __name__=="__main__":
    main()