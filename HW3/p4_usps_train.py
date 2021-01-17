import torch
import torch.nn as nn
import numpy as np
import glob 
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
from datetime import datetime
import time
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Function

# hyper parameter

class Parameter:
    
    # basic parameter
    NUMBER_OF_CLASSES = 10
    num_workers = 8
    
    # basic setting
    batchsize = 512
    n_epochs = 150
    lr = 0.0001
    
    # wether to print out information
    verbose = True          
    verbose_step = 1  
    
    # save model weight
    # path to save trained model
    folder = '../weights/[4]SVHN_USPS'  
    
    # data root
    TRAIN_ROOT_PATH_s = '../hw3_data/digits/svhn/train'
    TRAIN_ROOT_PATH_t = '../hw3_data/digits/usps/train'
    
    TEST_ROOT_PATH_s = '../hw3_data/digits/svhn/test'
    TEST_ROOT_PATH_t = '../hw3_data/digits/usps/test'
    
    ENCODER_PATH = '../weights/[3-1]SVHN/best-checkpoint_acc.bin'

def get_SVHN_transforms():     
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_USPS_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class DatasetRetriever_SVHN(Dataset):
    def __init__(self, root_path, meta, transforms):
        super().__init__()
        
        self.root_path = root_path                      
        self.meta = meta
        self.transforms = transforms

        print("Dataset is created!")
        print(f"root_path: {self.root_path}, transforms: {self.transforms}")
        
    def __getitem__(self, idx: int):
        # image information
        img_name = self.meta.iloc[idx]['image_name']
        label = self.meta.iloc[idx]['label']
        
        # load image
        with open(os.path.join(self.root_path, img_name), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            
        return self.transforms(img), torch.tensor(int(label), dtype=torch.int64)
    
    def __len__(self) -> int:
        return len(self.meta)

class DatasetRetriever_USPS(Dataset):
    def __init__(self, root_path, meta, transforms):
        super().__init__()
        
        self.root_path = root_path                      
        self.meta = meta
        self.transforms = transforms

        print("Dataset is created!")
        print(f"root_path: {self.root_path}, transforms: {self.transforms}")
        
    def __getitem__(self, idx: int):
        # image information
        img_name = self.meta.iloc[idx]['image_name']
        label = self.meta.iloc[idx]['label']
        
        # load image
        with open(os.path.join(self.root_path, img_name), 'rb') as f:
            img = Image.open(f)
            img.load()
        # grayscale to color
        temp = transforms.ToTensor()(img)
        img = torch.cat([temp, temp, temp], axis=0)
        img = transforms.ToPILImage()(img)
            
        return self.transforms(img), torch.tensor(int(label), dtype=torch.int64)
    
    def __len__(self) -> int:
        return len(self.meta)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0

        self.correct_count_d = 0
        self.correct_count_tc = 0
        self.correct_count_td = 0
        
        self.count = 0

        self.acc_d = 0
        self.acc_tc = 0
        self.acc_td = 0

    def update(self, val, batch_size, ncorrect_d, ncorrect_tc, ncorrect_td):
        self.sum += val * batch_size
        self.count += batch_size

        self.correct_count_d += ncorrect_d
        self.correct_count_tc += ncorrect_tc
        self.correct_count_td += ncorrect_td
        
        self.avg = self.sum / self.count

def rightness(predictions, labels):
    
    pred = torch.max(predictions.data, 1)[1]   # the result of prediction
    rights = pred.eq(labels.data.view_as(pred)).sum()  # count number of correct example
   
    return rights

class Fitter:
    def __init__(self, model, device, config):
        # assign parameter
        self.model = model
        self.device = device
        self.config = config
        
        self.epoch = 0
        
        # the folder saving the trained model, if the folder is not exist, create it
        self.base_dir = config.folder
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_loss = 10**5   
        self.best_acc = 0.0
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_t = torch.optim.Adam(self.model.TEncoder.parameters(), lr=config.lr, betas=(0.5, 0.9))
        self.optimizer_d = torch.optim.Adam(self.model.dsc.parameters(), lr=config.lr, betas=(0.5, 0.9))
        
        self.log(f'Fitter prepared. Device is {self.device}') 

    def fit(self, strain_loader, ttrain_loader, tvalid_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer_t.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
            
            # train
            t = time.time()
            loss = self.train_one_epoch(strain_loader, ttrain_loader)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, loss: {loss.avg:.5f},' +\
                     f'acc_d: {loss.acc_d:.5f}, acc_tc: {loss.acc_tc:.5f},  acc_td: {loss.acc_td:.5f}' +\
                     f'time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin') 
            
            # validation
            t = time.time()
            loss = self.validation(tvalid_loader)
            self.log(f'[RESULT]: Valid. Epoch: {self.epoch}, loss: {loss.avg:.5f},' +\
                     f'acc_d: {loss.acc_d:.5f}, acc_tc: {loss.acc_tc:.5f},  acc_td: {loss.acc_td:.5f}' +\
                     f'time: {(time.time() - t):.5f}')
            
            # if the new model is better, save it, otherwise abandant
            if loss.avg < self.best_loss:
                self.best_loss = loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint_loss.bin')
                
            if loss.acc_tc > self.best_acc:
                self.best_acc = loss.acc_tc
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint_acc.bin')
                
            self.model.eval()
            self.save(f'{self.base_dir}/last-checkpoint.bin')
            
            self.epoch += 1

    def validation(self, tvalid_loader):
        self.model.eval()
        
        # record information
        loss_recorder = AverageMeter()   
        t = time.time()
        
        for step, (data, target) in enumerate(tvalid_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(tvalid_loader)}, ' + \
                        f'loss: {loss_recorder.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
 
            with torch.no_grad():
                batch_size = data.shape[0]
                data, target = data.to(self.device), target.to(self.device)

                coutput = self.model(data, "T")
                loss = self.criterion(coutput, target)

            loss_recorder.update(loss.detach().item(), batch_size, 0, int(rightness(coutput, target).cpu()), 0)

            
        loss_recorder.acc_d = loss_recorder.correct_count_d / loss_recorder.count / 2
        loss_recorder.acc_tc = loss_recorder.correct_count_tc / loss_recorder.count
        loss_recorder.acc_td = loss_recorder.correct_count_td / loss_recorder.count
        
        return loss_recorder

    def train_one_epoch(self, strain_loader, ttrain_loader):
        self.model.train()
        
        loss_recorder = AverageMeter()
        t = time.time()
        
        sloader_len = min(len(strain_loader), len(ttrain_loader))
        i = 0
        
        iter_strain_loader, iter_ttrain_loader = iter(strain_loader), iter(ttrain_loader)
        
        while i < sloader_len:

            print(
                f'Train Step {i}/{len(strain_loader)}, ' + \
                f'loss: {loss_recorder.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}', end='\r'
            )
            
          
            # load data
            simg, _ = iter_strain_loader.next()
            timg, _ = iter_ttrain_loader.next()

            
            batch_size = simg.shape[0]
            sdlabel, tdlabel = torch.ones((batch_size), dtype=torch.int64), torch.zeros((batch_size), dtype=torch.int64)
            simg, sdlabel, timg, tdlabel = simg.cuda(), sdlabel.cuda(), timg.cuda(), tdlabel.cuda()
            
            self.optimizer_d.zero_grad()
            
            # =============================== train discriminator =================================
            # extract feature then concate them
            s_feature, t_feature = self.model.SEncoder(simg), self.model.TEncoder(timg)
            fuse_feature = torch.cat((s_feature, t_feature), dim=0)
            # concate label
            fuse_label = torch.cat((sdlabel, tdlabel), dim=0)
            
            # prediction of discriminator
            d_pre = self.model.dsc(fuse_feature)
            
            # compute loss and backward propagation
            d_loss = self.criterion(d_pre, fuse_label)
            d_loss.backward()
            self.optimizer_d.step()
            
            # =============================== train TEncoder========================================
            self.optimizer_d.zero_grad()
            self.optimizer_t.zero_grad()
            
            # prediction and fake label
            td_pre = self.model.dsc(self.model.TEncoder(timg))
            fake_tdlabel = torch.ones((batch_size), dtype=torch.int64).cuda()
            
            # compute loss and backward propagation
            loss_td = self.criterion(td_pre, fake_tdlabel)
            loss_td.backward()
            self.optimizer_t.step()
            
            
            loss_recorder.update(loss_td.detach().item(), batch_size, int(rightness(d_pre, fuse_label).cpu()),0, 
                    int(rightness(td_pre, fake_tdlabel).cpu()))
            i+=1

        loss_recorder.acc_d = loss_recorder.correct_count_d / loss_recorder.count / 2
        loss_recorder.acc_tc = loss_recorder.correct_count_tc / loss_recorder.count
        loss_recorder.acc_td = loss_recorder.correct_count_td / loss_recorder.count
        
        return loss_recorder
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer_t.state_dict(),
            'best_summary_loss': self.best_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_t.load_state_dict(checkpoint['optimizer_state_dict'])
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
    sall_meta = pd.read_csv(args.TRAIN_ROOT_PATH_s+'.csv')
    tall_meta = pd.read_csv(args.TRAIN_ROOT_PATH_t+'.csv')
    
    strain_meta = sall_meta
    ttrain_meta = tall_meta[:-len(tall_meta)//9]
    tvalid_meta = tall_meta[-len(tall_meta)//9:].reset_index()
    
    strain_dataset = DatasetRetriever_SVHN(args.TRAIN_ROOT_PATH_s, strain_meta, get_SVHN_transforms())
    ttrain_dataset = DatasetRetriever_USPS(args.TRAIN_ROOT_PATH_t, ttrain_meta, get_USPS_transforms())
    tvalid_dataset = DatasetRetriever_USPS(args.TRAIN_ROOT_PATH_t, tvalid_meta, get_USPS_transforms())
    
    # construct dataloader
    strain_loader = DataLoader(strain_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                                     drop_last=True)
    ttrain_loader = DataLoader(ttrain_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                                     drop_last=True)
    tvalid_loader = DataLoader(tvalid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers,
                                     drop_last=False)

    fitter = Fitter(model=net, device=device, config=args)
    fitter.fit(strain_loader, ttrain_loader, tvalid_loader)

class Encoder(nn.Module):
    def __init__(self, backbone):
        super(Encoder, self).__init__()
        
        self.layer = nn.Sequential(
                    backbone.conv1,
                    backbone.bn1,
                    backbone.relu,
                    backbone.maxpool,
                    backbone.layer1,
                    backbone.layer2,
                    backbone.layer3,
                    backbone.layer4,
                    backbone.avgpool,
                )
        
        self.fc = nn.Linear(512, 512)
    
    def forward(self, x):
        x = self.layer(x)
        #x = self.fc(x.view(x.shape[0], -1))
        
        return x.view(x.shape[0], -1)

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        
        self.layer = nn.Sequential(
                    nn.Linear(input_dim, output_dim, bias=True),
                )
        
    def forward(self, x):
        return self.layer(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        
        self.layer = nn.Sequential(
                    nn.Linear(input_dim, input_dim//4),
                    nn.ReLU(),
                    nn.Linear(input_dim//4, input_dim//4),
                    nn.ReLU(),
                    nn.Linear(input_dim//4, output_dim),
                    nn.LogSoftmax()
                )
        
    def forward(self, x):
        return self.layer(x)

class SVHN2USPS(nn.Module):
    def __init__(self, backbone):
        super(SVHN2USPS, self).__init__()
        
        self.SEncoder = Encoder(models.resnet18(pretrained=False))
        self.TEncoder = Encoder(models.resnet18(pretrained=False))
        
        self.cls = Classifier(512, 10)
        
        self.dsc = Discriminator(512, 2)
    
    def forward(self, x, domain):
        if domain=="S":
            x = self.SEncoder(x)
            return self.cls(x)
        else:
            x = self.TEncoder(x)
            return self.cls(x)

def main():
    # define model structure
    model = SVHN2USPS(models.resnet18(pretrained=False))    
    temp = models.resnet18(pretrained=False)
    temp.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    temp.load_state_dict(torch.load(Parameter.ENCODER_PATH)['model_state_dict'])

    temp_feature = nn.Sequential(
                            temp.conv1,
                            temp.bn1,
                            temp.relu,
                            temp.maxpool,
                            temp.layer1,
                            temp.layer2,
                            temp.layer3,
                            temp.layer4,
                            temp.avgpool)

    temp_cls = nn.Sequential(temp.fc)
    # weight copy
    model.SEncoder.layer.load_state_dict(temp_feature.state_dict())
    model.TEncoder.layer.load_state_dict(temp_feature.state_dict())
    model.cls.layer.load_state_dict(temp_cls.state_dict())
    run_training(model, Parameter)

if __name__=="__main__":
    main()