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
import random

# hyper parameter

class Parameter:
    
    # basic parameter
    NUMBER_OF_CLASSES = 7
    num_workers = 8
    
    # basic setting
    batchsize = 5
    n_epochs = 150
    lr = 0.0003
    
    # wether to print out information
    verbose = True          
    verbose_step = 1  
    
    pixel2class = {0:(0,255,255), 1:(255,255,0), 
                   2:(255,0,255), 3:(0,255,0), 
                   4:(0,0,255), 5:(255,255,255),
                   6:(0,0,0)}
                   
    
    # save model weight
    # path to save trained model
    folder = '../weight2_2/vgg16_bn_base'
    
    # data root
    TRAIN_ROOT_PATH = '../hw2_data/p2_data/train'
    VALID_ROOT_PATH = '../hw2_data/p2_data/validation'

# Data augmentation

def get_train_transforms():     
    return transforms.Compose([
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_valid_transforms():
    return transforms.Compose([
        #transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def flip(x, mode):
    # Note: the shape of x should be (#channel, h, w)
    r = torch.empty(x.shape)
    if mode=='horizontal':
        w = x.shape[2]
        for i in range(w):
            r[:,:,i] = x[:,:,w-1-i]
    elif mode=='vertical':
        h = x.shape[1]
        for i in range(h):
            r[:,i,:] = x[:,h-1-i,:]
    else:
        print("Undefined mode!")
        return 0
    
    return r

class DatasetRetriever(Dataset):

    def __init__(self, root_path, transforms, pixel2class, mode):
        super().__init__()
        
        self.root_path = root_path                      
        self.names = sorted([s[:4] for s in os.listdir(root_path) if s.endswith(".jpg")])
        self.transforms = transforms
        self.mode = mode
        self.pixel2class = pixel2class

    def __getitem__(self, idx: int):
        #resize32 = transforms.Resize((32, 32), interpolation=Image.NEAREST)
        
        # image info
        name = self.names[idx]
        
        # load image and mask
        with open(os.path.join(self.root_path, name+'_sat.jpg'), 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
        
        with open(os.path.join(self.root_path, name+'_mask.png'), 'rb') as f:
            mask3C = Image.open(f)
            mask3C.convert('RGB')
           
        #mask3C = resize32(mask3C)
        mask3C = np.array(mask3C)
        # preprocess image
        img = self.transforms(img)
        
        # preprocess mask
        mask = np.zeros((mask3C.shape[0], mask3C.shape[1]))
        for key in self.pixel2class:
            pixel_value = self.pixel2class[key]
            indices = np.where(np.all(mask3C == pixel_value, axis=-1))
            mask[indices] = key
        mask = torch.from_numpy(mask).type(torch.LongTensor)
        
        if  self.mode=='train':
            # random horizontal/vertical flip
            perform_hor, perform_ver = bool(random.getrandbits(1)), bool(random.getrandbits(1))
            if perform_hor:
                img = flip(img, 'horizontal')
                mask = flip(mask.unsqueeze(0), 'horizontal').squeeze(0)
            if perform_ver:
                img = flip(img, 'vertical')
                mask = flip(mask.unsqueeze(0), 'vertical').squeeze(0)
            
            # random shift
            shift_range = img.shape[1]//10
            hor_shift, ver_shift = random.randint(-shift_range,shift_range), random.randint(-shift_range,shift_range)

            # horizontal shift
            if hor_shift>0: # right shift
                img[:,:,-hor_shift:] = 0
                img = torch.roll(img, hor_shift, 2)
                mask[:,-hor_shift:] = 6
                mask = torch.roll(mask, hor_shift, 1)
            else: # left shift
                img[:,:,:-hor_shift] = 0
                img = torch.roll(img, hor_shift, 2)
                mask[:,:-hor_shift] = 6
                mask = torch.roll(mask, hor_shift, 1)
            # vertical shift
            if ver_shift>0: # down shift
                img[:,-ver_shift:,:] = 0
                img = torch.roll(img, ver_shift, 1)
                mask[-ver_shift:,:] = 6
                mask = torch.roll(mask, ver_shift, 0)
            else: # up shift
                img[:,:-ver_shift,:] = 0
                img = torch.roll(img, ver_shift, 1)
                mask[:-ver_shift,:] = 6
                mask = torch.roll(mask, ver_shift, 0)
            mask = mask.type(torch.LongTensor)

        return img, mask, name
    
    def __len__(self) -> int:
        return len(self.names)

class FCN16s(nn.Module):
    def __init__(self, num_class):
        super(FCN16s, self).__init__()
        
        self.num_class = num_class
        
        # model structure
        self.vgg_backbone4 = models.vgg16_bn(pretrained=True).features[:34]
        self.vgg_backbone5 = models.vgg16_bn(pretrained=True).features[34:]
        
        self.down_sampling = nn.Sequential(   
                                # fc6
                                nn.Conv2d(512, 4096, 1),
                                nn.BatchNorm2d(4096),
                                nn.ReLU(inplace=True),
                                #nn.Dropout2d(),

                                # fc7
                                nn.Conv2d(4096, 4096, 1),
                                nn.BatchNorm2d(4096),
                                nn.ReLU(inplace=True),
                                #nn.Dropout2d(),
            
                                nn.Conv2d(4096, self.num_class, 1)
                                )
        
        self.reduce_channel = nn.Conv2d(512, self.num_class, 1)
        self.up_sampling2 = nn.ConvTranspose2d(self.num_class, self.num_class, 2, stride=2,bias=True)
        self.up_sampling32 = nn.ConvTranspose2d(self.num_class, self.num_class, 16, stride=16,bias=True)
            
    def forward(self, x): 
        # forward path
        feature4 = self.vgg_backbone4(x)
        
        x = self.vgg_backbone5(feature4)
        x = self.down_sampling(x)
        x = self.up_sampling2(x)
        
        # fusion
        feature4 = self.reduce_channel(feature4)
        x = feature4 + x
        
        return self.up_sampling32(x)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.acc = 0

    def update(self, val, batch_size):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        if self.count: self.avg = self.sum / self.count
        else: self.avg=0

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
        
        self.criterion = nn.CrossEntropyLoss()
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
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, loss: {loss.avg:.5f},time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin') 
            
            # validation
            t = time.time()
            loss = self.validation(valid_loader)
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, loss: {loss.avg:.5f},time: {(time.time() - t):.5f}')
            
            # if the new model is better, save it, otherwise abandant
#             if loss.avg < self.best_loss:
#                 self.best_loss = loss.avg
#                 self.model.eval()
#                 self.save(f'{self.base_dir}/best-checkpoint.bin')
            self.save(f'{self.base_dir}/checkpoint-{self.epoch}.bin')
            
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        
        # record information
        loss_recorder = AverageMeter()   
        t = time.time()
        
        for step, (data, target, name) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'loss: {loss_recorder.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                batch_size = data.shape[0]
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

            loss_recorder.update(loss.detach().item(), batch_size)


        return loss_recorder

    def train_one_epoch(self, train_loader):
        self.model.train()
        
        loss_recorder = AverageMeter()
        t = time.time()
        
        for step, (data, target, name) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'loss: {loss_recorder.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            batch_size = data.shape[0]
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            #print(output.shape, target.shape, torch.max(target))
            loss = self.criterion(output, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_recorder.update(loss.detach().item(), batch_size)

                
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
    train_dataset = DatasetRetriever(args.TRAIN_ROOT_PATH, get_train_transforms(), args.pixel2class, mode='train')
    valid_dataset = DatasetRetriever(args.VALID_ROOT_PATH, get_valid_transforms(), args.pixel2class, mode='valid')
    
    # construct dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                                     drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers,
                                     drop_last=False)

    fitter = Fitter(model=net, device=device, config=args)
    fitter.fit(train_loader, valid_loader)

def main():
    model = FCN16s(Parameter.NUMBER_OF_CLASSES)
    run_training(model, Parameter)

if __name__=="__main__":
    main()