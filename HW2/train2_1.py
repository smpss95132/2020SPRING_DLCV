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

# hyper parameter

class Parameter:
    
    # basic parameter
    NUMBER_OF_CLASSES = 50
    num_workers = 8
    
    # basic setting
    batchsize = 32
    n_epochs = 150
    lr = 0.0003
    
    # wether to print out information
    verbose = True          
    verbose_step = 1  
    
    # save model weight
    # path to save trained model
    folder = '../model_bins/vgg16_cls_224'  
    
    # data root
    TRAIN_ROOT_PATH = '../hw2_data/p1_data/train_50'
    VALID_ROOT_PATH = '../hw2_data/p1_data/val_50'

# Data augmentation

def get_train_transforms():     
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.2,0.2), scale=None, shear=None, resample=False, fillcolor=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_valid_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_fivecrop_transforms():
    return transforms.Compose([
        transforms.Resize((36, 36)),
        transforms.FiveCrop(32)
    ])

class DatasetRetriever(Dataset):

    def __init__(self, root_path, transforms, mode):
        super().__init__()
        
        self.root_path = root_path                      
        self.image_names = sorted(os.listdir(root_path))
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, idx: int):
        
        # image info
        image_name = self.image_names[idx]
        path = os.path.join(self.root_path, image_name)
        label = torch.tensor(int(image_name.split('_')[0]), dtype=torch.int64)
        
        # load image and preprocess image
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
        
        if self.mode=="test": 
            cropped_img = get_fivecrop_transforms()(img)
            img = torch.empty(5, 3, 32, 32)
            for i in range(5):
                img[i] = self.transforms(cropped_img[i])
        else: img = self.transforms(img)
            
        return img, label
    
    def __len__(self) -> int:
        return len(self.image_names)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.correct_count = 0
        self.count = 0
        self.acc = 0

    def update(self, val, batch_size, ncorrect):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.correct_count += ncorrect
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
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, loss: {loss.avg:.5f}, acc: {loss.acc:.5f},time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin') 
            
            # validation
            t = time.time()
            loss = self.validation(valid_loader)
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, loss: {loss.avg:.5f}, acc: {loss.acc:.5f},time: {(time.time() - t):.5f}')
            
            # if the new model is better, save it, otherwise abandant
            if loss.avg < self.best_loss:
                self.best_loss = loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint_loss.bin')
            if loss.acc > self.best_acc:
                self.best_acc = loss.acc
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint_acc.bin')
            
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        
        # record information
        loss_recorder = AverageMeter()   
        t = time.time()
        
        for step, (data, target) in enumerate(val_loader):
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

            loss_recorder.update(loss.detach().item(), batch_size, int(rightness(output, target).cpu()))

            
        loss_recorder.acc = loss_recorder.correct_count / loss_recorder.count
        return loss_recorder

    def train_one_epoch(self, train_loader):
        self.model.train()
        
        loss_recorder = AverageMeter()
        t = time.time()
        
        for step, (data, target) in enumerate(train_loader):
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
            loss = self.criterion(output, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_recorder.update(loss.detach().item(), batch_size, int(rightness(output, target).cpu()))

        loss_recorder.acc = loss_recorder.correct_count / loss_recorder.count
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
    train_dataset = DatasetRetriever(args.TRAIN_ROOT_PATH, get_train_transforms(), mode='train')
    valid_dataset = DatasetRetriever(args.VALID_ROOT_PATH, get_valid_transforms(), mode='valid')
    
    # construct dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                                     drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers,
                                     drop_last=False)

    fitter = Fitter(model=net, device=device, config=args)
    fitter.fit(train_loader, valid_loader)

def main():
    model = models.vgg16_bn(pretrained=True)
    model.classifier = nn.Linear(in_features=25088, out_features=Parameter.NUMBER_OF_CLASSES, bias=True)
    run_training(model, Parameter)

if __name__=="__main__":
    main()