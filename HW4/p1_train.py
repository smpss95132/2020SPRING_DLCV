import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
from random import shuffle
import numpy as np
import pandas as pd
from datetime import datetime
import time

from PIL import Image
from datetime import datetime
import math

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# hyper parameter

class args:
    
    # few shot learning hyperparameter
    num_shot = 1
    num_train_ways, num_train_query = 32, 5
    num_valid_ways, num_valid_query = 5, 15
    
    # dataloader
    num_workers = 8
    use_GPU = True
    
    # basic setting
    n_epochs = 100
    lr = 0.001

    
    # wether to print out information
    verbose = True          
    verbose_step = 1  
                   
    # save model weight
    # path to save trained model
    folder = 'weights/temp'  
    
    # data root
    train_dir = "hw4_data/train"
    train_csv = "hw4_data/train.csv"
    valid_dir = "hw4_data/val"
    valid_csv = "hw4_data/val.csv"
    val_testcase_csv = "hw4_data/val_testcase.csv"

filenameToPILImage = lambda x: Image.open(x)

def get_train_transforms():     
    return transforms.Compose([
        filenameToPILImage,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_valid_transforms():     
    return transforms.Compose([
        filenameToPILImage,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_test_transforms():     
    return transforms.Compose([
        filenameToPILImage,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir, mode):
        print("dataset ini")
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        
        if mode=='train': self.transform = get_train_transforms()
        elif mode=='valid': self.transform = get_valid_transforms()
        else: self.transform = get_test_transforms()
        

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label, index

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, dataset_df, mode, args):
        print("Generator ini!!")
        self.dataset_df = dataset_df
        self.mode = mode
        self.args = args
        
        # training mode
        if mode=='train': self.sampling()
        # validation mode or test mode
        else:
            self.sampled_sequence = dataset_df.values.flatten().tolist()
        
    def __iter__(self):
        if self.mode=='train': self.sampling()
            
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)
    
    def sampling(self):
        random.seed(datetime.now())
        dataset_df = self.dataset_df
        mode = self.mode
        args = self.args
        
        num_train_episode = int(dataset_df.shape[0]//(args.num_train_ways*(args.num_shot+args.num_train_query)))
            
        # get classes in  train set, and count the number of episode
        train_classes = set()
        for label in dataset_df['label']:
            train_classes.add(label)

        num_train_class = len(train_classes)
        num_train_episode = int(dataset_df.shape[0]//(args.num_train_ways*(args.num_shot+args.num_train_query)))

        # assign each image to it's belong set
        train_set = {}
        for label in train_classes:
            temp = list(dataset_df[dataset_df['label']==label]['id'])
            train_set[label] = temp
        for key in train_set.keys():
            shuffle(train_set[key])

        # the mapping of [id -> image names] and [image names -> id]
        id2name = dict(zip(list(range(64)), list(train_classes)))
        name2id = dict(zip(list(train_classes), list(range(64))))

        # classes permutation for episodes
        permute_times = math.ceil(num_train_episode * args.num_train_ways / num_train_class)
        permuter_classes = []
        for _ in range(permute_times):
            temp = list(range(num_train_class))
            shuffle(temp)
            permuter_classes.extend(temp)

        # construct sampling list
        self.sampled_sequence = []
        for i in range(num_train_episode):
            choose_classes = permuter_classes[i*args.num_train_ways:(i+1)*args.num_train_ways]
            # support samples
            for choose_class in choose_classes:
                choose_class = id2name[choose_class]
                self.sampled_sequence.append(train_set[choose_class][args.num_shot-1])
            # query samples
            for choose_class in choose_classes:
                choose_class = id2name[choose_class]
                self.sampled_sequence.extend(train_set[choose_class][args.num_shot:args.num_shot+args.num_train_query])
                train_set[choose_class] = train_set[choose_class][args.num_shot+args.num_train_query:]

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def rightness(predictions, labels):
    
    pred = torch.max(predictions.data, 1)[1]   # the result of prediction
    rights = pred.eq(labels.data.view_as(pred)).sum()  # count number of correct example
    
    return rights

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.sum = 0
        self.correct_count = 0
        self.count = 0

    def update(self, loss, batch_size, ncorrect):
        self.sum += loss * batch_size
        self.count += batch_size
        self.correct_count += ncorrect

        
    @property
    def avg(self):
        if self.count==0: return 0
        else: return self.sum / self.count
    
    @property
    def acc(self):
        if self.count==0: return 0
        else: return self.correct_count / self.count

def eclidean(p_feature, q_feature):
    num_ways, feature_size = p_feature.shape
    num_query = q_feature.shape[0]
    
    p_feature = p_feature.unsqueeze(0).expand(num_query, num_ways, -1)
    q_feature = q_feature.unsqueeze(1).expand(num_query, num_ways, -1)
    
    return -((p_feature-q_feature)**2).sum(dim=2)

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
        self.best_acc = 0
        
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
            
            #self.save(f'{self.base_dir}/checkpoint-{e}.bin')
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        
        # record information
        loss_recorder = AverageMeter()   
        t = time.time()
        
        for step, (data, target, _) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Valid Step {step}/{len(val_loader)}, ' + \
                        f'loss: {loss_recorder.avg:.5f}, ' + \
                        f'acc: {loss_recorder.acc:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                # put data into GPU
                batch_size = self.config.num_valid_ways * self.config.num_valid_query
                data = data.to(self.device)
                
                # map label to templabel and put it into GPU
                label2tempLabel = {}
                for s in range(self.config.num_valid_ways):
                    label2tempLabel[target[s]] = s
                    
                label = [label2tempLabel[label] for label in target][self.config.num_valid_ways:]
                label = torch.tensor(label).type(torch.LongTensor).to(self.device)

                # seperate support and query
                support = data[:self.config.num_valid_ways*self.config.num_shot]
                query = data[self.config.num_valid_ways*self.config.num_shot:]

                # evaluate prototype and calculate the mean 
                prototype = model(support)
                prototype = prototype.view(self.config.num_shot, self.config.num_valid_ways, -1).mean(dim=0)

                # evaluate query
                query_feature = model(query)
                # eclidean distance for each query
                e_distance = eclidean(prototype, query_feature)

                # loss
                loss = self.criterion(e_distance, label)

            loss_recorder.update(loss.detach().item(), batch_size, int(rightness(e_distance, label).cpu()))

        return loss_recorder

    def train_one_epoch(self, train_loader):
        self.model.train()
        
        loss_recorder = AverageMeter()
        t = time.time()
        
        for step, (data, target, _) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'loss: {loss_recorder.avg:.5f}, ' + \
                        f'acc: {loss_recorder.acc:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            # put data and label into GPU
            batch_size = self.config.num_train_ways * self.config.num_train_query
            data = data.to(self.device)
            label = []
            for l in range(self.config.num_train_ways):
                label.extend([l]*self.config.num_train_query)
            label = torch.tensor(label).type(torch.LongTensor).to(self.device)
            
            # seperate support and query
            support = data[:self.config.num_train_ways*self.config.num_shot]
            query = data[self.config.num_train_ways*self.config.num_shot:]
            
            # evaluate prototype and calculate the mean 
            prototype = model(support)
            prototype = prototype.view(self.config.num_shot, self.config.num_train_ways, -1).mean(dim=0)
            
            # evaluate query
            query_feature = model(query)
            
            # eclidean distance for each query
            e_distance = eclidean(prototype, query_feature)
            
            # loss
            loss = self.criterion(e_distance, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_recorder.update(loss.detach().item(), batch_size, int(rightness(e_distance, label).cpu()))

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
    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.val_testcase_csv)
    val_testcase_df = pd.read_csv(args.val_testcase_csv)
    train_dataset = MiniDataset(args.train_csv, args.train_dir, mode='train')
    valid_dataset = MiniDataset(args.valid_csv, args.valid_dir, mode='valid')
    
    # construct dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.num_train_ways * (args.num_train_query + args.num_shot),
        num_workers=8, pin_memory=False,sampler=GeneratorSampler(train_df,'train',args))
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.num_valid_ways * (args.num_valid_query + args.num_shot),
        num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(val_testcase_df.set_index("episode_id"),'valid', args))

    fitter = Fitter(model=net, device=device, config=args)
    fitter.fit(train_loader, valid_loader)

if __name__=="__main__":
    model = Convnet()
    run_training(model, args)