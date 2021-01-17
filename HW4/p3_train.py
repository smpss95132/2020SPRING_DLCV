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

class Parametric(nn.Module):
    def __init__(self, input_dim=1600):
        super().__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_dim, input_dim//4),
                    nn.ReLU(),
                    nn.Linear(input_dim//4, input_dim//16),
                    nn.ReLU(),
                    nn.Linear(input_dim//16, 1),
                    nn.ReLU())
    
    def forward(self, x):
        return self.fc(x)

class PrototypicalHallucination(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, use_GPU=True):
        super().__init__()
        self.use_GPU = use_GPU
        self.feature_extractor = Convnet(x_dim, hid_dim, z_dim)
        
        self.hallucinator = nn.Sequential(
                                nn.Linear(1600, 1600),
                                nn.BatchNorm1d(1600),
                                nn.ReLU(),
                                nn.Linear(1600, 1600),
                                nn.BatchNorm1d(1600),
                                nn.ReLU(),
                                nn.Linear(1600, 1600),
                                nn.BatchNorm1d(1600),
                                nn.ReLU(),
                            )
        self.distance_matric = Parametric()
        
        self.hal_weight = nn.Sequential(
                                nn.Linear(1600, 1),
                                nn.ReLU(),
                            )

        
        
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.MLP(x)
    
    def extract_feature(self, x):
        return self.feature_extractor(x)
    
    def hallucinate(self, x, m_augment):
        b, feature_dim = x.shape
        
        x_noises = []
        for _ in range(m_augment):
            noise = torch.normal(0, 1, size=(b, feature_dim)).cuda() if self.use_GPU else torch.normal(0, 1, size=(b, feature_dim))
            x_noises.append(x + noise)
                
        y = torch.cat(x_noises, dim=0)
        
        return self.hallucinator(y)
    
    def pass_MLP(self, x):
        return self.MLP(x)
    
    def distance_evaluation(self, prototype, query_feature):
        num_way, feature_len = prototype.shape
        num_query, _ = query_feature.shape
        
        distances = []
        for q in range(num_query):
            query = query_feature[q]
            diff = query - prototype
            distances.append(self.distance_matric(diff))
       
        return torch.stack(distances).squeeze(2)
    def weight_evaluation(self, prototype):
        return self.hal_weight(prototype)

# hyper parameter

class args:
    
    # few shot learning hyperparameter
    num_shot = 1
    num_train_ways, num_train_query = 32, 5
    num_valid_ways, num_valid_query = 5, 15
    num_hallucinate = 1
    
    # dataloader
    num_workers = 8
    use_GPU = True
    
    # basic setting
    n_epochs = 200
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

# training mode : each sample occur at most one times
# valid mode : each sample may occur several times which depend on the number of episode

class GeneratorSampler(Sampler):
    def __init__(self, dataset_df, mode, args, num_episode=600):
        assert mode in ['full data', 'random', 'specified'], "invalid mode"
        '''
        Mode explain:
            'full data': Each sample used at most one time(some may not be used, if drop_lost is adapt)
                              Note: num_episode is ignore!
            'random': sampling samples randomly until num episode is furfill
            'specified': the sampleing sequence is defined by the dataframe
                              Note: the only used parameter is dataset_df
        '''
        
        self.dataset_df = dataset_df
        self.mode = mode
        self.args = args
        self.num_episode = num_episode
        
        if mode in ['full data', 'random']: self.sampled_sequence = self.sampling()
        else: self.sampled_sequence = dataset_df.values.flatten().tolist()
        
        print(f"Generator({mode}) initilized!!")
        
        
    def __iter__(self):
        # resampling for ['full data loop', 'random'] mode
        if self.mode in ['full data', 'random']: self.sampled_sequence = self.sampling()
    
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)
    
    def sampling(self):
        assert self.mode in ['full data', 'random'], "sampling function can only be used in 'full data', 'random mode'"
        random.seed(datetime.now())
        args = self.args
        if self.mode=='full data':
            num_ways, num_query = args.num_train_ways, args.num_train_query
            num_episode = int(self.dataset_df.shape[0]//(num_ways*(args.num_shot+num_query)))
        else:
            num_ways, num_query = args.num_valid_ways, args.num_valid_query
            num_episode = self.num_episode
        
        # get classes in  train set, and count the number of episode
        classes = set()
        for label in self.dataset_df['label']:
            classes.add(label)

        num_class = len(classes)

        # assign each image to it's belong set
        image_set = {}
        for label in classes:
            temp = list(self.dataset_df[self.dataset_df['label']==label]['id'])
            image_set[label] = temp
        for key in image_set.keys():
            shuffle(image_set[key])

        # the mapping of [id -> image names] and [image names -> id]
        id2name = dict(zip(list(range(num_class)), list(classes)))

        # classes permutation for episodes
        permute_times = math.ceil(num_episode * num_ways / num_class)
        permuter_classes = []
        for _ in range(permute_times):
            temp = list(range(num_class))
            shuffle(temp)
            permuter_classes.extend(temp)
        
        # construct sampling list
        num_episode_sample = num_ways*(args.num_shot+num_query)
        sampled_sequence = [-1]*num_episode*num_episode_sample
        
        if self.mode=='full data': 
            for e in range(num_episode):
                choose_classes = permuter_classes[e*num_ways:(e+1)*num_ways]
                
                # support samples and query samples
                for s, (choose_class) in enumerate(choose_classes):
                    choose_class = id2name[choose_class]
                    sampled_sequence[e*num_episode_sample+s:(e+1)*num_episode_sample:num_ways] = image_set[choose_class][:args.num_shot+num_query]
                    image_set[choose_class] = image_set[choose_class][args.num_shot+num_query:]
                    
        else:
            for e in range(num_episode):
                choose_classes = permuter_classes[e*num_ways:(e+1)*num_ways]
                
                # support samples and query samples
                for s, (choose_class) in enumerate(choose_classes):
                    choose_class = id2name[choose_class]
                    choosen_sample = random.sample(range(0, len(image_set[choose_class])-1), args.num_shot+num_query)
                    sampled_sequence[e*num_episode_sample+s:(e+1)*num_episode_sample:num_ways] = [image_set[choose_class][c] for c in choosen_sample]
            
        return sampled_sequence

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
                prototype_feature = model.extract_feature(support)  # (num_shot*num_ways) * 1600
                prototype_ori = prototype_feature.unsqueeze(1)          # (num_shot*num_ways) * 1 * 1600
                prototype_hal = model.hallucinate(prototype_feature, self.config.num_hallucinate)
                prototype_hal = prototype_hal.view(prototype_feature.shape[0], self.config.num_hallucinate, -1) # (num_shot*num_ways) * num_hallucinate * 1600
#                 print(prototype_ori.shape)
#                 print(prototype_hal.shape)
                prototype = torch.cat([prototype_ori, prototype_hal], dim=1) # (num_shot*num_ways) * (num_hallucinate+1) * 1600
                prototype_weight = model.weight_evaluation(prototype.view(-1,prototype.shape[2]))
                weighted_prototype = (prototype.view(-1,prototype.shape[2]) * prototype_weight).view(-1,self.config.num_hallucinate+1,prototype.shape[2])        
                prototype = prototype.mean(dim=1) # (num_shot*num_ways) * 1600
            
                query_feature = model.extract_feature(query)

                # eclidean distance for each query
                e_distance = model.distance_evaluation(prototype, query_feature)

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
            label = list(range(self.config.num_train_ways)) * self.config.num_train_query
            label = torch.tensor(label).type(torch.LongTensor).to(self.device)
            
            # seperate support and query
            support = data[:self.config.num_train_ways*self.config.num_shot]
            query = data[self.config.num_train_ways*self.config.num_shot:]
            
            # evaluate prototype and calculate the mean 
            prototype_feature = model.extract_feature(support)  # (num_shot*num_ways) * 1600
            #prototype_ori = model.pass_MLP(prototype_feature)   # (num_shot*num_ways) * 1600
            prototype_ori = prototype_feature.unsqueeze(1)          # (num_shot*num_ways) * 1 * 1600
            #prototype_hal = model.pass_MLP(model.hallucinate(prototype_feature, self.config.num_hallucinate)) # (num_shot*num_ways*num_hallucinate) * 1600
            prototype_hal = model.hallucinate(prototype_feature, self.config.num_hallucinate)
            prototype_hal = prototype_hal.view(prototype_feature.shape[0], self.config.num_hallucinate, -1) # (num_shot*num_ways) * num_hallucinate * 1600
            #print(prototype_ori.shape)
            #print(prototype_hal.shape)
            prototype = torch.cat([prototype_ori, prototype_hal], dim=1) # (num_shot*num_ways) * (num_hallucinate+1) * 1600
            
            
            prototype_weight = model.weight_evaluation(prototype.view(-1,prototype.shape[2]))
            weighted_prototype = (prototype.view(-1,prototype.shape[2]) * prototype_weight).view(-1,self.config.num_hallucinate+1,prototype.shape[2])
            prototype = prototype.mean(dim=1) # (num_shot*num_ways) * 1600
            query_feature = model.extract_feature(query)
            
            # eclidean distance for each query
            e_distance = model.distance_evaluation(prototype, query_feature)
  
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
        num_workers=8, pin_memory=False,sampler=GeneratorSampler(train_df,'full data',args))
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.num_valid_ways * (args.num_valid_query + args.num_shot),
        num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(val_testcase_df.set_index("episode_id"),'specified', args))

    fitter = Fitter(model=net, device=device, config=args)
    fitter.fit(train_loader, valid_loader)

if __name__=='__main__':
    model = PrototypicalHallucination()
    run_training(model, args)