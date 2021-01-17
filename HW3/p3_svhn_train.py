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

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Parameter:
    
    # basic parameter
    NUMBER_OF_CLASSES = 10
    num_workers = 8
    
    # basic setting
    batchsize = 128
    n_epochs = 150
    lr = 0.001
    
    # wether to print out information
    verbose = True          
    verbose_step = 1  
    
    # save model weight
    # path to save trained model
    folder = '../weights/[3-2]MNISTM_SVHN_re'  
    
    # data root
    TRAIN_ROOT_PATH_s = '../hw3_data/digits/mnistm/train'
    TRAIN_ROOT_PATH_t = '../hw3_data/digits/svhn/train'
    
    TEST_ROOT_PATH_s = '../hw3_data/digits/mnistm/test'
    TEST_ROOT_PATH_t = '../hw3_data/digits/svhn/test'

def get_MNISTM_transforms():     
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_SVHN_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class DatasetRetriever_MNISTM(Dataset):
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.correct_count_sc = 0
        self.correct_count_sd = 0
        self.correct_count_tc = 0
        self.correct_count_td = 0
        self.count = 0
        self.acc_sc = 0
        self.acc_sd = 0
        self.acc_tc = 0
        self.acc_td = 0

    def update(self, val, batch_size, ncorrect_sc, ncorrect_sd, ncorrect_tc, ncorrect_td):
        self.sum += val * batch_size
        self.count += batch_size
        self.correct_count_sc += ncorrect_sc
        self.correct_count_sd += ncorrect_sd
        self.correct_count_tc += ncorrect_tc
        self.correct_count_td += ncorrect_td
        self.avg = self.sum / self.count

def rightness(predictions, labels):
    
    pred = torch.max(predictions.data, 1)[1]   # the result of prediction
    rights = pred.eq(labels.data.view_as(pred)).sum()  # count number of correct example
    #print(pred, labels)
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
        
        self.ccriterion = nn.NLLLoss()
        self.dcriterion = nn.NLLLoss()
        #self.dcriterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        
        self.log(f'Fitter prepared. Device is {self.device}') 

    def fit(self, strain_loader, ttrain_loader, tvalid_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
            
            # train
            t = time.time()
            loss = self.train_one_epoch(strain_loader, ttrain_loader)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, loss: {loss.avg:.5f}, acc_sc: {loss.acc_sc:.5f} ,' +\
                     f'acc_sd: {loss.acc_sd:.5f}, acc_tc: {loss.acc_tc:.5f},  acc_td: {loss.acc_td:.5f}' +\
                     f'time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin') 
            
            # validation
            t = time.time()
            loss = self.validation(tvalid_loader)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, loss: {loss.avg:.5f}, acc_sc: {loss.acc_sc:.5f} ,' +\
                     f'acc_sd: {loss.acc_sd:.5f}, acc_tc: {loss.acc_tc:.5f},  acc_td: {loss.acc_td:.5f}' +\
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
            if loss.acc_tc > 0.46:
                self.model.eval()
                self.save(f'{self.base_dir}/{self.epoch}-checkpoint_acc.bin')
            
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

                coutput, _ = self.model(data, 0)
                loss = self.ccriterion(coutput, target)

            loss_recorder.update(loss.detach().item(), batch_size, 0, 0, int(rightness(coutput, target).cpu()), 0)

            
        loss_recorder.acc_sc = loss_recorder.correct_count_sc / loss_recorder.count
        loss_recorder.acc_sd = loss_recorder.correct_count_sd / loss_recorder.count
        loss_recorder.acc_tc = loss_recorder.correct_count_tc / loss_recorder.count
        loss_recorder.acc_td = loss_recorder.correct_count_td / loss_recorder.count
        return loss_recorder

    def train_one_epoch(self, strain_loader, ttrain_loader):
        self.model.train()
        
        loss_recorder = AverageMeter()
        t = time.time()
        
        sloader_len = len(strain_loader)
        i = 0
        
        iter_strain_loader, iter_ttrain_loader = iter(strain_loader), iter(ttrain_loader)
        
        while i < sloader_len:

            print(
                f'Train Step {i}/{len(strain_loader)}, ' + \
                f'loss: {loss_recorder.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}', end='\r'
            )
            
            # count alpha
            p = float(i + self.epoch * sloader_len) / self.config.n_epochs / sloader_len
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
          
            # load data
            simg, slabel = iter_strain_loader.next()
            timg, _ = iter_ttrain_loader.next()
            #print(simg.shape, timg.shape)
            #print(torch.sum(simg), torch.sum(timg))
            batch_size = simg.shape[0]
            sdlabel, tdlabel = torch.zeros((batch_size), dtype=torch.int64), torch.ones((batch_size), dtype=torch.int64)
            
            simg, slabel, sdlabel, timg, tdlabel = simg.cuda(), slabel.cuda(), sdlabel.cuda(), timg.cuda(), tdlabel.cuda()
            
            self.optimizer.zero_grad()
            
            # training by sourse data
            scoutput, sdoutput = self.model(simg, alpha)
            sc_loss = self.ccriterion(scoutput, slabel)
            sd_loss = self.dcriterion(sdoutput, sdlabel)
            
            # training by target data
            _, tdoutput = self.model(timg, alpha)
            td_loss = self.dcriterion(tdoutput, tdlabel)
            
            loss = sc_loss + sd_loss + td_loss
            loss.backward()
            self.optimizer.step()
        
            loss_recorder.update(loss.detach().item(), batch_size, int(rightness(scoutput, slabel).cpu()), 
                    int(rightness(sdoutput, sdlabel).cpu()), 0, int(rightness(tdoutput, tdlabel).cpu()))
            i+=1

        loss_recorder.acc_sc = loss_recorder.correct_count_sc / loss_recorder.count
        loss_recorder.acc_sd = loss_recorder.correct_count_sd / loss_recorder.count
        loss_recorder.acc_tc = loss_recorder.correct_count_tc / loss_recorder.count
        loss_recorder.acc_td = loss_recorder.correct_count_td / loss_recorder.count
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
    sall_meta = pd.read_csv(args.TRAIN_ROOT_PATH_s+'.csv')
    tall_meta = pd.read_csv(args.TRAIN_ROOT_PATH_t+'.csv')
    
    strain_meta = sall_meta
    ttrain_meta = tall_meta[:-len(tall_meta)//9]
    tvalid_meta = tall_meta[-len(tall_meta)//9:].reset_index()
    
    strain_dataset = DatasetRetriever_MNISTM(args.TRAIN_ROOT_PATH_s, strain_meta, get_MNISTM_transforms())
    ttrain_dataset = DatasetRetriever_SVHN(args.TRAIN_ROOT_PATH_t, ttrain_meta, get_SVHN_transforms())
    tvalid_dataset = DatasetRetriever_SVHN(args.TRAIN_ROOT_PATH_t, tvalid_meta, get_SVHN_transforms())
    
    # construct dataloader
    strain_loader = DataLoader(strain_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                                     drop_last=True)
    ttrain_loader = DataLoader(ttrain_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                                     drop_last=True)
    tvalid_loader = DataLoader(tvalid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers,
                                     drop_last=False)

    fitter = Fitter(model=net, device=device, config=args)
    fitter.fit(strain_loader, ttrain_loader, tvalid_loader)

class MNISTM2SVHN(nn.Module):
    def __init__(self, backbone):
        super(MNISTM2SVHN, self).__init__()
        
        # feature extractor
        self.feature = nn.Sequential(
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
        
        # class clasifier
        self.cclassifier = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),

                    nn.Linear(512, 512),
                    nn.ReLU(),

                    nn.Linear(512, 10),
                    nn.LogSoftmax()
#                     nn.BatchNorm1d(64),
#                     nn.ReLU(),
#                     nn.Linear(64, 10),
#                     nn.LogSoftmax()
                )
        
        # domain classifier
        self.dclassifier = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),

                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),

                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),

                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),

                    nn.Linear(512, 2),
                    nn.LogSoftmax(dim=1)
#                     nn.Linear(512, out_features=2, bias=True),
#                     nn.LogSoftmax(dim=1)
#                     nn.BatchNorm1d(64),
#                     nn.ReLU(),
#                     nn.Linear(64, 2),
#                     nn.LogSoftmax(dim=1)
                )
    
    def forward(self, x, alpha):
        x = self.feature(x)
        x = x.view(x.shape[0],-1)
        reverse_x = ReverseLayerF.apply(x, alpha)
        
        return self.cclassifier(x), self.dclassifier(reverse_x)

def main():
    model = MNISTM2SVHN(models.resnet18(pretrained=False))
    run_training(model, Parameter)

if __name__=="__main__":
    main()
