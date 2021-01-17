import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from torchvision import transforms
import torchvision.models as models
import numpy as np
import pandas as pd
import glob 
import os
import sys
from PIL import Image


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def get_transforms():     
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class DatasetRetriever_MNISTM(Dataset):
    def __init__(self, root_path, transforms):
        super().__init__()
        
        self.root_path = root_path                 
        self.transforms = transforms
        self.img_names = sorted(os.listdir(root_path))

        print("Dataset is created!")
        print(f"transforms: {self.transforms}")
        
    def __getitem__(self, idx: int):
        # image information
        img_name = self.img_names[idx]
        
        # load image
        with open(os.path.join(self.root_path, img_name), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            
        return self.transforms(img)
    
    def __len__(self) -> int:
        return len(self.img_names)

class DatasetRetriever_SVHN(Dataset):
    def __init__(self, root_path, transforms):
        super().__init__()
        
        self.root_path = root_path                 
        self.transforms = transforms
        self.img_names = sorted(os.listdir(root_path))

        print("Dataset is created!")
        print(f"transforms: {self.transforms}")
        
    def __getitem__(self, idx: int):
        # image information
        img_name = self.img_names[idx]
        
        # load image
        with open(os.path.join(self.root_path, img_name), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            
        return self.transforms(img)
    
    def __len__(self) -> int:
        return len(self.img_names)

class DatasetRetriever_USPS(Dataset):
    def __init__(self, root_path, transforms):
        super().__init__()
        
        self.root_path = root_path                 
        self.transforms = transforms
        self.img_names = sorted(os.listdir(root_path))

        print("Dataset is created!")
        print(f"transforms: {self.transforms}")
        
    def __getitem__(self, idx: int):
        # image information
        img_name = self.img_names[idx]
        
        # load image
        with open(os.path.join(self.root_path, img_name), 'rb') as f:
            img = Image.open(f)
            img.load()
        # grayscale to color
        temp = transforms.ToTensor()(img)
        img = torch.cat([temp, temp, temp], axis=0)
        img = transforms.ToPILImage()(img)
            
        return self.transforms(img)
    
    def __len__(self) -> int:
        return len(self.img_names)

class DANN(nn.Module):
    def __init__(self, backbone):
        super(DANN, self).__init__()
        
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
                )
    
    def forward(self, x, alpha):
        x = self.feature(x)
        x = x.view(x.shape[0],-1)
        reverse_x = ReverseLayerF.apply(x, alpha)
        
        return self.cclassifier(x), self.dclassifier(reverse_x)

def main():
    # define and load model / dataset
    model = DANN(models.resnet18(pretrained=False))

    if sys.argv[2]=="mnistm":
        checkpoint = torch.load("USPS2MNISTM.bin?dl=1")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        test_dataset = DatasetRetriever_MNISTM(sys.argv[1], get_transforms())
    elif sys.argv[2]=="usps":
        checkpoint = torch.load("SVHN2USPS.bin?dl=1")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        model.eval()
        test_dataset = DatasetRetriever_USPS(sys.argv[1], get_transforms())
    else:
        checkpoint = torch.load("MNISTM2SVHN.bin?dl=1")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        model.eval()
        test_dataset = DatasetRetriever_SVHN(sys.argv[1], get_transforms())
    
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8,drop_last=False)
    
    preds = []
    # testing
    for step, data in enumerate(test_loader):
        with torch.no_grad():
            batch_size = data.shape[0]
            data = data.cuda()

            output, _ = model(data, 0)
            pred = torch.max(output.data, 1)[1]
            preds.append(pred)
    
    result = list(torch.cat(preds, dim=0))
    result = [int(n) for n in result]

    # create csv file
    result_df = pd.DataFrame (zip(test_dataset.img_names, result), columns=['image_name', 'label'])
    result_df.to_csv(sys.argv[3], index = False)

if __name__=="__main__":
    main()