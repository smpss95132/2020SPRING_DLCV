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

class Discriminator0(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator0, self).__init__()
        
        self.layer = nn.Sequential(
                    nn.Linear(input_dim, input_dim//4),
                    nn.ReLU(),
                    nn.Linear(input_dim//4, input_dim//16),
                    nn.ReLU(),
                    nn.Linear(input_dim//16, input_dim//64),
                    nn.ReLU(),
                    nn.Linear(input_dim//64, output_dim),
                    nn.LogSoftmax()
                )
        
    def forward(self, x):
        return self.layer(x)

class Discriminator1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator1, self).__init__()
        
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

class ADDA(nn.Module):
    def __init__(self, backbone):
        super(ADDA, self).__init__()
        
        self.SEncoder = Encoder(models.resnet18(pretrained=False))
        self.TEncoder = Encoder(models.resnet18(pretrained=False))
        
        self.cls = Classifier(512, 10)
        
        self.dsc = Discriminator1(512, 2)
    
    def forward(self, x, domain):
        if domain=="S":
            x = self.SEncoder(x)
            return self.cls(x)
        else:
            x = self.TEncoder(x)
            return self.cls(x)

def main():
    # define and load model / dataset
    model = ADDA(models.resnet18(pretrained=False))

    if sys.argv[2]=="mnistm":
        checkpoint = torch.load("[4]USPS2MNISTM.bin?dl=1")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        test_dataset = DatasetRetriever_MNISTM(sys.argv[1], get_transforms())
    elif sys.argv[2]=="usps":
        checkpoint = torch.load("[4]SVHN2USPS.bin?dl=1")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        model.eval()
        test_dataset = DatasetRetriever_USPS(sys.argv[1], get_transforms())
    else:
        model.dsc = Discriminator0(512, 2)
        checkpoint = torch.load("[4]MNISTM2SVHN.bin?dl=1")
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

            output = model(data, "T")
            pred = torch.max(output.data, 1)[1]
            preds.append(pred)
    
    result = list(torch.cat(preds, dim=0))
    result = [int(n) for n in result]

    # create csv file
    result_df = pd.DataFrame (zip(test_dataset.img_names, result), columns=['image_name', 'label'])
    result_df.to_csv(sys.argv[3], index = False)

if __name__=="__main__":
    main()