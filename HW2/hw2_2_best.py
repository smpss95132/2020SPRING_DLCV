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
from datetime import datetime
import time
import torchvision.models as models
import matplotlib.pyplot as plt
import random
import sys


pixel2class = {0:(0,255,255), 1:(255,255,0), 
                   2:(255,0,255), 3:(0,255,0), 
                   4:(0,0,255), 5:(255,255,255),
                   6:(0,0,0)}

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# dataset define
class DatasetRetriever(Dataset):
    def __init__(self, root_path, transforms):
        super().__init__()
        
        self.root_path = root_path                      
        self.names = sorted([s[:4] for s in os.listdir(root_path) if s.endswith("_sat.jpg")])
        self.transforms = transforms

    def __getitem__(self, idx: int):
        # image info
        name = self.names[idx]
        
        # load image
        with open(os.path.join(self.root_path, name+'_sat.jpg'), 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
        
        img = self.transforms(img)

        return img, name.split('_')[0]
    
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

def mask2RGB(mask):
    maskRGB = np.empty((mask.shape[0], mask.shape[1], 3))
    
    for key in pixel2class.keys():
        pixel_value = pixel2class[key]
        maskRGB[mask==key] = pixel_value
    
    #return np.moveaxis(maskRGB, -1, 0).astype(np.uint8)
    return maskRGB.astype(np.uint8)

def main():
    # define and load model
    model = FCN16s(num_class=7)
    model.load_state_dict(torch.load('hw2_FCN16S_bn.bin?dl=1')['model_state_dict'])

    # dataset and dataloader
    test_dataset = DatasetRetriever(sys.argv[1], get_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,drop_last=False)

    model.cuda()
    model.eval()    

    for _, (data, name) in enumerate(test_loader):
        with torch.no_grad():
    
            data = data.cuda()

            # predict and post process
            output = model(data).detach().cpu()
            pre = torch.argmax(output, dim=1)
            
            # label mask to RGB mask
            RGB_pre = mask2RGB(pre.numpy().reshape(512,512))
            RGB_pre = Image.fromarray(RGB_pre) 
            
            # save mask
            save_path = os.path.join(sys.argv[2], name[0]+'_mask.png')
            RGB_pre.save(save_path)

if __name__=='__main__':
    main()
