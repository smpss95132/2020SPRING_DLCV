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
import sys
from pandas import DataFrame

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class DatasetRetriever(Dataset):

    def __init__(self, root_path, transforms):
        super().__init__()
        
        self.root_path = root_path                      
        self.image_names = sorted(os.listdir(root_path))
        self.transforms = transforms

    def __getitem__(self, idx: int):
        
        # image info
        image_name = self.image_names[idx]
        path = os.path.join(self.root_path, image_name)
        
        # load image and preprocess image
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
        
        img = self.transforms(img)
            
        return img, image_name
    
    def __len__(self) -> int:
        return len(self.image_names)
    
def main():
    # define model
    model = models.vgg16_bn(pretrained=False)
    model.classifier = nn.Linear(in_features=25088, out_features=50, bias=True)
    
    # load model
    model.load_state_dict(torch.load('hw1_vgg16_224.bin?dl=1')['model_state_dict'])
    model.cuda()
    model.eval()
    
    # create dataset
    test_dataset = DatasetRetriever(sys.argv[1], get_test_transforms())
    device = torch.device('cuda:0')
    
    # testing
    recorder = []
    t = time.time()
    for data, name in test_dataset:
        with torch.no_grad():
            data = data.cuda()
            output = model(data.unsqueeze(0)).detach()
            
            output = torch.sum(output, axis=0)
            output = int(torch.max(output.data, 0)[1].cpu())
            recorder.append((name, output))

    print("time uesd: ", time.time()-t)
    
    # create csv file
    result_df = DataFrame (columns=['image_id', 'label'])

    for record in recorder:
        result_df = result_df.append({'image_id':record[0], 'label':record[1]}, ignore_index=True)

    result_df.to_csv(os.path.join(sys.argv[2], 'test_pred.csv'), index = False)
    
if __name__ == "__main__":
    main()
