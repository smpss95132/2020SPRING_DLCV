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
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

# model structure
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
        
        self.hal_weight = nn.Sequential(
                                nn.Linear(1600, 1),
                                nn.ReLU(),
                            )
        
        self.out_channels = 1600
        self.distance_matric = Parametric(self.out_channels)


    def forward(self, x):
        x = self.feature_extractor(x)
        return x
    
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
    
    def weight_evaluation(self, prototype):
        return self.hal_weight(prototype)
    
    def distance_evaluation(self, prototype, query_feature):
        num_way, feature_len = prototype.shape
        num_query, _ = query_feature.shape
        
        distances = []
        for q in range(num_query):
            query = query_feature[q]
            diff = query - prototype
            distances.append(self.distance_matric(diff))
       
        return torch.stack(distances).squeeze(2)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def predict(args, model, data_loader):

    with torch.no_grad():
        preds = []
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            batch_size = args.N_way * args.N_query

            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] .cuda()
            query_input   = data[args.N_way * args.N_shot:,:,:,:].cuda()

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            # TODO: extract the feature of support and query data
            prototype_feature = model.extract_feature(support_input)
            prototype_ori = prototype_feature.unsqueeze(1)
            prototype_hal = model.hallucinate(prototype_feature, args.N_hal)
            prototype_hal = prototype_hal.view(prototype_feature.shape[0], args.N_hal, -1)

            query_feature = model.extract_feature(query_input)
            
            # TODO: calculate the prototype for each class according to its support data
            prototype = torch.cat([prototype_ori, prototype_hal], dim=1) 
            prototype_weight = model.weight_evaluation(prototype.view(-1,prototype.shape[2]))
            weighted_prototype = (prototype.view(-1,prototype.shape[2]) * prototype_weight).view(-1,args.N_hal+1,prototype.shape[2])
            prototype = prototype.mean(dim=1)

            # TODO: classify the query data depending on the its distense with each prototype
            distance = model.distance_evaluation(prototype, query_feature)
            preds.append(torch.max(distance.data, 1)[1])

        prediction_results = list(torch.cat(preds, dim=0))
        prediction_results = [int(n) for n in prediction_results]

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--N-hal', default=1, type=int, help='N_hal (default: 1)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model
    model = PrototypicalHallucination()
    checkpoint = torch.load(args.load, map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()

    prediction_results = predict(args, model, test_loader)

    # TODO: output your prediction to csv
    batch_size = args.N_way * args.N_query
    columns = ['episode_id'] + ['query'+str(i) for i in range(batch_size)]
    prediction_results = [tuple([i] + prediction_results[i*batch_size:(i+1)*batch_size]) for i  in range(len(test_loader))]

    result_df = pd.DataFrame(prediction_results, columns=columns)
    result_df.to_csv(os.path.join(args.output_csv, ), index = False)
