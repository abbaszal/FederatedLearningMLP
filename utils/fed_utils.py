import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import copy

class HuGaDataset(Dataset):
    def __init__(self, csv_path=None, dataframe=None, label_map=None):
        if csv_path:
            df = pd.read_csv(csv_path)
        else:
            df = dataframe
        self.X = df.iloc[:, :-1].values.astype(np.float32)
        y_strings = df.iloc[:, -1].values
        self.y = np.array([label_map[label] for label in y_strings], dtype=np.int64)
        mean = self.X.mean(axis=0)
        std = self.X.std(axis=0) + 1e-8
        self.X = (self.X - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# fedavg
def aggregate_models(global_model, client_models, client_weights):

    global_dict = global_model.state_dict()
    
    avg_dict = copy.deepcopy(global_dict)
    for key in avg_dict:
        avg_dict[key] = torch.zeros_like(avg_dict[key])
        
    for client_model, weight in zip(client_models, client_weights):
        client_dict = client_model.state_dict()
        for key in avg_dict:
            avg_dict[key] += client_dict[key] * weight

    global_model.load_state_dict(avg_dict)
    return global_model