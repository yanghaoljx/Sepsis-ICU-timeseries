import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd 
class Positive_dataset(Dataset):
    def __init__(self,tpath):
        super().__init__()
        label_csv = pd.read_csv(tpath+'/labels.csv')
        self.stayids=  label_csv[label_csv['actualhospitalmortality']==1]['patient'].to_list()
        self.timeseries = pd.read_csv(tpath+'/timeseries.csv')
        self.timeseries.replace('Unknown',0,inplace=True)
        self.timeseries.fillna(0,inplace=True)
        self.flat = pd.read_csv(tpath+'/flat.csv')
        self.flat.replace('Unknown',0,inplace=True)
        self.flat.fillna(0,inplace=True)
        self.merge = pd.merge(self.timeseries,self.flat,on='patient',how='left')
        self.labels = pd.read_csv(tpath+'/labels.csv')[['patient','actualhospitalmortality']]
    def __getitem__(self, index):
        self.patient = self.stayids[index]
        self.src = self.merge[self.merge['patient']==self.patient].iloc[:,2:]
        self.src = np.array(self.src).astype(float)
        self.src = torch.tensor(self.src).unsqueeze(0)
        self.src = self.src.to(torch.float32)
        self.label = self.labels.loc[self.labels['patient']==self.patient]['actualhospitalmortality'].values
        self.label = torch.LongTensor(self.label)
        return self.src, self.label
    
    def __len__(self):
        return len(self.stayids)