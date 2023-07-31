import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd 
class Custom_dataset(Dataset):
    def __init__(self,tpath):
        super().__init__()
        self.stayids = pd.read_csv(tpath+'/labels.csv')['patient'].to_list()
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

if __name__=='__main__':
    custom_dataset = Custom_dataset('./val')
    data_loader = DataLoader(custom_dataset,batch_size=1,shuffle=False)
    for i_batch,batch_data in enumerate(data_loader):
        if i_batch > 5:
            break
        print(i_batch)
        print(batch_data[0].shape)
        print(batch_data[1].shape)
    timeseries = pd.read_csv('train'+'/timeseries.csv')
    flat = pd.read_csv('train'+'/flat.csv')
    merge = pd.merge(timeseries,flat,on='patient',how='left')
    pd.Series(merge.columns[2:]).to_csv('总特征.csv',header=None)


        