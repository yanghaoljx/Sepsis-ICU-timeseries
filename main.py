from __future__ import print_function
#from functools import total_ordering
from custom_dataloader import Custom_dataset
from torch.utils.data import  DataLoader
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from model import Transformer
from focalloss import FocalLoss



class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer'
        self.train_path = 'train'                        # 训练集
        self.dev_path = 'val'                                 # 验证集
        self.test_path = 'test'                                 # 测试集
        self.save_path =  '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path =  '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.num_classes = 2                                            # 类别数
        self.num_epochs = 50                                          # epoch数
        self.batch_size = 32                                          # mini-batch大小
        self.learning_rate = 1e-5                                   # 学习率
        self.seed = 0
        self.dim = 24 
        self.heads = 4
        self.depth = 3
        #self.feature_num = 226
        self.dropout = 0.5
         
config = Config()
device = config.device
trans_model = Transformer(
    d_model=config.dim,
    #m = config.feature_num,
    N = config.depth,
    heads = config.heads,
    dropout = config.dropout
).to(device)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
seed_everything(config.seed)
custom_train = Custom_dataset(config.train_path)
train_loader = DataLoader(custom_train,batch_size=config.batch_size,shuffle=False,drop_last=True)
custom_val = Custom_dataset(config.dev_path)
val_loader = DataLoader(custom_val,config.batch_size,shuffle=False,drop_last=True)
custom_test = Custom_dataset(config.test_path)
test_loader = DataLoader(custom_test,config.batch_size,shuffle=False,drop_last=True)

# loss function
criterion = FocalLoss(gamma=2,alpha=0.8)
# optimizer
optimizer = optim.Adam(trans_model.parameters(), lr=config.learning_rate)

min_loss = 150000
for epoch in range(config.num_epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader,total = len(train_loader)):
        data = data.to(device)
        label = label.to(device)
        
        output = trans_model(data)
        loss = criterion(output, label.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)  
            val_output = trans_model(data)
            #print(val_output.shape)
            val_loss = criterion(val_output, label.squeeze())
            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)
        if epoch_val_loss < min_loss:
            min_loss = epoch_val_loss
            print('save model')
            torch.save(trans_model.state_dict(),'record/model.pth')
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
#####测试集测试 
trans_model.load_state_dict(torch.load('./record/model.pth'))
test_accuracy = 0 
for data, label in test_loader:
    data = data.to(device)
    label = label.to(device)
    test_output = trans_model(data)
    acc = (test_output.argmax(dim=1) == label).float().mean()
    test_accuracy += acc/len(test_loader)
print(
    f"Test accuracy value:{test_accuracy:.4f}-----\n"
)
