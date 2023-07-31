from __future__ import print_function
#from functools import total_ordering
from custom_dataloader import Custom_dataset
from torch.utils.data import  DataLoader
from sklearn.metrics import roc_curve,auc
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import Transformer
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer'
        self.test_path = 'test'                                 # 测试集
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.batch_size = 32                                          # mini-batch大小
        self.learning_rate = 1e-6                                   # 学习率
        self.seed = 0
        self.dim = 24 
        self.heads = 4
        self.depth = 3
        self.dropout = 0.1
         
config = Config()
custom_test = Custom_dataset(config.test_path)
test_loader = DataLoader(custom_test,config.batch_size,shuffle=False,drop_last=True)
device = config.device
trans_model = Transformer(
    d_model=config.dim,
    N = config.depth,
    heads = config.heads,
    dropout = config.dropout
).to(device)

trans_model.load_state_dict(torch.load('./record/model.pth'))
score_list = [] 
label_list = [] 
preds_list = [] 
for data, label in test_loader:
    data = data.to(device)
    label = label.to(device)
    test_output = trans_model(data)
    preds = torch.argmax(test_output,1)
    preds_list.extend(preds.cpu().numpy())
    score_list.extend(test_output.detach().cpu().numpy())
    label_list.extend(label.cpu().numpy())
print(classification_report(label_list,preds_list,digits=3))
conf_matrix=confusion_matrix(label_list, preds_list)
sns.heatmap(conf_matrix,fmt='g',annot=True,cbar=False)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
plt.close()
score_array = np.array(score_list)
label_tensor = torch.tensor(label_list)
label_tensor = label_tensor.reshape((label_tensor.shape[0],1))
label_onehot = torch.zeros(label_tensor.shape[0],2)
label_onehot.scatter_(dim=1,index=label_tensor,value=1)
label_onehot = np.array(label_onehot)

print("score_array: ", score_array.shape)
print("label_onehot: ", label_onehot.shape)

fpr_dict = dict()
tpr_dict = dict()
roc_auc_dict = dict()

fpr_dict['micro'],tpr_dict['micro'], _ = roc_curve(label_onehot.ravel(),score_array.ravel())
roc_auc_dict['micro'] = auc(fpr_dict['micro'],tpr_dict['micro'])
import pandas as pd 
test_df = pd.read_csv('test/labels.csv')
fpr1,tpr1, _ = roc_curve(test_df['actualhospitalmortality'],test_df['predictedhospitalmortality'])
roc_auc1 = auc(fpr1,tpr1)

plt.figure()
lw =2 
plt.plot(fpr_dict['micro'],tpr_dict['micro'],
            label='ROC curve (area={0:0.2f})'.format(roc_auc_dict['micro']),
            color='gold',linestyle=':',linewidth=4)
plt.plot(fpr1,tpr1,
            label='Apache Score ROC curve (area={0:0.2f})'.format(roc_auc1),
            color='gray',linestyle=':',linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Day-5 Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('set113_roc.jpg')
plt.show()



