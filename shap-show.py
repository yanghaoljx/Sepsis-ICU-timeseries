from __future__ import print_function
import shap
from Postive_dataloader import Positive_dataset
import torch
from model import Transformer
from torch.utils.data import  DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns
dim = 24
depth = 3
heads = 4
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_path = 'test'
batch_size = 100
trans_model = Transformer(
    d_model=dim,
    N = depth,
    heads = heads,
    dropout = dropout
).to(device)
trans_model.load_state_dict(torch.load('./record/model.pth'))
trans_model.eval()
positive_test = Positive_dataset(test_path)
test_loader = DataLoader(positive_test,batch_size,shuffle=False,drop_last=True)
shap_merge = 0
test_num = 1
for adm_matrix, labels in test_loader:
    adm_matrix = adm_matrix.to(device)
    background = adm_matrix
    explainer = shap.DeepExplainer(trans_model,background)
    shap_values = explainer.shap_values(background)
    shap_value = shap_values[1]
    for i in range(shap_value.shape[0]):
        shap_merge += shap_value[i]
        test_num += 1
shap_average = shap_merge/test_num
minMax = MinMaxScaler()
shap_std = minMax.fit_transform(shap_average.reshape((24,-1)))
feature_names = pd.read_csv('总特征.csv',header=None)
for i in range(0,shap_std.shape[1],24):
    sns.heatmap(shap_std[:,i:i+24],cmap='RdYlGn_r', robust=True, fmt='.2f', 
                    annot=True, linewidths=.5, annot_kws={'size':11},xticklabels=feature_names[1][i:i+24])
    plt.xticks(fontsize=12)
    plt.xticks(rotation=30)
    plt.show()
    plt.close()
# batch = next(iter(test_loader))
# background,_ = batch 
# background = background.to(device)
# explainer = shap.DeepExplainer(trans_model,background)
# shap_values = explainer.shap_values(background[0:1])
# shap_values = shap_values.cpu()
# test_images = background[0:1].cpu()
# shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
# test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
# test_numpy.shape
# shap.image_plot(shap_numpy[0][:,:,:24,:],-test_numpy[:,:,:24,:])


