from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle 
import pandas as pd 
import os 

labels = pd.read_csv('preprocessed_labels.csv')
labels.set_index('patient',inplace=True)
labels.index
train,test = train_test_split(labels.index,test_size=0.15,random_state=41)
train,val = train_test_split(train,test_size=0.15/0.85,random_state=41)
print('===> loading data for splitting....')
timeseries = pd.read_csv('preprocessed_timeseries.csv')
timeseries.set_index('patient',inplace=True)
flat_features = pd.read_csv('preprocessed_flat.csv')
flat_features.set_index('patient',inplace=True)
for partition_name,partition in zip(['train','val','test'],[train,test,val]):
    print('===>Preparing {} data...'.format(partition_name))
    stays = partition 
    folder_path = './'+partition_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    stays = shuffle(stays)
    with open(folder_path+'/stays.txt','w') as f:
        for table_name,table in zip(['labels','flat','timeseries'],[labels, flat_features, timeseries]):
            table = table.loc[stays].copy()
            table.to_csv('{}/{}.csv'.format(folder_path,table_name))
            for stay in table.index:
                f.write("%s\n"%stay)
    
