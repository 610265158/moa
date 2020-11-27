import warnings

import sklearn
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')

#the basics
import pandas as pd, numpy as np
import math, json, gc, random, os, sys
from matplotlib import pyplot as plt
from tqdm import tqdm



#for model evaluation
from sklearn.model_selection import train_test_split, KFold



feature_file='../lish-moa/train_features.csv'
target_file='../lish-moa/train_targets_scored.csv'
noscore_target='../lish-moa/train_targets_nonscored.csv'

drug_id_file='../lish-moa/train_drug.csv'

train_features=pd.read_csv(feature_file)
labels_train=pd.read_csv(target_file)
extra_labels_train=pd.read_csv(noscore_target)
drug_info=pd.read_csv(drug_id_file)
###ctrl_vehicle have no moa , set 0 directly in submission


# ####
# ####
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
####
##RankGauss - transform to Gauss
for col in (GENES + CELLS):
    transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
    vec_len = len(train_features[col].values)

    data = train_features[col].values.reshape(vec_len, 1)

    transformer.fit(data)

    train_features[col] = \
        transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]



def preprocess(df):
    """Returns preprocessed data frame"""
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})

    return df
train_features=preprocess(train_features)

train_features = train_features.drop(['sig_id',],axis=1)

labels_train = labels_train.drop('sig_id',axis=1)
extra_labels_train=extra_labels_train.drop('sig_id',axis=1)


print(train_features.shape)
print(labels_train.shape)
print(extra_labels_train.shape)

#### 5 fols split, and ana with val
features = train_features.copy()
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
n_fold=7
target_cols = [c for c in labels_train.columns if c not in ['sig_id']]
features['fold'] = -1
Fold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=0)
for fold, (train_index, test_index) in enumerate(Fold.split(features, labels_train[target_cols])):
    features['fold'][test_index] = fold

oof = np.zeros((features.shape[0], len(target_cols)))


fold=0

###build dataset


val_ind = features.loc[features['fold'] != fold].index.to_list()
train_features = features.iloc[val_ind].copy()
labels_train = labels_train.iloc[val_ind].copy()
drug_info=drug_info.iloc[val_ind].copy()


train_features = train_features.drop(['fold',],axis=1)


print(labels_train.columns.values)

label_cnt_dict={}


for item in labels_train.columns.values:
    label_cnt_dict[item]=np.sum(labels_train[item])


label_cnt_dict_sorted=sorted(label_cnt_dict.items(), key=lambda x: x[1],reverse=True)


for k ,v in label_cnt_dict_sorted:
    print('%-30s: %d sample'%(k,v))





drugid=set(drug_info['drug_id'].tolist())

from lib.core.base_trainer.model import Complexer
import torch

model=Complexer()

weight='resnetlike0_fold0_epoch_28_val_loss0.014526.pth'

device='cpu'
model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
model.to(device)

model.eval()



for id in drugid:

    index=drug_info['drug_id']==id
    cur_sig_ids= drug_info.loc[index]['sig_id']


    data=train_features.loc[index].values
    label=labels_train.loc[index].values




    with torch.no_grad():
        input_data = torch.from_numpy(data).to(device).float()
        # input_data = input_data.unsqueeze(0)
        pre, _ = model(input_data)
        pre = torch.sigmoid(pre)
        pre = pre.cpu().numpy()


    y_true=label

    y_pred=pre


    score=0.
    for i in range(206):
        score_ = log_loss(y_true[:, i], y_pred[:, i],labels=[0,1])
        score += score_ / y_pred.shape[1]





    if score>0.01:
        print('with drugid: ', id, 'there is %2d samples ' % (np.sum(index)), '  logloss is %.6f' % score)

        for kk in range(data.shape[0]):
            cur_item=data[kk,...]
            cur_label=label[kk,...]

            with torch.no_grad():
                input_data = torch.from_numpy(cur_item).to(device).float()
                input_data = input_data.unsqueeze(0)
                pre, _ = model(input_data)
                pre = torch.sigmoid(pre)
                pre = pre.cpu().numpy()[0]

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(range(len(pre)), pre,color='blue', label='pre',linewidth=1.8)

            plt.plot(range(len(cur_label)), cur_label,color='red', label='target', linewidth=0.8)
            plt.ylim(0, 1)

            plt.subplot(2, 1, 2)
            plt.plot(range(len(cur_item)), cur_item)
            plt.ylim(-5, 5)
            #
            plt.show()




