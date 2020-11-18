import pandas as pd

import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from train_config import config as cfg
from lib.core.base_trainer.model import Complexer
from lib.dataset.dataietr import DataIter




fold=0
n_fold=7
cur_seed=40

###build dataset
train_feature_file = '../lish-moa/train_features.csv'
target_file = '../lish-moa/train_targets_scored.csv'
noscore_target = '../lish-moa/train_targets_nonscored.csv'

train_features = pd.read_csv(train_feature_file)
labels = pd.read_csv(target_file)
extra_labels = pd.read_csv(noscore_target)

test_features = pd.read_csv('../lish-moa/test_features.csv')

pub_test_features = test_features.copy()

#### 5 fols split
features = train_features.copy()
target_cols = [c for c in labels.columns if c not in ['sig_id']]
features['fold'] = -1
Fold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cur_seed)
for fold, (train_index, test_index) in enumerate(Fold.split(features, labels[target_cols])):
    features['fold'][test_index] = fold

train_ind = features[features['fold'] != fold].index.to_list()
train_features_=features.iloc[train_ind].copy()
train_target_ = labels.iloc[train_ind].copy()
train_extra_Target_ = extra_labels.iloc[train_ind].copy()

val_ind=features.loc[features['fold'] == fold].index.to_list()
val_features_ = features.iloc[val_ind].copy()
val_target_ = labels.iloc[val_ind].copy()
val_extra_Target_ = extra_labels.iloc[val_ind].copy()







cfg.TRAIN.batch_size=1
train_ds=DataIter(train_features_,train_target_,train_extra_Target_,shuffle=True,training_flag=True)
val_ds=DataIter(val_features_,val_target_,val_extra_Target_,shuffle=False,training_flag=False)


model=Complexer()

weight='./resnetlike40_fold0_epoch_42_val_loss0.014642.pth'
device="cpu"

model.load_state_dict(torch.load(weight, map_location=device))
model.to(device)
model.eval()





result_dict={}

for i in range(val_ds.size):
    data, target1, target2 = val_ds()





    for kk in range(1):

        for j in range(data.shape[0]):


            input=data[j,...]
            label1=target1[j,...]
            label2 = target2[j, ...]






            input = torch.from_numpy(input).to(device).float()

            print(input.shape)
            input = input.unsqueeze(0)

            pre,__=model(input)
            pre=torch.nn.functional.sigmoid(pre)
            pre=pre.data.cpu().numpy()[0]


            plt.subplot(2, 1, 1)

            plt.plot(range(206), pre, color='blue', label='pre', linewidth=1.8) #
            plt.plot(range(206), label1, color='red', label='target', linewidth=0.8)  #
            plt.legend()
            plt.ylim(0, 1)
            plt.subplot(2, 1, 2)
            plt.plot(range(875), input.cpu().numpy()[0,...], color='blue', label='data', linewidth=1.8)  #
            plt.ylim(-10, 10)
            plt.legend()
            plt.show()



    plt.close('all')
##predict


