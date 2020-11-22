import random

import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from lib.core.base_trainer.net_work import Train
from sklearn.metrics import log_loss

from lib.core.model.semodel.SeResnet import se_resnet50
import cv2
import numpy as np
import pandas as pd, numpy as np
from train_config import config as cfg
import setproctitle

from lib.dataset.dataietr import DataIter
setproctitle.setproctitle("alaska")

from train_config import seed_everything
from lib.helper.logger import logger

import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from lib.core.base_trainer.model import Complexer
from lib.core.base_trainer.densenet import Denseplexer
from lib.core.base_trainer.table import Tablenet
from lib.core.base_trainer.hourglass import Hourglass
from lib.core.base_trainer.mlp import MLP
def main():


    train_feature_file = '../lish-moa/train_features.csv'
    target_file = '../lish-moa/train_targets_scored.csv'
    noscore_target = '../lish-moa/train_targets_nonscored.csv'

    train_features = pd.read_csv(train_feature_file)
    labels = pd.read_csv(target_file)
    extra_labels = pd.read_csv(noscore_target)

    test_features = pd.read_csv('../lish-moa/test_features.csv')

    pub_test_features=test_features.copy()

    def get_fe(train_features,test_features):
        n_comp_GENES = 463  ## pca features
        n_comp_CELLS = 60  ## pca features
        VarianceThreshold_for_FS = 0.9

        GENES = [col for col in train_features.columns if col.startswith('g-')]
        CELLS = [col for col in train_features.columns if col.startswith('c-')]
        for col in (GENES + CELLS):
            transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
            vec_len = len(train_features[col].values)
            vec_len_test = len(test_features[col].values)
            raw_vec = train_features[col].values.reshape(vec_len, 1)
            transformer.fit(raw_vec)
            train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
            test_features[col] = \
            transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]

        def seed_everything(seed=42):
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        seed_everything(seed=42)

        g_pca = PCA(n_components=n_comp_GENES, random_state=42)
        train_features = pd.concat((train_features, pd.DataFrame(g_pca.fit_transform(train_features[GENES]),
                                                                 columns=[f'pca_G-{i}' for i in range(n_comp_GENES)])),
                                   axis=1)
        test_features = pd.concat((test_features, pd.DataFrame(g_pca.transform(test_features[GENES]),
                                                               columns=[f'pca_G-{i}' for i in range(n_comp_GENES)])),
                                  axis=1)

        c_pca = PCA(n_components=n_comp_CELLS, random_state=42)
        train_features = pd.concat((train_features, pd.DataFrame(c_pca.fit_transform(train_features[CELLS]),
                                                                 columns=[f'pca_C-{i}' for i in range(n_comp_CELLS)])),
                                   axis=1)
        test_features = pd.concat((test_features, pd.DataFrame(c_pca.transform(test_features[CELLS]),
                                                               columns=[f'pca_C-{i}' for i in range(n_comp_CELLS)])),
                                  axis=1)

        var_thresh = VarianceThreshold(VarianceThreshold_for_FS)
        train_transformed = var_thresh.fit_transform(train_features.iloc[:, 4:])
        test_transformed = var_thresh.transform(test_features.iloc[:, 4:])

        train_features = pd.DataFrame(train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                      columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
        train_features = pd.concat([train_features, pd.DataFrame(train_transformed)], axis=1)
        test_features = pd.DataFrame(test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                     columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
        test_features = pd.concat([test_features, pd.DataFrame(test_transformed)], axis=1)
        print(train_features.shape)


        return train_features,test_features


    train_features,test_features=get_fe(train_features,test_features)


    print(train_features.shape)
    num_features=train_features.shape[1]-1
    losscolector=[]
    folds=[0,1,2,3,4,5,6]
    seeds=[0]

    n_fold=len(folds)

    model_dicts=[{'name': 'MLP','func':MLP}
                 ]






    def run_k_fold(model_dict, seed):
        seed_everything(seed)

        #### 5 fols split
        features = train_features.copy()

        print(features.shape)
        target_cols = [c for c in labels.columns if c not in ['sig_id']]
        features['fold'] = -1
        Fold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cur_seed)
        for fold, (train_index, test_index) in enumerate(Fold.split(features, labels[target_cols])):
            features['fold'][test_index] = fold



        oof=np.zeros((features.shape[0], len(target_cols)))

        print('oof',oof.shape)

        for fold in folds:


            logger.info('train with seed %d' % (cur_seed))

            ###build dataset
            train_ind = features[features['fold'] != fold].index.to_list()
            train_features_=features.iloc[train_ind].copy()
            train_target_ = labels.iloc[train_ind].copy()
            train_extra_Target_ = extra_labels.iloc[train_ind].copy()

            val_ind=features.loc[features['fold'] == fold].index.to_list()
            val_features_ = features.iloc[val_ind].copy()
            val_target_ = labels.iloc[val_ind].copy()
            val_extra_Target_ = extra_labels.iloc[val_ind].copy()



            if cfg.DATA.filter_ctl_vehicle:

                oof_index=(features['fold'] == fold)&(features['cp_type']!='ctl_vehicle')

                oof_index = features.loc[oof_index].index.to_list()
                print('zzzzz',len(oof_index))

            else:
                oof_index = (features['fold'] == fold)
            train_ds=DataIter(train_features_,train_target_,train_extra_Target_,shuffle=True,training_flag=True)
            val_ds=DataIter(val_features_,val_target_,val_extra_Target_,shuffle=False,training_flag=False)

            ### build model

            model=model_dict['func'](num_features=num_features)

            model_name=str(model_dict['name']+str(cur_seed))



            if cfg.TRAIN.pretrain_on_no_score:
                ###build trainer
                trainer = Train(model_name='PRETRAIN_'+model_name, model=model, train_ds=train_ds, val_ds=val_ds, fold=fold)

                trainer.pretrain=True
                ### pretrian first with no score
                loss,best_model=trainer.custom_loop()

                ### train
                trainer_fine = Train(model_name=model_name, model=model, train_ds=train_ds, val_ds=val_ds, fold=fold)
                trainer_fine.load_from(best_model)
                trainer_fine.pretrain=False
                loss, best_model = trainer_fine.custom_loop()
            else:
                ###build trainer
                trainer = Train(model_name=model_name, model=model, train_ds=train_ds, val_ds=val_ds, fold=fold)

                trainer.pretrain = False
                loss, best_model,oof_predict = trainer.custom_loop()

                oof[oof_index]=oof_predict.cpu().numpy()



            if cfg.TRAIN.finetune_alldata:
                ### finetune with all data
                train_features_ = features.copy()
                train_target_ = labels.copy()
                train_extra_Target_ = extra_labels.copy()

                val_ind = features.loc[features['fold'] == fold].index.to_list()
                val_features_ = features.iloc[val_ind].copy()
                val_target_ = labels.iloc[val_ind].copy()
                val_extra_Target_ = extra_labels.iloc[val_ind].copy()

                train_ds = DataIter(train_features_, train_target_, train_extra_Target_, shuffle=True,
                                    training_flag=True)
                val_ds = DataIter(val_features_, val_target_, val_extra_Target_, shuffle=False, training_flag=False)

                ### build model
                model = model_dict['func']()
                model_name = str(model_dict['name'] + str(cur_seed))

                ###build trainer
                trainer = Train(model_name=model_name, model=model, train_ds=train_ds, val_ds=val_ds, fold=fold)

                trainer.reset(best_model)

                loss, best_model = trainer.custom_loop()


            losscolector.append([loss,best_model])


        return oof

    for model_dict in model_dicts:
        # for cur_seed in seeds:
        if model_dict['name']=='tablenet':
            cfg.TRAIN.init_lr = 5.e-3

        oof = np.zeros((train_features.shape[0], 206))
        for cur_seed in seeds:
            oof_=run_k_fold(model_dict,cur_seed)

            oof+=oof_/len(seeds)


        y_true = labels.drop('sig_id', axis=1).values

        y_pred = oof
        score=0
        for i in range(206):
            score_= log_loss(y_true[:, i], y_pred[:, i])
            score += score_ / y_pred.shape[1]
        print("CV log_loss: ", score)

        avg_loss=0
        for k,loss_and_model in enumerate(losscolector):
            print('fold %d : loss %.5f modelname: %s'%(k,loss_and_model[0],loss_and_model[1]))
            avg_loss+=loss_and_model[0]
        print('simple,average loss is ',avg_loss/(len(losscolector)))


if __name__=='__main__':
    main()