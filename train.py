import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from lib.core.base_trainer.net_work import Train


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
from lib.core.base_trainer.wide_and_depp import WideAndDeep

def main():


    train_feature_file = '../lish-moa/train_features.csv'
    target_file = '../lish-moa/train_targets_scored.csv'
    noscore_target = '../lish-moa/train_targets_nonscored.csv'

    train_features = pd.read_csv(train_feature_file)
    labels = pd.read_csv(target_file)
    extra_labels = pd.read_csv(noscore_target)

    test_features = pd.read_csv('../lish-moa/test_features.csv')

    pub_test_features=test_features.copy()

    ####
    ####
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]
    ####
    # RankGauss - transform to Gauss
    for col in (GENES + CELLS):
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        vec_len = len(train_features[col].values)
        vec_len_pub_test = len(pub_test_features[col].values)
        vec_len_test=len(test_features[col].values)

        data = np.concatenate([train_features[col].values.reshape(vec_len, 1),
                               pub_test_features[col].values.reshape(vec_len_pub_test, 1)], axis=0)

        transformer.fit(data)

        train_features[col] = \
            transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]

        pub_test_features[col] = \
            transformer.transform(pub_test_features[col].values.reshape(vec_len_pub_test, 1)).reshape(1, vec_len_pub_test)[
                0]

    def get_selected_index(train_features,test_features,thres=0.8,selected_index=None):


        ###PCA

        # GENES
        n_comp_GENES=872//2
        data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
        data2 = (PCA(n_components=n_comp_GENES, random_state=42).fit_transform(data[GENES]))
        train2 = data2[:train_features.shape[0]];
        test2 = data2[-test_features.shape[0]:]

        train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp_GENES)])
        test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp_GENES)])

        train_features = pd.concat((train_features, train2), axis=1)
        test_features = pd.concat((test_features, test2), axis=1)

        # CELLS
        n_comp_CELLS=100//2
        data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
        data2 = (PCA(n_components=n_comp_CELLS, random_state=42).fit_transform(data[CELLS]))
        train2 = data2[:train_features.shape[0]];
        test2 = data2[-test_features.shape[0]:]

        train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp_CELLS)])
        test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp_CELLS)])

        train_features = pd.concat((train_features, train2), axis=1)
        test_features = pd.concat((test_features, test2), axis=1)


        if selected_index is None:
            ###get the selected feature

            var_thresh = VarianceThreshold(thres)

            data = train_features.append(test_features)
            data_transformed = var_thresh.fit(data.iloc[:, 4:])
            selected_index = var_thresh._get_support_mask().tolist()

            selected_index=[True,True,True,True]+selected_index


            return train_features.columns.values[selected_index]
        else:
            ### get the feature

            return train_features[selected_index],test_features[selected_index]


    selected_deature_names=get_selected_index(train_features,pub_test_features,thres=0.8)


    train_features,test_features=get_selected_index(train_features,pub_test_features,0.8,selected_deature_names)
    print(selected_deature_names)
    print(train_features.shape)

    num_features=train_features.shape[1]-1   #### - sigid
    losscolector=[]
    folds=[0,1,2,3,4,5,6]
    seeds=[40,42]

    n_fold=len(folds)

    model_dicts=[{'name':'resnetlike','func':Complexer},
                 {'name':'densenetlike','func':Denseplexer},
                 {'name':'tablenet','func':Tablenet},
                 ]


    for model_dict in model_dicts:
        # for cur_seed in seeds:

        for cur_seed in seeds:
            seed_everything(cur_seed)

            #### 5 fols split
            features = train_features.copy()
            target_cols = [c for c in labels.columns if c not in ['sig_id']]
            features['fold'] = -1
            Fold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cur_seed)
            for fold, (train_index, test_index) in enumerate(Fold.split(features, labels[target_cols])):
                features['fold'][test_index] = fold

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
                    loss, best_model = trainer.custom_loop()



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
                    model = model_dict['func'](num_features=num_features)
                    model_name = str(model_dict['name'] + str(cur_seed))

                    ###build trainer
                    trainer = Train(model_name=model_name, model=model, train_ds=train_ds, val_ds=val_ds, fold=fold)

                    trainer.reset(best_model)

                    loss, best_model = trainer.custom_loop()


                losscolector.append([loss,best_model])

        avg_loss=0
        for k,loss_and_model in enumerate(losscolector):
            print('fold %d : loss %.5f modelname: %s'%(k,loss_and_model[0],loss_and_model[1]))
            avg_loss+=loss_and_model[0]
        print('simple,average loss is ',avg_loss/(len(losscolector)))


if __name__=='__main__':
    main()