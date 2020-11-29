import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from tqdm import tqdm

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
from lib.core.base_trainer.wide_and_depp import WideAndDeep
from lib.core.base_trainer.mlp import MLP
from lib.core.base_trainer.group_linear import GroupModel
def main():


    train_feature_file = '../lish-moa/train_features.csv'
    target_file = '../lish-moa/train_targets_scored.csv'
    noscore_target = '../lish-moa/train_targets_nonscored.csv'

    train_features = pd.read_csv(train_feature_file)
    labels = pd.read_csv(target_file)
    extra_labels = pd.read_csv(noscore_target)

    test_features = pd.read_csv('../lish-moa/test_features.csv')

    pub_test_features=test_features.copy()




    variance_threshould=0.8

    # 筛掉方差小于 variance_threshould 的特征
    cols_numeric = [feat for feat in list(train_features.columns) if feat not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
    mask = (train_features[cols_numeric].var() >= variance_threshould).values
    tmp = train_features[cols_numeric].loc[:, mask]
    train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
    #cols_numeric = [feat for feat in list(data_all.columns) if feat not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
    tmp = test_features[cols_numeric].loc[:, mask]
    test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)

    tmp = pub_test_features[cols_numeric].loc[:, mask]
    pub_test_features = pd.concat([pub_test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)


    # ####
    # ####

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]
    ####
    ##RankGauss - transform to Gauss

    def rank_gauss(train_features, pub_features, test_features):
        ##RankGauss - transform to Gauss
        for col in (GENES + CELLS):
            transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
            vec_len = len(train_features[col].values)
            vec_len_pub_test = len(pub_test_features[col].values)
            vec_len_test = len(test_features[col].values)

            data = np.concatenate([train_features[col].values.reshape(vec_len, 1),
                                   pub_features[col].values.reshape(vec_len_pub_test, 1),
                                   ])

            transformer.fit(data)

            train_features[col] = \
                transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]

            pub_features[col] = \
                transformer.transform(pub_features[col].values.reshape(vec_len_pub_test, 1)).reshape(1,
                                                                                                     vec_len_pub_test)[
                    0]

            test_features[col] = \
                transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]

        return train_features, pub_features, test_features

    def pca(train_features, pub_features, test_features):

        # GENES
        n_comp = 600  # <--Update

        data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(pub_features[GENES])])

        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(data[GENES])

        train2 = pca.transform(train_features[GENES])
        pubtest2 = pca.transform(pub_features[GENES])
        test2 = pca.transform(test_features[GENES])

        train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
        pubtest2 = pd.DataFrame(pubtest2, columns=[f'pca_G-{i}' for i in range(n_comp)])
        test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

        # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
        train_features = pd.concat((train_features, train2), axis=1)
        pub_features = pd.concat((pub_features, pubtest2), axis=1)
        test_features = pd.concat((test_features, test2), axis=1)

        # CELLS
        n_comp = 50  # <--Update

        data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(pub_features[CELLS])])

        pca = PCA(n_components=n_comp, random_state=42)

        pca.fit(data[CELLS])

        train2 = pca.transform(train_features[CELLS])
        pubtest2 = pca.transform(pub_features[CELLS])
        test2 = pca.transform(test_features[CELLS])

        train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
        pubtest2 = pd.DataFrame(pubtest2, columns=[f'pca_C-{i}' for i in range(n_comp)])
        test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

        # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
        train_features = pd.concat((train_features, train2), axis=1)
        pub_features = pd.concat((pub_features, pubtest2), axis=1)
        test_features = pd.concat((test_features, test2), axis=1)

        return train_features, pub_features, test_features

    def selection(train_features, pub_features, test_features):
        var_thresh = VarianceThreshold(0.9)  # <-- Update
        data = train_features.append(pub_features)
        var_thresh.fit(data.iloc[:, 4:])

        train_features_transformed = var_thresh.transform(train_features.iloc[:, 4:])
        pubtest_features_transformed = var_thresh.transform(pub_features.iloc[:, 4:])
        test_features_transformed = var_thresh.transform(test_features.iloc[:, 4:])

        train_features = pd.DataFrame(train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                      columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

        train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)

        test_features = pd.DataFrame(test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                     columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

        pub_features = pd.concat([pub_features, pd.DataFrame(pubtest_features_transformed)], axis=1)

        test_features = pd.DataFrame(test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                     columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

        test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

        train_features.shape

        return train_features, pub_features, test_features

    train_features, pub_test_features, test_features = rank_gauss(train_features, pub_test_features, test_features)
    train_features, pub_test_features, test_features = pca(train_features, pub_test_features, test_features)
    train_features, pub_test_features, test_features = selection(train_features, pub_test_features, test_features)

    losscolector=[]
    folds=[0,1,2,3,4,5,6]
    seeds=[0,40,42,10086,1]

    n_fold=len(folds)

    model_dicts=[{'name':'resnetlike','func':Complexer},
                 {'name':'densenetlike','func':Denseplexer},
                 {'name':'tablenet','func':Tablenet},
                 {'name': 'mlp', 'func': MLP},
                 ]

    def run_k_fold(model_dict, seed):
        seed_everything(seed)

        #### 5 fols split
        features = train_features.copy()
        target_cols = [c for c in labels.columns if c not in ['sig_id']]
        features['fold'] = -1
        Fold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cur_seed)
        for fold, (train_index, test_index) in enumerate(Fold.split(features, labels[target_cols])):
            features['fold'][test_index] = fold

        oof = np.zeros((features.shape[0], len(target_cols)))

        for fold in folds:

            logger.info('train with seed %d' % (cur_seed))

            ###build dataset
            train_ind = features[features['fold'] != fold].index.to_list()
            train_features_ = features.iloc[train_ind].copy()
            train_target_ = labels.iloc[train_ind].copy()
            train_extra_Target_ = extra_labels.iloc[train_ind].copy()

            val_ind = features.loc[features['fold'] == fold].index.to_list()
            val_features_ = features.iloc[val_ind].copy()
            val_target_ = labels.iloc[val_ind].copy()
            val_extra_Target_ = extra_labels.iloc[val_ind].copy()



            train_ds = DataIter(train_features_, train_target_, train_extra_Target_, shuffle=True, training_flag=True)
            val_ds = DataIter(val_features_, val_target_, val_extra_Target_, shuffle=False, training_flag=False)

            ### build model

            model = model_dict['func'](936)

            model_name = str(model_dict['name'] + str(cur_seed))

            if cfg.TRAIN.pretrain_on_no_score:
                ###build trainer
                trainer = Train(model_name='PRETRAIN_' + model_name, model=model, train_ds=train_ds, val_ds=val_ds,
                                fold=fold)

                trainer.pretrain = True
                ### pretrian first with no score
                loss, best_model, oof_predict = trainer.custom_loop()

                ### train
                trainer_fine = Train(model_name=model_name, model=model, train_ds=train_ds, val_ds=val_ds, fold=fold)
                trainer_fine.load_from(best_model)
                trainer_fine.pretrain = False
                loss, best_model, oof_predict = trainer_fine.custom_loop()
            else:
                ###build trainer
                trainer = Train(model_name=model_name, model=model, train_ds=train_ds, val_ds=val_ds, fold=fold)

                trainer.pretrain = False
                loss, best_model, oof_predict = trainer.custom_loop()

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
                trainer.pretrain = False

                trainer.reset(best_model)

                loss, best_model, oof_predict = trainer.custom_loop()

            losscolector.append([loss, best_model])
            oof[val_ind]=predict(model,best_model,val_features_)

        return oof
    def predict(model,model_name,feature):
        def preprocess(df):
            """Returns preprocessed data frame"""
            df = df.copy()
            df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
            df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
            df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})
            df = df.drop(['sig_id','fold'], axis=1)
            return df

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_name, map_location=device))
        model.to(device)
        model.eval()


        feature_processed=preprocess(feature.copy())

        feature_processed=torch.from_numpy(feature_processed.values).to(device).float()
        with torch.no_grad():
            predict,__=model(feature_processed)

            predict=torch.nn.functional.sigmoid(predict)

            predict=predict.cpu().numpy()

        predict[feature['cp_type']=='ctl_vehicle']=0

        return predict

    for model_dict in model_dicts:
        # for cur_seed in seeds:

        for cur_seed in seeds:

            if model_dict['name'] == 'tablenet':
                cfg.TRAIN.init_lr = 1.e-2

            oof = np.zeros((train_features.shape[0], 206))

            oof_ = run_k_fold(model_dict, cur_seed)

            oof += oof_ 

            y_true = labels.drop('sig_id', axis=1).values

            y_pred = oof
            score = 0
            for i in range(206):
                score_ = log_loss(y_true[:, i], y_pred[:, i])
                score += score_ / y_pred.shape[1]
            print("%s model CV log_loss:%.6f "%( model_dict['name'],score))

            avg_loss = 0
            for k, loss_and_model in enumerate(losscolector):
                print('fold %d : loss %.5f modelname: %s' % (k, loss_and_model[0], loss_and_model[1]))
                avg_loss += loss_and_model[0]
            print('simple,average loss is ', avg_loss / (len(losscolector)))


if __name__=='__main__':
    main()