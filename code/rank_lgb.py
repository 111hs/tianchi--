import argparse
import gc
import os
import random
import warnings

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import Logger, evaluate, gen_sub

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='lightgbm 排序')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'lightgbm 排序，mode: {mode}')


def train_model(df_feature, df_query):
    df_train = df_feature[df_feature['label'].notnull()]
    df_test = df_feature[df_feature['label'].isnull()]

    del df_feature
    gc.collect()

    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_train.columns))
    # =============== 确保包含新特征 ===============
    # 检查是否包含新增的Embedding特征
    new_features = ['emb_sim_last', 'emb_sim_avg', 'last_emb_norm', 'cand_emb_norm']
    missing_features = [f for f in new_features if f not in feature_names]
    
    if missing_features:
        log.warning(f'缺失新特征: {missing_features}')
    else:
        log.info(f'成功包含所有新特征: {new_features}')
    
    # 确保特征列表正确
    feature_names = [f for f in feature_names if f in df_train.columns]
    # =============== End of 修改 ===============
    feature_names.sort()
    log.info(f'最终特征数量: {len(feature_names)}')
    log.debug(f'特征列表: {feature_names}')

    model = lgb.LGBMClassifier(num_leaves=64,
                               max_depth=10,
                               learning_rate=0.05,
                               n_estimators=10000,
                               subsample=0.8,
                               feature_fraction=0.8,
                               reg_alpha=0.5,
                               reg_lambda=0.5,
                               random_state=seed,
                               importance_type='gain',
                               metric=None)

    oof = []
    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0
    df_importance_list = []

    # 训练模型
    kfold = GroupKFold(n_splits=5)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train[feature_names], df_train[ycol],
                        df_train['user_id'])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][ycol]

        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][ycol]

        log.debug(
            f'\nFold_{fold_id + 1} Training ================================\n'
        )

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=100,
                              eval_metric='auc',
                              early_stopping_rounds=100)

        pred_val = lgb_model.predict_proba(
            X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        df_oof = df_train.iloc[val_idx][['user_id', 'article_id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict_proba(
            df_test[feature_names], num_iteration=lgb_model.best_iteration_)[:,
                                                                             1]
        prediction['pred'] += pred_test / 5

        df_importance = pd.DataFrame({
            'feature_name':
            feature_names,
            'importance':
            lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        joblib.dump(model, f'../user_data/model/lgb{fold_id}.pkl')

    # 特征重要性
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')

    # 生成线下
    df_oof = pd.concat(oof)
    df_oof.sort_values(['user_id', 'pred'],
                       inplace=True,
                       ascending=[True, False])
    log.debug(f'df_oof.head: {df_oof.head()}')

    # 计算相关指标
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)
    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv(f'../prediction_result/result.csv', index=False)


def online_predict(df_test):
    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_test.columns))
    # =============== 确保包含新特征 ===============
    new_features = ['emb_sim_last', 'emb_sim_avg', 'last_emb_norm', 'cand_emb_norm']
    missing_features = [f for f in new_features if f not in feature_names]
    
    if missing_features:
        log.warning(f'缺失新特征: {missing_features}')
    else:
        log.info(f'成功包含所有新特征: {new_features}')
    
    feature_names = [f for f in feature_names if f in df_test.columns]
    # =============== End of 修改 ===============
    feature_names.sort()

    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0

    for fold_id in tqdm(range(5)):
        model = joblib.load(f'../user_data/model/lgb{fold_id}.pkl')
        pred_test = model.predict_proba(df_test[feature_names])[:, 1]
        prediction['pred'] += pred_test / 5

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv(f'../prediction_result/result.csv', index=False)


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/feature.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

        train_model(df_feature, df_query)
    else:
        df_feature = pd.read_pickle('../user_data/data/online/feature.pkl')
        online_predict(df_feature)
