import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scipy.spatial.distance import cosine
from tqdm import tqdm
from utils import Logger

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pandarallel.initialize()

warnings.filterwarnings('ignore')

seed = 2020

# 命令行参数
parser = argparse.ArgumentParser(description='排序特征')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'排序特征，mode: {mode}')


def func_if_sum(x):
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += item_sim[i][article_id] * (0.7**loc)
        except Exception as e:
            pass
    return sim_sum


def func_if_last(x):
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = item_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim


def func_binetwork_sim_last(x):
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = binetwork_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim


def consine_distance(vector1, vector2):
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray:
        return -1
    distance = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return distance


def func_w2w_sum(x, num):
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1][:num]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += consine_distance(article_vec_map[article_id],
                                        article_vec_map[i])
        except Exception as e:
            pass
    return sim_sum


def func_w2w_last_sim(x):
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = consine_distance(article_vec_map[article_id],
                               article_vec_map[last_item])
    except Exception as e:
        pass
    return sim
# =============== 新增 Embedding 特征函数 ===============
def emb_sim_features(x, emb_map):
    """计算embedding相似度特征"""
    user_id = x['user_id']
    article_id = x['article_id']
    
    # 获取用户最后点击的文章
    if user_id in user_item_dict:
        last_item = user_item_dict[user_id][-1]
    else:
        return 0.0
    
    # 获取embedding
    emb1 = emb_map.get(last_item)
    emb2 = emb_map.get(article_id)
    
    if emb1 is None or emb2 is None:
        return 0.0
    
    # 计算余弦相似度
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def emb_sim_avg(x, emb_map, user_emb_map):
    """计算与用户平均embedding的相似度"""
    user_id = x['user_id']
    article_id = x['article_id']
    
    avg_emb = user_emb_map.get(user_id)
    cand_emb = emb_map.get(article_id)
    
    if avg_emb is None or cand_emb is None:
        return 0.0
    
    dot_product = np.dot(avg_emb, cand_emb)
    norm1 = np.linalg.norm(avg_emb)
    norm2 = np.linalg.norm(cand_emb)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def emb_norm_features(x, emb_map, emb_type):
    """计算embedding范数特征"""
    if emb_type == 'last':
        user_id = x['user_id']
        if user_id in user_item_dict:
            last_item = user_item_dict[user_id][-1]
            emb = emb_map.get(last_item)
            return np.linalg.norm(emb) if emb is not None else 0.0
        return 0.0
    elif emb_type == 'candidate':
        article_id = x['article_id']
        emb = emb_map.get(article_id)
        return np.linalg.norm(emb) if emb is not None else 0.0
    return 0.0

if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')

    else:
        df_feature = pd.read_pickle('../user_data/data/online/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')

    # 文章特征
    log.debug(f'df_feature.shape: {df_feature.shape}')

    df_article = pd.read_csv('../tcdata/articles.csv')
    df_article['created_at_ts'] = df_article['created_at_ts'] / 1000
    df_article['created_at_ts'] = df_article['created_at_ts'].astype('int')
    df_feature = df_feature.merge(df_article, how='left')
    df_feature['created_at_datetime'] = pd.to_datetime(
        df_feature['created_at_ts'], unit='s')

    log.debug(f'df_article.head(): {df_article.head()}')
    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 历史记录相关特征
    df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)
    df_click.rename(columns={'click_article_id': 'article_id'}, inplace=True)
    df_click = df_click.merge(df_article, how='left')

    df_click['click_timestamp'] = df_click['click_timestamp'] / 1000
    df_click['click_datetime'] = pd.to_datetime(df_click['click_timestamp'],
                                                unit='s',
                                                errors='coerce')
    df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour

    # 用户点击文章的创建时间差的平均值
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby(
        ['user_id'])['created_at_ts'].diff()
    df_temp = df_click.groupby([
        'user_id'
    ])['user_id_click_article_created_at_ts_diff'].mean().reset_index()
    df_temp.columns = [
        'user_id', 'user_id_click_article_created_at_ts_diff_mean'
    ]
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 用户点击文章的时间差的平均值
    df_click['user_id_click_diff'] = df_click.groupby(
        ['user_id'])['click_timestamp'].diff()
    df_temp = df_click.groupby(['user_id'
                                ])['user_id_click_diff'].mean().reset_index()
    df_temp.columns = ['user_id', 'user_id_click_diff_mean']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_click['click_timestamp_created_at_ts_diff'] = df_click[
        'click_timestamp'] - df_click['created_at_ts']

    # 点击文章的创建时间差的统计值
    df_temp = df_click.groupby(
        ['user_id'])['click_timestamp_created_at_ts_diff'].agg({
            'user_click_timestamp_created_at_ts_diff_mean':
            'mean',
            'user_click_timestamp_created_at_ts_diff_std':
            'std'
        }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_datetime_hour 统计值
    df_temp = df_click.groupby(['user_id'])['click_datetime_hour'].agg({
        'user_click_datetime_hour_std':
        'std'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 words_count 统计值
    df_temp = df_click.groupby(['user_id'])['words_count'].agg({
        'user_clicked_article_words_count_mean':
        'mean',
        'user_click_last_article_words_count':
        lambda x: x.iloc[-1]
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 created_at_ts 统计值
    df_temp = df_click.groupby('user_id')['created_at_ts'].agg({
        'user_click_last_article_created_time':
        lambda x: x.iloc[-1],
        'user_clicked_article_created_time_max':
        'max',
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_timestamp 统计值
    df_temp = df_click.groupby('user_id')['click_timestamp'].agg({
        'user_click_last_article_click_time':
        lambda x: x.iloc[-1],
        'user_clicked_article_click_time_mean':
        'mean',
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_feature['user_last_click_created_at_ts_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_created_time']
    df_feature['user_last_click_timestamp_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_click_time']
    df_feature['user_last_click_words_count_diff'] = df_feature[
        'words_count'] - df_feature['user_click_last_article_words_count']

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 计数统计
    for f in [['user_id'], ['article_id'], ['user_id', 'category_id']]:
        df_temp = df_click.groupby(f).size().reset_index()
        df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]

        df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 召回相关特征
    ## itemcf 相关
    user_item_ = df_click.groupby('user_id')['article_id'].agg(
        list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['article_id']))

    if mode == 'valid':
        f = open('../user_data/sim/offline/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()

    # 用户历史点击物品与待预测物品相似度
    df_feature['user_clicked_article_itemcf_sim_sum'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_if_sum, axis=1)
    df_feature['user_last_click_article_itemcf_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_if_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## binetwork 相关
    if mode == 'valid':
        f = open('../user_data/sim/offline/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()

    df_feature['user_last_click_article_binetwork_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_binetwork_sim_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## w2v 相关
    if mode == 'valid':
        f = open('../user_data/data/offline/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/data/online/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()

    df_feature['user_last_click_article_w2v_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_w2w_last_sim, axis=1)
    df_feature['user_click_article_w2w_sim_sum_2'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(lambda x: func_w2w_sum(x, 2), axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # =============== 新增 Embedding 特征 ===============
    # 加载官方embedding
    if mode == 'valid':
        emb_file = '../user_data/data/offline/article_emb.pkl'
    else:
        emb_file = '../user_data/data/online/article_emb.pkl'
    
    with open(emb_file, 'rb') as f:
        article_emb_map = pickle.load(f)
    
    log.debug(f'Loaded {len(article_emb_map)} embeddings for feature engineering')
    # 动态获取embedding维度
    emb_dim = 250  # 默认值
    if article_emb_map:
        sample_emb = next(iter(article_emb_map.values()))
        emb_dim = len(sample_emb)
        log.debug(f"Embedding维度: {emb_dim}")
   
    # 特征1: 与最后点击文章的相似度
    df_feature['emb_sim_last'] = df_feature.parallel_apply(
        lambda x: emb_sim_features(x, article_emb_map), axis=1
    )
    
    # 特征2: 与用户历史平均embedding的相似度
    # 先计算每个用户的平均embedding
    user_avg_emb = {}
    for user_id, items in tqdm(user_item_dict.items()):
        embs = [article_emb_map.get(item) for item in items]
        valid_embs = [e for e in embs if e is not None]
        
        if valid_embs:
            user_avg_emb[user_id] = np.mean(valid_embs, axis=0)
        else:
            user_avg_emb[user_id] = np.zeros(250)  # 假设embedding维度为250
    
    df_feature['emb_sim_avg'] = df_feature.parallel_apply(
        lambda x: emb_sim_avg(x, article_emb_map, user_avg_emb), axis=1
    )
    
    # 特征3: 最后点击文章的embedding范数
    df_feature['last_emb_norm'] = df_feature.parallel_apply(
        lambda x: emb_norm_features(x, article_emb_map, 'last'), axis=1
    )
    
    # 特征4: 当前文章的embedding范数
    df_feature['cand_emb_norm'] = df_feature.parallel_apply(
        lambda x: emb_norm_features(x, article_emb_map, 'candidate'), axis=1
    )
    
    log.debug(f'新增Embedding特征完成，特征维度: {df_feature.shape[1]}')
    log.debug(f'Embedding特征示例: {df_feature[["emb_sim_last", "emb_sim_avg", "last_emb_norm", "cand_emb_norm"]].head()}')
    # =============== End of Embedding 特征 ===============

    # 保存特征文件
    if mode == 'valid':
        df_feature.to_pickle('../user_data/data/offline/feature.pkl')

    else:
        df_feature.to_pickle('../user_data/data/online/feature.pkl')
