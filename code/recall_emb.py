import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle

import faiss
import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='embedding 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'embedding 召回，mode: {mode}')

def load_embeddings(emb_file):
    """加载官方embedding文件"""
    df_emb = pd.read_csv(emb_file)
    article_emb_map = {}
    # 获取所有embedding列
    emb_cols = [col for col in df_emb.columns if col.startswith('emb_')]
    
    # 确保列按数字顺序排序
    emb_cols = sorted(emb_cols, key=lambda x: int(x.split('_')[1]))
    
    log.debug(f"检测到 {len(emb_cols)} 维embedding")
    
    for _, row in tqdm(df_emb.iterrows(), total=len(df_emb)):
        article_id = row['article_id']
        emb = row[emb_cols].values.astype(np.float32)
        article_emb_map[article_id] = emb
        
    return article_emb_map
   

def build_faiss_index(emb_map):
    """构建FAISS索引（自动检测维度）"""
    if not emb_map:
        log.warning("embedding映射为空！")
        return None, []
    
    # 动态获取维度
    sample_emb = next(iter(emb_map.values()))
    dim = sample_emb.shape[0]
    log.debug(f"FAISS索引维度: {dim}")
    
    article_ids = list(emb_map.keys())
    embeddings = np.array([emb_map[aid] for aid in article_ids])
    
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index, article_ids
   

@multitasking.task
def recall(df_query, emb_map, faiss_index, article_id_map, user_item_dict, worker_id):
    data_list = []
    
    # 将article_id映射到索引位置
    id_to_index = {aid: idx for idx, aid in enumerate(article_id_map)}
    
    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(float)
        
        if user_id not in user_item_dict:
            continue
            
        # 获取用户最后点击的文章
        last_item = user_item_dict[user_id][-1]
        
        if last_item not in emb_map:
            continue
            
        # 获取embedding向量
        query_emb = emb_map[last_item].reshape(1, -1).astype(np.float32)
        
        # FAISS搜索最近邻
        distances, indices = faiss_index.search(query_emb, 100)
        
        # 处理结果
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:  # FAISS可能返回-1
                continue
                
            relate_item = article_id_map[idx]
            
            # 排除用户已点击的文章
            if relate_item not in user_item_dict[user_id]:
                rank[relate_item] = max(rank.get(relate_item, 0), score)
        
        # 排序并保留Top50
        sim_items = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:50]
        item_ids = [item[0] for item in sim_items]
        item_scores = [item[1] for item in sim_items]
        
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_scores
        df_temp['user_id'] = user_id
        
        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1
        
        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')
        
        data_list.append(df_temp)
    
    df_data = pd.concat(data_list, sort=False)
    os.makedirs('../user_data/tmp/emb', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/emb/{worker_id}.pkl')

if __name__ == '__main__':
    # 加载官方embedding
    emb_file = '../tcdata/articles_emb.csv'
    article_emb_map = load_embeddings(emb_file)
    faiss_index, article_id_map = build_faiss_index(article_emb_map)
    
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        
        os.makedirs('../user_data/data/offline', exist_ok=True)
        emb_pkl_file = '../user_data/data/offline/article_emb.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
        
        os.makedirs('../user_data/data/online', exist_ok=True)
        emb_pkl_file = '../user_data/data/online/article_emb.pkl'
    
    # 保存embedding映射
    with open(emb_pkl_file, 'wb') as f:
        pickle.dump(article_emb_map, f)
    
    log.debug(f'Loaded {len(article_emb_map)} embeddings')
    
    # 构建用户历史字典
    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))
    
    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split
    
    # 清空临时文件夹
    tmp_dir = '../user_data/tmp/emb'
    os.makedirs(tmp_dir, exist_ok=True)
    for file_name in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, file_name))
    
    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_emb_map, faiss_index, article_id_map, user_item_dict, i)
    
    multitasking.wait_for_tasks()
    log.info('合并召回任务')
    
    df_data = pd.DataFrame()
    for path, _, file_list in os.walk(tmp_dir):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp])
    
    # 排序
    df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)
    
    # 计算召回指标
    if mode == 'valid':
        total_users = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        metrics = evaluate(df_data[df_data['label'].notnull()], total_users)
        log.debug(f'Embedding召回指标: {metrics}')
    
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_emb.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_emb.pkl')
