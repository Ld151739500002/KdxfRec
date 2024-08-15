import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os

# 读取数据
df_train_click = pd.read_csv('../data/uid_click_log.csv')
# 将数据中playtime为0的行去掉
df_train_click = df_train_click[df_train_click['playtime'] != 0]

# 获取用户id的list,随机拿到1000个用户做为验证集
uid_list = df_train_click['uid'].drop_duplicates().to_list()
test_uid_list = random.sample(uid_list, 1000)

# 排序并聚类
df_sorted = df_train_click.sort_values(by=['uid', 'date', 'rank'], ascending=[True, True, False])
groups = df_sorted.groupby('uid', sort=False)

# 保存最后一次交互的验证集
valid_query_list = []
# 保存被去掉最后一次交互的验证集
click_list = []

for user_id, g in tqdm(groups):
    if user_id in test_uid_list:
        valid_query = g.tail(1)
        valid_query_list.append(valid_query[['uid', 'vid']])
        train_click = g.head(g.shape[0] - 1)
        click_list.append(train_click)
    else:
        click_list.append(g)

df_train_click = pd.concat(click_list, sort=False)
df_valid_query = pd.concat(valid_query_list, sort=False)

# 创建保存Pickle文件的目录，如果目录已存在则忽略。
os.makedirs('../dataset', exist_ok=True)
# 将点击DataFrame和查询DataFrame保存为Pickle文件。
df_train_click.to_pickle('../dataset/train.pkl')
df_valid_query.to_pickle('../dataset/val.pkl')