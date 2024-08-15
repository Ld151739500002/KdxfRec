import pandas as pd
import os

# 读取数据
data1 = pd.read_pickle('../dataset/train.pkl')


###################################### HOT 100   #####################################
# 计算vid的计数
countv = data1[['vid']]
vid_counts = countv['vid'].value_counts()
# 获取前100个最常见的vid
top_100 = vid_counts.head(100)
# 归一化处理
top_100_normalized = top_100 / top_100.sum()
# 创建一个新的DataFrame，包含uid和归一化后的count
df_normalized = pd.DataFrame({
    'vid': top_100_normalized.index,
    'count': top_100_normalized.values
})

os.makedirs('../dataset/recall', exist_ok=True)
df_normalized.to_pickle('../dataset/recall/hot100.pkl')

###################################### 重复观看  #####################################
# 找出重复的行
data_s=data1[['uid', 'vid']]
duplicates = data_s[data_s.duplicated(subset=['uid', 'vid'], keep=False)]
# 创建一个新的DataFrame，包含重复的uid和vid
new_df = duplicates.copy().drop_duplicates()
new_df.to_pickle('../dataset/recall/callback.pkl')
