from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

seq_len = 15


def generate_features(df , seq_len):
    """
    获取数据时间序列特征
    features: (N, seq_len, 5),
    labels: (N,)
    """
    features = []
    labels = []
    groups = df.groupby('user_id')

    used_attrs = ['event_type', 'product_id', 'category_id', 'brand', 'price']
    for user_id, group in tqdm(groups):
        # 如果用户的浏览数据量小于15，就用之前浏览的数据加入
        if len(group) < seq_len + 1:
            insert_rows = pd.DataFrame([group.iloc[0, :].values] * (seq_len + 1 - len(group)),columns=group.columns)
            group = insert_rows.append(group, ignore_index=True)
        for i in range(0, len(group) - seq_len):
            seq_features = group.iloc[i:i + seq_len, :][used_attrs]
            label = group.iloc[i + seq_len, :]['product_id']
            features.append(seq_features.values.astype(float))
            labels.append(int(label))
    return np.array(features), np.array(labels)


train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

# 填补nan
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# 编码
lbe1 = LabelEncoder()
train_df['event_type'] = lbe1.fit_transform(train_df['event_type'].astype(str))
test_df['event_type'] = lbe1.transform(test_df['event_type'].astype(str))

lbe2 = LabelEncoder()
product_ids = np.hstack((train_df['product_id'], test_df['product_id'])).astype(int)
lbe2.fit_transform(product_ids)
train_df['product_id'] = lbe2.transform(train_df['product_id'])
test_df['product_id'] = lbe2.transform(test_df['product_id'])

lbe3 = LabelEncoder()
train_df['category_id'] = lbe3.fit_transform(train_df['category_id'].astype(int))
test_df['category_id'] = lbe3.transform(test_df['category_id'].astype(int))

lbe4 = LabelEncoder()
train_df['brand'] = lbe4.fit_transform(train_df['brand'].astype(str))
test_df['brand'] = lbe4.transform(test_df['brand'].astype(str))

if __name__ == '__main__':
    # 获取训练数据特征并保存为npy
    train_features, train_labels = generate_features(train_df, seq_len=seq_len)
    np.save('train_features.npy', train_features)
    np.save('train_labels.npy', train_labels)
    # 获取测试数据特征并保存为npy
    test_features, test_labels = generate_features(test_df, seq_len=seq_len)
    np.save('test_features.npy', test_features)
    np.save('test_labels.npy', test_labels)
    print(train_features.shape, train_labels.shape, test_features.shape, test_labels.shape)
