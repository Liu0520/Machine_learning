from main import *
from train import *


def generate_pred_features(df, seq_len):
    features = []
    user_ids = []
    groups = df.groupby('user_id')
    used_attrs = ['event_type', 'product_id', 'category_id', 'brand', 'price']
    for user_id, group in tqdm(groups):
        if len(group) < seq_len:
            insert_rows = pd.DataFrame([group.iloc[0, :].values] * (seq_len - len(group)), columns=group.columns)
            group = insert_rows.append(group, ignore_index=True)
        elif len(group) > seq_len:
            group = group.iloc[-seq_len:, :]
        seq_features = group[used_attrs].values
        features.append(seq_features)
        user_ids.append(user_id)
    return np.array(features), user_ids


features, user_ids = generate_pred_features(test_df, seq_len)
model = pkl_load('model.pkl')
# 创建模型并加载训练好的圈中
# 进行预测

from copy import deepcopy

preds = []
# 测试集特征(558,15,5)
print(features.shape)

for i, feature in enumerate(features):
    f = deepcopy(feature)
    # 将feature变成（1,75）
    feature = feature.reshape([-1])[np.newaxis, :]
    pred = model.predict(feature)[0]
    print(pred)
    if pred == seq_len:
        # 说明没找到，采取每组最后一个数据，即用户浏览的最后一个商品ID
        pred = f[-1, 1]
    else:
        pred = f[pred, 1]
    preds.append(int(pred))
preds = lbe2.inverse_transform(preds)
# 保存结果
submit = pd.read_csv('data/submit_example.csv')
submit['product_id'] = preds
submit.to_csv('data/submit.csv', index=False)
