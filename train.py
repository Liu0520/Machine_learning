import numpy as np
import lightgbm as lgb
import pickle

seq_len = 15

train_features, train_labels = np.load('train_features.npy'), np.load('train_labels.npy')
test_features, test_labels = np.load('test_features.npy'), np.load('test_labels.npy')
# 竖直方向进行了堆叠
features = np.vstack((train_features, test_features))
# 水平方向进行了堆叠
labels = np.hstack((train_labels, test_labels))

# 将标签进行转换
features_, labels_ = [], []
# features(181027, 15, 5)   labels(181027,)
for f, l in zip(features, labels):
    # 如果商品ID为876，且在前面1-15个数据里第一个商品ID也为876的话，那么返回值为0，
    # 如果这一组数据里都没有该商品ID就返回15
    # 查找商品ID,并返回一个0-15的值
    idx = np.where(f[:, 1] == l)[0]
    if len(idx) == 0:
        labels_.append(seq_len)
    else:
        labels_.append(np.random.choice(idx))
labels = np.array(labels_)
class_weight = {}
for i in np.unique(labels):
    if i != seq_len:
        class_weight[i] = 1.
    else:
        class_weight[i] = 1 / 3
print(labels)
print(features.shape, labels.shape)

from collections import Counter

print(Counter(labels))
print(features.shape)


# 三维变二维，合并后面的5x15,变为(181027,75)
features = features.reshape([len(features), -1])
print(features.shape)

def pkl_save(filename, file):
    output = open(filename, 'wb')
    pickle.dump(file, output)
    output.close()


def pkl_load(filename):
    pkl_file = open(filename, 'rb')
    file = pickle.load(pkl_file)
    pkl_file.close()
    return file


if __name__ == '__main__':
    model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=32,
        max_depth=100,
        learning_rate=0.05,
        n_estimators=1500,
        class_weight=class_weight,
    )
    print('Training ...')
    model.fit(features, labels,)
    # 多分类的对数损失
    y_pred = model.predict(features)
    # 平均精度得分
    acc = np.mean(y_pred == labels)
    print(acc)#0.9023
    pkl_save('model.pkl', model)
