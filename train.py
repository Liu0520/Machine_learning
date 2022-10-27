import numpy as np
from torch import nn
import lightgbm as lgb

seq_len = 15
input_size = 5
hidden_size = 64
num_layers = 3
dropout = 0.5
# per_iter = 100

num_epochs = 500
batch_size = 512
lr = 1e-4


class Model(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=5, num_layers=3, dropout=0.5):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1]
        self.dropout(x)
        out = self.fc(x)
        return out


# 读取已经保存的特征
# features=['event_type','product_id' , 'category_id', 'brand', 'price']
# labels=['product_id']
train_features, train_labels = np.load('train_features.npy'), np.load('train_labels.npy')
test_features, test_labels = np.load('test_features.npy'), np.load('test_labels.npy')
# 竖直方向进行了堆叠
features = np.vstack((train_features, test_features))
# 水平方向进行了堆叠
labels = np.hstack((train_labels, test_labels))

# 将标签进行转换
features_, labels_ = [], []
# features(230552, 25, 5)   labels(230553,)
for f, l in zip(features, labels):
    idx = np.where(f[:, 1] == l)[0]
    # print(f[:,1])
    # 以元组的形式返回
    if len(idx) == 0:
        labels_.append(seq_len)
    else:
        # np.random.choice()从数组中随机抽取元素
        labels_.append(np.random.choice(idx))
labels = np.array(labels_)
print(labels)
print(features.shape, labels.shape)

# 样本平衡
num_samples = len(features)
from collections import Counter

print(Counter(labels))
print(features.shape, Counter(labels))
# features = features.reshape([len(features), seq_len, 5])

# 将列变成行
features = features.reshape([len(features), -1])
import pickle


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
        max_depth=100,
        n_estimators=1000,
    )

    print('Training ...')
    model.fit(features, labels, eval_set=[(features, labels)], verbose=True, )

    y_pred = model.predict(features)
    acc = np.mean(y_pred == labels)
    print(acc)

    pkl_save('model.pkl', model)
