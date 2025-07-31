import numpy as np
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import os

input_args = "16_10"

current_script_path = os.path.abspath(__file__)

# 获取当前脚本所在的目录
current_directory = os.path.dirname(current_script_path)

data_path = os.path.join(current_directory, 'Fdc')
# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 假设 config 和 args 已经定义
config = {
    'data_dir': os.path.join(data_path, f'StaNum_{input_args.split("_")[0]}_SNR_{input_args.split("_")[1]}'),
    'val_ratio': 0.4,  # 验证集比例
    'Norm': True,  # 是否进行归一化
    'test_ratio': 0.5
}

args = {
    'data_path': os.path.join(config['data_dir'], f'data_freq_signal_{input_args}.mat'),
    'label_path': os.path.join(config['data_dir'], f'label_winner_subc_{input_args}.mat')
}


def load_data(data_path, label_path):
    X = sio.loadmat(data_path)
    Y = sio.loadmat(label_path)
    data = np.stack((np.real(X['data_freq_signal']), np.imag(X['data_freq_signal'])), axis=-1)
    reverse_data = data[:, ::-1, :]
    label = Y['TrueMinSubc'][:, 0]
    return reverse_data, label


def process_ts_data(data, max_seq_len, normalise=False):
    # 假设 data 是一个三维数组 (样本数, 时间步, 特征数)
    # 这里我们不做任何处理，直接返回数据
    return data


def mean_std(data):
    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))
    return mean, std


def mean_std_transform(data, mean, std):
    return (data - mean) / std


def split_dataset(X, y, val_ratio):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=42)
    return X_train, y_train, X_val, y_val


def split_test_dataset(X, y, test_ratio):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_ratio, random_state=42)
    return X_train, y_train, X_val, y_val


# 加载数据
logger.info("Loading and preprocessing data ...")
data, label = load_data(args['data_path'], args['label_path'])

# 标签编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(label)

# 计算最大序列长度
max_seq_len = data.shape[1]

# 处理数据
X_train = process_ts_data(data, max_seq_len, normalise=False)

# 归一化处理（可选）
if config['Norm']:
    mean, std = mean_std(X_train)
    mean = np.repeat(mean, max_seq_len).reshape(X_train.shape[1], X_train.shape[2])
    std = np.repeat(std, max_seq_len).reshape(X_train.shape[1], X_train.shape[2])
    X_train = mean_std_transform(X_train, mean, std)

# 分割数据集
if config['val_ratio'] > 0:
    train_data, train_label, val_data_, val_label_ = split_dataset(X_train, y_train, config['val_ratio'])
    val_data, val_label, test_data, test_label = split_test_dataset(val_data_, val_label_, config['test_ratio'])
else:
    train_data, train_label, val_data, val_label = X_train, y_train, None, None

# 日志信息
logger.info("{} samples will be used for training".format(len(train_label)))
logger.info("{} samples will be used for validation".format(len(val_label) if val_label is not None else 0))
logger.info("{} samples will be used for testing".format(len(test_label) if val_label is not None else 0))

# 保存数据
Data = {
    'max_len': max_seq_len,
    'All_train_data': X_train,
    'All_train_label': y_train,
    'train_data': train_data,
    'train_label': train_label,
    'val_data': val_data,
    'val_label': val_label,
    'test_data': test_data,
    'test_label': test_label
}

np.save(config['data_dir'] + f"/StaNum_{input_args.split('_')[0]}_SNR_{input_args.split('_')[1]}", Data,
        allow_pickle=True)
logger.info("Data saved successfully.")
