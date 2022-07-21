# @Time     : Jan. 10, 2019 15:26
# @Author   : Veritas YIN
# @FileName : data_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from utils.math_utils import z_score

import numpy as np
import pandas as pd
import os
import pickle


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq


def data_gen(file_path, data_config, n_route, n_frame=24, day_slot=288):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :return: dict, dataset that contains training, validation and test with stats.
    '''
    n_train, n_val, n_test = data_config
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path, header=None).values #(44*288, 325)
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot) # [len_seq, n_frame, n_route, C_0]
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'x_train': x_train, 'x_val': x_val, 'x_test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False, dataset='PeMS-M', n_his=12):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    if isinstance(inputs, (list, tuple)):
        len_inputs = len(inputs[0])
    else:
        len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)
        
        if dataset=='PeMSD7':
            yield inputs[slide][:,:n_his,:,:], inputs[slide][:,n_his:,:,:]
        elif dataset=='PEMS-BAY' or dataset=='METR':
            yield inputs[0][slide], inputs[1][slide] # x, y [batch_size, n_frame, n_route, C_0]
        else:
            raise ValueError("Invalid dataset name = %s" %
                     dataset)
    

# for processing of dataset PEMS-BAY & METR
def load_dataset(dataset_dir):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x'][..., 0] #(23974, 12, 207, 2) (speed, time_of_day)
        data['y_' + category] = cat_data['y'][..., 0]
        data['x_' + category] = np.expand_dims(data['x_' + category], axis=-1)
        data['y_' + category] = np.expand_dims(data['y_' + category], axis=-1)
    x_stats = {'mean': np.mean(data['x_train']), 'std': np.std(data['x_train'])}
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = z_score(data['x_' + category], x_stats['mean'], x_stats['std'])

    dataset = Dataset(data, x_stats)
    return dataset


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
