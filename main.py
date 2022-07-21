# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)
tf.compat.v1.random.set_random_seed(1234)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=228, help='207 sensors for METR, 228 for PeMSD7, and 325 for PEMS-BAY.') #
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=12, help='9 time steps for PeMSD7, 12 for others.')
parser.add_argument('--batch_size', type=int, default=32, help='if bsz if enough big  use bn instead of ln.')
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='ADAM', help='use adam and RMSProp, SGD')#RMSProp
parser.add_argument('--graph', type=str, default='default', help=" 'default' for PeMSD7, not 'data_loader/adj_mx_bay.pkl' for PEMS-BAY.")
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--dataset', type=str, default='PeMSD7', help='options for PeMSD7 / PEMS-BAY / METR.')

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
dataset = args.dataset

blocks=[[64,64],[64,64]]

# Load wighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(pjoin('./data_loader/PeMSD7/', f'W_{n}.csv'))
else:
    # load customized graph weight matrix
    _, _, adj_mx = load_pickle(args.graph)
    W = np.maximum.reduce([adj_mx, adj_mx.T])


# Calculate graph kernel
L = scaled_laplacian(W)
V,U=np.linalg.eig(L) # U:228*228

tf.compat.v1.add_to_collection(name='initial_spatial_embeddings', value=tf.cast(tf.constant(U), tf.float32))
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n) # [n_route, Ks*n_route]
tf.compat.v1.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

time_embedding = np.eye(n_his)
tf.compat.v1.add_to_collection(name='initial_temporal_embeddings', value=tf.cast(tf.constant(time_embedding), tf.float32))

# Data Preprocessing
if dataset=='PeMSD7':
    data_file = f'V_{n}.csv'
    n_train, n_val, n_test = 34, 5, 5
    inputs = data_gen(pjoin('./data_loader/PeMSD7/', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
    print(f'>> Loading dataset with Mean: {inputs.mean:.2f}, STD: {inputs.std:.2f}')
elif dataset=='PEMS-BAY':
    data_file = 'data_loader/PEMS-BAY'
    inputs = load_dataset(data_file)
    print(f'>> Loading dataset with Mean: {inputs.mean:.2f}, STD: {inputs.std:.2f}')
elif dataset=='METR':
    data_file = 'data_loader/METR-LA'
    inputs = load_dataset(data_file)
    print(f'>> Loading dataset with Mean: {inputs.mean:.2f}, STD: {inputs.std:.2f}')
else:
    raise ValueError("Invalid dataset Name = %s" % dataset)


if __name__ == '__main__':
    print(str(args), flush=True)
    model_train(inputs, blocks, args)
    # model_test(PeMS,args.batch_size, n_his, n_pred, args.inf_mode)
