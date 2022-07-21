# @Time     : Jan. 10, 2019 17:52
# @Author   : Veritas YIN
# @FileName : tester.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation, z_inverse
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time


def multi_pred(sess, y_pred, loss, seq, batch_size, step_idx, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param sess: tf.Session().
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
            len_ : int, the length of prediction.
    '''
    pred_list = []
    total_loss = 0
    for (x_data,y_data) in gen_batch(seq, batch_size, dynamic_batch=dynamic_batch):
        pred, val_loss = sess.run([y_pred, loss],#pred [B,T,n,1]
                            feed_dict={'data_input:0': x_data, 'data_label:0': y_data, 'keep_prob:0': 1.0,'is_training:0':False})
        pred_list.append(pred)
        total_loss += val_loss*pred.shape[0]
    #  pred_array -> [n_pred, n_val, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=0)# [n_val, n_pred, n_route, C_0)]
    pred_array = np.swapaxes(pred_array, 0, 1)# [n_pred, n_val, n_route, C_0]
    # print('pred_array:{}'.format(pred_array.shape)) #pred_array:(9, 1340, 228, 1)
    val_loss = total_loss / pred_array.shape[1]
    return pred_array[step_idx], pred_array.shape[1], val_loss


def multi_pred2(sess, y_pred, loss, seq, batch_size, step_idx, dataset=None, dynamic_batch=True):
    pred_list = []
    total_loss = 0 
    for (x_data,y_data) in gen_batch(seq, batch_size, dynamic_batch=dynamic_batch, dataset=dataset):
        pred, val_loss = sess.run([y_pred, loss],#pred [B,T,n,1]
                            feed_dict={'data_input:0': x_data, 'data_label:0': y_data, 'keep_prob:0': 1.0,'is_training:0':False})
        pred_list.append(pred)
        total_loss+=val_loss*pred.shape[0]
    pred_array = np.concatenate(pred_list, axis=0)# [n_val, n_pred, n_route, C_0)]
    pred_array = np.swapaxes(pred_array, 0, 1)# [n_pred, n_val, n_route, C_0]
    # print('pred_array:{}'.format(pred_array.shape)) #pred_array:(9, 1340, 228, 1)
    val_loss = total_loss / pred_array.shape[1]
    return pred_array[step_idx], val_loss


def model_inference(sess, pred, loss, inputs, batch_size, n_his, n_pred, step_idx, dataset='PeMSD7'):
    '''
    Model inference function.
    :param sess: tf.Session().
    :param pred: placeholder.
    :param inputs: instance of class Dataset, data source for inference.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param min_va_val: np.ndarray, metric values on validation set.
    :param min_val: np.ndarray, metric values on test set.
    '''
    x_val, x_test, x_stats = inputs.get_data('x_val'), inputs.get_data('x_test'), inputs.get_stats()
    
    if dataset=='PeMSD7':
        if n_his + n_pred > x_val.shape[1]:
            raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')
        val_pred, len_val, val_loss = multi_pred(sess, pred, loss, x_val, batch_size, step_idx) # y_val:(3, 1340, 228, 1)
        test_pred, len_pred, _ = multi_pred(sess, pred, loss, x_test, batch_size, step_idx)
        val_labels=x_val[0:len_val, step_idx + n_his, :, 0]
        val_labels = z_inverse(val_labels, x_stats['mean'], x_stats['std'])
        test_labels=x_test[0:len_pred, step_idx + n_his, :, 0]
        test_labels = z_inverse(test_labels, x_stats['mean'], x_stats['std'])
    else:
        y_val, y_test = inputs.get_data('y_val'), inputs.get_data('y_test')
        val_pred, val_loss = multi_pred2(sess, pred, loss, (x_val,y_val), batch_size, step_idx, dataset) # y_val:(3, 1340, 228, 1)
        test_pred, _ = multi_pred2(sess, pred, loss, (x_test,y_test), batch_size, step_idx, dataset)
        
        val_labels=y_val[:, step_idx, :, 0]
        test_labels=y_test[:, step_idx, :, 0]
    
    # print('label:{}'.format(labels.shape))# (3, 1340, 228)
    evl_val = evaluation(val_labels, val_pred, x_stats)
    evl_test = evaluation(test_labels, test_pred, x_stats)
    
    return evl_val, evl_test, val_loss


def model_test(inputs, batch_size, n_his, n_pred, inf_mode, load_path='./output/models/'):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    start_time = time.time()
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()
        print('stats:{}'.format(x_stats))
        y_test, len_test = multi_pred(test_sess, pred, x_test, batch_size, n_his,n_pred, step_idx, x_stats)
        np.save('y_groundtruth',x_test[0:len_test, step_idx + n_his, :, :])
        np.save('y_prediction',y_test)
        labels=x_test[0:len_test, step_idx + n_his, :, 0]
        evl = evaluation(labels, y_test, x_stats)

        for ix in tmp_idx:
            te = evl[ix - 2:ix + 1]
            print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
        print(f'Model Test Time {time.time() - start_time:.3f}s')
    print('Testing model finished!')
