# @Time     : Jan. 10, 2019 15:15
# @Author   : Veritas YIN
# @FileName : math_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import numpy as np


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    '''
    mask=np.not_equal(v,0)
    mask=mask.astype('float32')
    mask=mask/np.mean(mask)
    mask=np.where(np.isnan(mask), np.zeros_like(mask), mask)
    mape=np.abs(v_-v)/v
    mape=mask*mape
    mape=np.where(np.isnan(mape), np.zeros_like(mape), mape)

    return np.mean(mape)


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    mask=np.not_equal(v,0)
    mask=mask.astype('float32')
    mask=mask/np.mean(mask)
    mask=np.where(np.isnan(mask), np.zeros_like(mask), mask)
    mse=(v_-v) ** 2
    mse=mse*mask
    mse=np.where(np.isnan(mse), np.zeros_like(mse), mse)
    return np.sqrt(np.mean(mse))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    mask=np.not_equal(v,0)
    mask=mask.astype('float32')
    mask=mask/np.mean(mask)
    mask=np.where(np.isnan(mask), np.zeros_like(mask), mask)
    mae=np.abs(v_ - v)
    mae=mae*mask
    mae=np.where(np.isnan(mae), np.zeros_like(mae), mae)
    return np.mean(mae)


def evaluation(y, y_, x_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)
    # print('y:{},y_:{}'.format(y.shape,y_.shape)) # y:(3, 1340, 228),y_:(3, 1340, 228, 1)
    if dim == 3:
        v = y
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y=np.expand_dims(y,axis=-1)
        #y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)
