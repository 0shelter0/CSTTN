# @Time     : Jan. 12, 2019 19:01
# @Author   : Veritas YIN
# @FileName : base_model.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from models.modules import *
from os.path import join as pjoin
import tensorflow as tf

def masked_mae(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds: [B,time_steps,n,1]
    :param labels: [B,time_steps,n,1]
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.math.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.math.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss *mask
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)


def build_model(inputs, labels, x_stats,n_his, n_pred, Ks, blocks, keep_prob,is_training=True):
    '''
        Build the base model.
        :param inputs: placeholder.
        :param labels:placeholder. [B, T, n, 1]
        :param n_his: int, size of historical records for training.
        :param Ks: int, kernel size of spatial convolution.
        :param Kt: int, kernel size of temporal convolution.
        :param blocks: list, channel configs of st_conv blocks.
        :param keep_prob: placeholder.
        '''
    x = inputs[:, 0:n_his, :, :]

    with tf.compat.v1.variable_scope('aggregation'):
        w1 = tf.compat.v1.get_variable(name='w1', shape=[1,1,1, 32], dtype=tf.float32) #kernel: [filter_height, filter_width,in_channels, out_channels]
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w1))
        #variable_summaries(ws, 'theta')
        b1 = tf.compat.v1.get_variable(name='b1', initializer=tf.zeros([32]), dtype=tf.float32)
        x=tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
    
                                     
    for i, channels in enumerate(blocks):
        x = sttn_cross_block(x, Ks, channels, i, keep_prob,is_training)
    
    # Output Layer x:[batch_size, T, n_route, c_out]
    y = output_layer(x, n_pred, 'output_layer',keep_prob,is_training=is_training) #[B,T,n,1]
    y_t=y*x_stats['std']+x_stats['mean']
    
    train_loss = masked_mae(y_t,labels)

    return train_loss, y #[B,T,n,1]

def model_save(sess, global_steps, model_name, save_path='./output/models/'):
    '''
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    :return:
    '''
    saver = tf.compat.v1.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
