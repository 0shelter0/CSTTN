# @Time     : Jan. 13, 2019 20:16
# @Author   : Veritas YIN
# @FileName : trainer.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from data_loader.data_utils import gen_batch
from models.tester import model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time


def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    x_stats = inputs.get_stats() #  {'mean': self.mean, 'std': self.std}
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt
    dataset_name = args.dataset
    
    x = tf.compat.v1.placeholder(tf.float32, [None, n_his, n, 1], name='data_input')
    label = tf.compat.v1.placeholder(tf.float32, [None, n_pred, n, 1], name='data_label')
    keep_prob=tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    is_training=tf.compat.v1.placeholder(tf.bool, name='is_training')
    # Define model loss
    train_loss, pred = build_model(x, label,x_stats,n_his, n_pred, Ks, blocks, keep_prob) #MAE loss
    tf.compat.v1.summary.scalar('train_loss', train_loss)
    
    weight_loss=0.0001*tf.add_n(tf.compat.v1.get_collection('weight_decay'))
    train_loss_w=train_loss+weight_loss

    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('x_train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.5, staircase=True)
    lr=tf.maximum(1e-5,lr)

    # Learning rate decayed with cosine_decay
    # first_decay_steps = int(epoch_step*2 / 1)
    # lr = tf.compat.v1.train.cosine_decay_restarts(learning_rate=args.lr, global_step=global_steps, first_decay_steps=first_decay_steps)

    tf.compat.v1.summary.scalar('learning_rate', lr)
    step_op = tf.compat.v1.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.compat.v1.train.RMSPropOptimizer(lr).minimize(train_loss_w, global_step = global_steps)
        elif opt == 'ADAM':
            train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(train_loss_w, global_step = global_steps)
        elif opt == 'SGD':
            train_op = tf.compat.v1.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=True).minimize(train_loss_w, global_step = global_steps)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.compat.v1.summary.merge_all()

    with tf.compat.v1.Session() as sess:        
        writer = tf.compat.v1.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')
        best=1e5
        best_epoch=-1
        for i in range(epoch):
            start_time = time.time()
            # total_loss = 0
            for j, (x_batch, y_batch) in enumerate(  # for PeMSD7  need to adjust args for `gen_batch`, add arg `inputs.get_data('y_train')`
                    gen_batch((inputs.get_data('x_train')), batch_size, dynamic_batch=True, shuffle=True, dataset=dataset_name)):
                if dataset_name == 'PeMSD7':
                    y_batch = y_batch*x_stats['std'] + x_stats['mean']
                summary, _ = sess.run([merged, train_op], 
                                    feed_dict={x: x_batch, label:y_batch, keep_prob: 0.9,is_training:True})
                writer.add_summary(summary, i * epoch_step + j)
                if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss_w,train_loss],
                                 feed_dict={x: x_batch, label:y_batch, keep_prob:0.9,is_training:True})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f},{loss_value[1]:.3f}]', flush=True)
                    
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s', flush=True)
            
            
            start_time = time.time()
            min_va_val, min_val, val_loss = \
                model_inference(sess, pred, train_loss, inputs, batch_size, n_his, n_pred, step_idx, dataset_name)

            for ix in tmp_idx:
                va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
                print(f'Time Step {ix + 1}: '
                      f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                      f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                      f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.', flush=True)
            print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s', flush=True)
            
            print('current val_loss:{:.6f}, last best epoch is {}'.format(val_loss, best_epoch))
            if val_loss<=best:
               best=val_loss
               best_epoch=i
               print('best_validation:{:.6f} and current best epoch: {:2d}.'.format(best, best_epoch))
               model_save(sess,global_steps,f'Model-best-{best:.3f}-epoch{i}')            
            #if (i + 1) % args.save == 0:
                #model_save(sess, global_steps, 'STTN-PeMSD7')
            #if (i + 1) % args.save == 0:
                #model_save(sess, global_steps, 'STTN-PeMSD7')
        writer.close()
    print('Training model finished!')

