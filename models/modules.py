# @Time     : Jan. 12, 2019 17:45
# @Author   : Veritas YIN
# @FileName : layers.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import tensorflow as tf
import six
import math
import numpy as np

def gconv(x, theta, Ks, c_in, c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    kernel = tf.compat.v1.get_collection('graph_kernel')[0]
    n = tf.shape(kernel)[0]
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv


def gconv_withSAt(x, theta, Ks, c_in, c_out, spatial_attention):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param spatial_attention: [B,n_route,n_route]
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    kernel = tf.compat.v1.get_collection('graph_kernel')[0]
    n = tf.shape(kernel)[0]

    kernels = tf.split(kernel, num_or_size_splits=Ks, axis=1)#[L0,L1,L2] list
    kernel_withAtt=[]
    for i in range(Ks):
        tmp = kernels[i]*spatial_attention
        kernel_withAtt.append(tmp)
    kernel = tf.concat(kernel_withAtt,-1)#[B, n, n*Ks]

    
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.transpose(x, [0, 2, 1])# x -> [batch_size, c_in, n_route]
    # x_mul = x_tmp * ker -> [batch_size, c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv


def embedding_postprocessor(input_tensor,
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            embeddings_length=512,
                            dropout_prob=0.1,training=True):
                            
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  if training==False:
      dropout_prob=0.0
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2] 
  # intial_spatial_embeddings
  # initial_spatial_embeddings
  s_ini=tf.compat.v1.get_collection('initial_spatial_embeddings')[0]
  t_ini=tf.compat.v1.get_collection('initial_temporal_embeddings')[0]
  
  output = input_tensor
  if use_position_embeddings:
    assert_op = tf.compat.v1.assert_equal(seq_length,embeddings_length)
    with tf.control_dependencies([assert_op]):
      if position_embedding_name=='spatial_position_embeddings':
          full_position_embeddings = tf.Variable(
              s_ini, dtype = tf.float32, trainable = True, name = position_embedding_name)
      else:
          full_position_embeddings = tf.Variable(
              t_ini, dtype = tf.float32, trainable = True, name = position_embedding_name)
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, embeddings_length]) # [1, 228, 228]
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      position_embeddings=tf.tile(position_embeddings,[batch_size,1,1])
      output =tf.concat([output, position_embeddings],axis=-1)

  return output

def layer_norm(x, scope):
    '''
    Layer normalization function.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, time_step, n_route, channel].
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

    with tf.compat.v1.variable_scope(scope):
        gamma = tf.compat.v1.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.compat.v1.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x


def batch_norm(x, name=None,is_training=True):
  '''
  Batch normalization function only working for TFv1.
  :param x: tensor, [batch_size, time_step, n_route, channel].
  :param name: str, name of the layer.
  :param is_training:boolean, Whether to return the output in training mode (normalized with statistics of the current batch)  
  :return: tensor, [batch_size, time_step, n_route, channel].
  '''
  x_norm = tf.compat.v1.layers.batch_normalization(x, training=is_training, name=name)
  return x_norm


def fully_con_layer(x, n, channel, scope, keep_prob):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, T, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, T, n_route, 1].
    '''
    w1 = tf.compat.v1.get_variable(name=f'w1_{scope}', shape=[1, 1, channel, 512], dtype=tf.float32)
    tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w1))
    b1 = tf.compat.v1.get_variable(name=f'b1_{scope}', initializer=tf.zeros([n, 512]), dtype=tf.float32)
    x1=tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
    # x1=tf.nn.dropout(x1,keep_prob)
    x1=tf.nn.relu(x1)
    # x1=layer_norm(x1,f'fully_layer_norm_{scope}')
    w2 = tf.compat.v1.get_variable(name=f'w2_{scope}', shape=[1, 1, 512, 1], dtype=tf.float32)
    tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w2))
    b2 = tf.compat.v1.get_variable(name=f'b2_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2


def output_layer(x, n_pred, scope,keep_prob,is_training=True):
    '''
        Output layer: temporal convolution layers attach with one fully connected layer,
        which map outputs of the last st_conv block to a single-step prediction.
        :param x: tensor, [batch_size, time_step, n_route, channel].
        :param T: int, kernel size of temporal convolution.
        :param scope: str, variable scope.
        :param act_func: str, activation function.
        :return: tensor, [batch_size, T, n_route, 1].
        '''
    B, T, n, channel = x.get_shape().as_list() # [batch_size, time_step, n_route, c_out]

    x = tf.nn.relu(x) # unlinearly activate option

    # method2: first using fully-connected then with a temporal dim conv and mapping multi-channels to one
    with tf.compat.v1.variable_scope(f'output_layer_{scope}'):
      w = tf.compat.v1.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 512], dtype=tf.float32)
      tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
      b = tf.compat.v1.get_variable(name=f'b_{scope}', initializer=tf.zeros([512]), dtype=tf.float32)
      x=tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
      
      x=tf.nn.relu(x)

    # x [B,T,n,c] NCHW
    
    with tf.compat.v1.variable_scope('time_step_align'):
      w1 = tf.compat.v1.get_variable(name=f'w1_{scope}', shape=[1, 1, T, n_pred], dtype=tf.float32)
      tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w1))
      b1 = tf.compat.v1.get_variable(name=f'b1_{scope}', initializer=tf.zeros([n_pred]), dtype=tf.float32)
      x_i=tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME',data_format= "NCHW")

      x_i=tf.nn.bias_add(x_i, b1, data_format='NC...' )
      x_i=tf.nn.relu(x_i)

    with tf.compat.v1.variable_scope('multi_channel_to_one'):
      w = tf.compat.v1.get_variable(name=f'w_{scope}', shape=[1, 1, 512, 1], dtype=tf.float32)
      tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
      b = tf.compat.v1.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
      x_i=tf.nn.conv2d(x_i, w, strides=[1, 1, 1, 1], padding='SAME',data_format= "NHWC") + b
    
    return x_i

    # method1: maps multi-channels to one using `fully_con_layer`
    # x_fc = fully_con_layer(x, n, channel, scope,keep_prob)
    # return x_fc


def spatial_encoder_layer(x,Ks,c_out,is_training=True):
    _, T, n, C = x.get_shape().as_list() # C=c_in+n
    if C != c_out:
            # bottleneck down-sampling
            w_input = tf.compat.v1.get_variable('ws_input', shape=[1, 1, C, c_out], dtype=tf.float32)
            tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
            x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    #elif C < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        #x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - C])], axis=3)
    else:
        x_input = x
    
    x=tf.reshape(x,[-1,n,C])
    
    s_conv, spatial_att=encoder_layer(x,attention_mask=None,
                                      hidden_size=c_out,
                                      num_hidden_layers=2,
                                      num_attention_heads=4,
                                      intermediate_size=128,
                                      intermediate_act_fn=tf.nn.relu,
                                      hidden_dropout_prob=0.1,
                                      do_return_scores=True,
                                      initializer_range=0.02,do_return_all_layers=False,is_training=is_training)
    s_conv=tf.reshape(s_conv,[-1,T,n,c_out])

    # define graph convolution kernel using Chebyshev Poly approximation
    ws = tf.compat.v1.get_variable(name='ws', shape=[Ks * C, c_out], dtype=tf.float32)
    tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')
    bs = tf.compat.v1.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)

    # graph convolution with spatial correlation matrix S
    gconv_SAtt = gconv_withSAt(tf.reshape(x, [-1, n, C]), ws, Ks, C,  c_out, spatial_att) + bs
    gconv_SAtt=tf.reshape(gconv_SAtt,[-1,T,n,c_out])
    
    return tf.nn.relu(gconv_SAtt+x_input)


def temporal_encoder_layer(x,c_out,is_training=True):
    _, T, n, C = x.get_shape().as_list()
    
    if C > c_out:
        # bottleneck down-sampling
        w_input = tf.compat.v1.get_variable('ws_input', shape=[1, 1, C, c_out], dtype=tf.float32)
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif C < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - C])], axis=3)
    else:
        x_input = x
    x=tf.transpose(x,[0,2,1,3]) # [B, n, T, C]
    x=tf.reshape(x,[-1,T,C]) # [B*n,T,C]
    with tf.compat.v1.variable_scope('encoder'):
        t_conv,_=encoder_layer(x,attention_mask=None,hidden_size=c_out,num_hidden_layers=2,num_attention_heads=4,intermediate_size=128,intermediate_act_fn=tf.nn.relu,hidden_dropout_prob=0.1,initializer_range=0.02,do_return_all_layers=False,is_training=is_training)
    t_conv=tf.reshape(t_conv,[-1,n,T,c_out])
    t_conv=tf.transpose(t_conv,[0,2,1,3])
  
    return tf.nn.relu(t_conv+x_input)


def variable_summaries(var, v_name):
    '''
    Attach summaries to a Tensor (for TensorBoard visualization).
    Ref: https://zhuanlan.zhihu.com/p/33178205
    :param var: tf.Variable().
    :param v_name: str, name of the variable.
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar(f'mean_{v_name}', mean)

        with tf.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.compat.v1.summary.scalar(f'stddev_{v_name}', stddev)

        tf.compat.v1.summary.scalar(f'max_{v_name}', tf.reduce_max(var))
        tf.compat.v1.summary.scalar(f'min_{v_name}', tf.reduce_min(var))

        tf.compat.v1.summary.histogram(f'histogram_{v_name}', var)

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    do_return_scores=False,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,training=True):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-width vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    #print('input_tensor:{}'.format(input_tensor))
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3]) #[batch_size,input_width]
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
  # print('from_tensor:{},to_tensor:{}'.format(from_tensor,to_tensor))
  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`
  
  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor) #[batch_size * from_seq_length, num_attention_heads * size_per_head]

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d, # input tensor
      num_attention_heads * size_per_head, # output dims
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))
  
  # only needed for calculation spatial correlation matrix
  if do_return_scores:
    q = tf.reshape(query_layer, [-1, from_seq_length, num_attention_heads * size_per_head])
    k = tf.reshape(key_layer, [-1, to_seq_length, num_attention_heads * size_per_head])
    att_mat = tf.matmul(q, k, transpose_b=True)
    att_mat = tf.multiply(att_mat, 1.0 / math.sqrt(float(num_attention_heads * size_per_head)))
    att_mat = tf.nn.softmax(att_mat) # [B, F, T]
  else:
    att_mat = None

  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  # att_mat = tf.reduce_mean(attention_probs, 1)
  return context_layer, att_mat # [B, F, T]


def encoder_layer(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=2048,
                      intermediate_act_fn='relu',
                      hidden_dropout_prob=0.1,
                      do_return_scores=False,
                      initializer_range=0.02,
                      do_return_all_layers=False,is_training=True):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if is_training==False:
     hidden_dropout_prob=0
     attention_probs_dropout_prob=0
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]  
    
  
  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  #prev_output = reshape_to_matrix(x_input)
  prev_output=reshape_to_matrix(input_tensor) # [-1, width]
  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.compat.v1.variable_scope("layer_%d" % layer_idx):      
      B1,C1=prev_output.get_shape().as_list()
      
      layer_input =tf.expand_dims(prev_output,axis=1)
      layer_input=tf.expand_dims(layer_input,axis=1)
      with tf.compat.v1.variable_scope('transformer'):
        if C1>hidden_size:
          w_input=tf.compat.v1.get_variable('ws_input',shape=[1,1,C1,hidden_size],dtype=tf.float32)
          tf.compat.v1.add_to_collection(name='weight_decay',value=tf.nn.l2_loss(w_input))
          layer_input=tf.nn.conv2d(layer_input,w_input,strides=[1,1,1,1],padding='SAME')
        elif C1<hidden_size:
          layer_input=tf.concat([layer_input,tf.zeros([batch_size*seq_length,1,1,hidden_size-C1])],axis=3)
        else:
          layer_input=prev_output
        layer_input=tf.reshape(layer_input,[-1,hidden_size])

      with tf.compat.v1.variable_scope("attention"):
        attention_heads = []
        with tf.compat.v1.variable_scope("self"):
          attention_head, att_mat = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              do_return_scores=do_return_scores,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length,training=is_training)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)


        attention_output = dropout(attention_output, hidden_dropout_prob)
        attention_output = layer_norm_atten(layer_input + attention_output)
        residual = attention_output

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        # with tf.compat.v1.variable_scope("output1"):
        #   attention_output = tf.layers.dense(
        #       attention_output,
        #       hidden_size,
        #       kernel_initializer=create_initializer(initializer_range))
        #   attention_output = dropout(attention_output, hidden_dropout_prob)
        #   attention_output = layer_norm_atten(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.compat.v1.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))
        intermediate_output = dropout(intermediate_output, hidden_dropout_prob)

      # Down-project back to `hidden_size` then add the residual.
      with tf.compat.v1.variable_scope("output2"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm_atten(layer_output + residual)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output, att_mat


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.compat.v1.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output

def layer_norm_atten(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def decoder_cross_layer(target_tensor,input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=2048,
                      intermediate_act_fn='relu',
                      hidden_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,is_training=True):
 
  if is_training==False:
     hidden_dropout_prob=0
     
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2] 

  from_shape = get_shape_list(target_tensor, expected_rank=3)
  from_seq_length = from_shape[1]
  
  prev_output=reshape_to_matrix(input_tensor) #[-1,width]
  target_tensor=reshape_to_matrix(target_tensor)
  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers): # best for 2
    with tf.compat.v1.variable_scope("layer_%d" % layer_idx):      
      B1,C1=prev_output.get_shape().as_list()
      print('B1:{},C1:{}'.format(B1,C1))
      layer_input =tf.expand_dims(prev_output,axis=1)
      layer_input=tf.expand_dims(layer_input,axis=1)
      #B1,T1,C1=layer_input.get_shape().as_list()
      with tf.compat.v1.variable_scope('transformer'):
        if C1>hidden_size:
          w_input=tf.compat.v1.get_variable('ws_input',shape=[1,1,C1,hidden_size],dtype=tf.float32)
          tf.compat.v1.add_to_collection(name='weight_decay',value=tf.nn.l2_loss(w_input))
          layer_input=tf.nn.conv2d(layer_input,w_input,strides=[1,1,1,1],padding='SAME')
        elif C1<hidden_size:
          layer_input=tf.concat([layer_input,tf.zeros([batch_size*seq_length,1,1,hidden_size-C1])],axis=3)
        else:
          layer_input=prev_output
        layer_input=tf.reshape(layer_input,[-1,hidden_size])

      with tf.compat.v1.variable_scope("cross-attention"):
        attention_heads = []
        with tf.compat.v1.variable_scope("self"):
          attention_head,_ = attention_layer( # [B*F, N*H]
              from_tensor=target_tensor, #  q
              to_tensor=layer_input, # k, v
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=from_seq_length,
              to_seq_length=seq_length,training=is_training)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

      
        attention_output = dropout(attention_output, hidden_dropout_prob)
        attention_output = layer_norm_atten(layer_input + attention_output)
        residual = attention_output

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        # with tf.compat.v1.variable_scope("output"):
        #   attention_output = tf.layers.dense(
        #       attention_output,
        #       hidden_size,
        #       kernel_initializer=create_initializer(initializer_range))
        #   attention_output = dropout(attention_output, hidden_dropout_prob)
        #   attention_output = layer_norm_atten(attention_output)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.compat.v1.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))
        intermediate_output = dropout(intermediate_output, hidden_dropout_prob)
      # Down-project back to `hidden_size` then add the residual.
      with tf.compat.v1.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm_atten(residual+layer_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output


def sttn_cross_block(input, Ks, channels, scope, keep_prob,is_training=True):
    '''
        Spatio-temporal convolutional block, which contains two temporal gated convolution layers
        and one spatial graph convolution layer in the middle.
        :param x: tensor, [batch_size, time_step, n_route, c_in].
        :param Ks: int, kernel size of spatial convolution.
        :param channels: list, channel configs of a single st_conv block.
        :param scope: str, variable scope.
        :param keep_prob: placeholder, prob of dropout.
        :return: tensor, [batch_size, time_step, n_route, c_out].
        '''
    c_i,c_o = channels  #[64,64]
    x = input
    B,T,n,c_in=input.get_shape().as_list()
    with tf.compat.v1.variable_scope(f'stn_block_{scope}_spatial'):
        with tf.compat.v1.variable_scope('spatial_embeddings'):
            x=tf.reshape(x,[-1,n,c_in])
            x=embedding_postprocessor(x,
                            use_position_embeddings=True,
                            position_embedding_name="spatial_position_embeddings",
                            initializer_range=0.02,
                            embeddings_length=n,
                            dropout_prob=0.1,training=is_training)
            x=tf.reshape(x,[-1,T,n,c_in+n])
        with tf.compat.v1.variable_scope('temporal_embeddings'):
            x=tf.transpose(x,[0,2,1,3])
            x=tf.reshape(x,[-1,T,c_in+n])
            x=embedding_postprocessor(x,
                            use_position_embeddings=True,
                            position_embedding_name="temporal_position_embeddings",
                            initializer_range=0.02,
                            embeddings_length=T,
                            dropout_prob=0.1,training=is_training)
            x=tf.reshape(x,[-1,n,T,c_in+T+n])
            x=tf.transpose(x,[0,2,1,3])
        print('embedded_inputs:{}'.format(x.get_shape()))#[B,T,n,c_i]
        x_s=spatial_encoder_layer(x,Ks,c_i,is_training)
    
    with tf.compat.v1.variable_scope(f'stn_block_{scope}_temporal'):
        with tf.compat.v1.variable_scope('temporal_embedding'):
            #x=tf.reshape(x,[-1,n,c_in])
            x_t=tf.transpose(input,[0,2,1,3]) # [B,T,n,C]->[B,n,T,C]
            x_t=tf.reshape(x_t,[-1,T,c_in]) # [B*n, T,C]
            x_t=embedding_postprocessor(x_t,
                            use_position_embeddings=True,
                            position_embedding_name="temporal_position_embeddings",
                            initializer_range=0.02,
                            embeddings_length=T,
                            dropout_prob=0.1)
        x_t=tf.reshape(x_t,[-1,n,T,c_in+12])
        x_t=tf.transpose(x_t,[0,2,1,3])#[B,T,n,C]
        x_t=temporal_encoder_layer(x_t,c_i,is_training)

    with tf.compat.v1.variable_scope(f'stn_block_{scope}_decoder'):
      x_o = cross_block(x_s,x_t,c_o,is_training)

    x_o = layer_norm(x_o, f'layer_norm_{scope}')
    # x_o = batch_norm(x_o, name=f'batch_norm_{scope}', is_training=is_training)
    return x_o

def cross_block(x_s,x_t,c_out,is_training=True):
    # query: []
    # key & value: []
    B, T, n, C = x_s.get_shape().as_list()

    # spatial decoder: x_s is qurey, and x_t is key and value
    x_s = tf.reshape(x_s, [-1,n,C]) #[B*T,n,C] query
    x_t = tf.reshape(x_t, [-1,n,C]) #[B*T,n,C] key and value
    with tf.compat.v1.variable_scope('spatial_decode'):
      memory1 = decoder_cross_layer( # [B*T, n, C]
                      target_tensor=x_s,
                      input_tensor=x_t,
                      attention_mask=None,
                      hidden_size=c_out,
                      num_hidden_layers=1,
                      num_attention_heads=4,
                      intermediate_size=128,
                      intermediate_act_fn=tf.nn.relu,
                      hidden_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      is_training=is_training)

    # temporal decoder: x_s is key and value, and x_t denotes query
    x_s = tf.reshape(x_s, [-1, T, n, C])
    x_s = tf.reshape(tf.transpose(x_s, [0, 2, 1, 3]), [-1, T, C])
    x_t = tf.reshape(x_t, [-1, T, n, C])
    x_t = tf.reshape(tf.transpose(x_t, [0, 2, 1, 3]), [-1, T, C])
    with tf.compat.v1.variable_scope('temporal_decode'):
      memory2 = decoder_cross_layer( # [B*n, T, C]
                      target_tensor=x_t,
                      input_tensor=x_s,
                      attention_mask=None,
                      hidden_size=c_out,
                      num_hidden_layers=1,
                      num_attention_heads=4,
                      intermediate_size=128,
                      intermediate_act_fn=tf.nn.relu,
                      hidden_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      is_training=is_training)

    memory1 = tf.reshape(memory1, [-1, T, n, C]) #[B, T, n, C]
    memory2 = tf.transpose(tf.reshape(memory2, [-1, n, T, C]), [0, 2, 1, 3])
    # H = memory1 + memory2 # optional: add directly

    f_g = tf.layers.dense(
      memory1,
      c_out,
      name='fg',
      use_bias=False,
      kernel_initializer= create_initializer()
    )

    f_s = tf.layers.dense(
      memory2,
      c_out,
      name='fs',
      use_bias=True,
      kernel_initializer= create_initializer()
    )

    g = tf.nn.sigmoid(f_g+f_s)
    H = tf.add(tf.multiply(g, memory1), tf.multiply(1-g, memory2))
    
    return H
