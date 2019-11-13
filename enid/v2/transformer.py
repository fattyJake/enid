# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Tranformer (Attention is all you need, https://arxiv.org/abs/1706.03762) 
# Authors:     Yage Wang
# Created:     5.29.2019
###############################################################################

import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi head attention.
    1.linearly project the queries, keys and values h times(with different, learned linear projections to d_k, d_k, d_v dimensions)
    2.scaled dot product attention for each projected version of Q,K,V
    3.concatenated result
    4.linear projection to get final result
    """

    def __init__(self, layer_index, d_model, sequence_length, h, dropout_prob=0.0, **kwargs):
        super(MultiHeadAttention, self).__init__(name=f'MultiHeadAttention_{layer_index}', **kwargs)

        self.d_model = d_model
        self.sequence_length = sequence_length
        self.h = h
        self.dropout_prob = dropout_prob

        self.Q_proj_layer = tf.keras.layers.Dense(self.d_model, activation='relu', kernel_initializer='he_normal', name="Q")
        self.K_proj_layer = tf.keras.layers.Dense(self.d_model, activation='relu', kernel_initializer='he_normal', name="K")
        self.V_proj_layer = tf.keras.layers.Dense(self.d_model, activation='relu', kernel_initializer='he_normal', name="V")
        self.output_layer = tf.keras.layers.Dense(self.d_model, activation='relu', kernel_initializer='he_normal', name="output")

    @staticmethod
    def masking(input_, queries=None, keys=None, type_=None):
        """Masks paddings on keys or queries to inputs
        inputs: 3d tensor. (N, T_q, T_k)
        queries: 3d tensor. (N, T_q, d)
        keys: 3d tensor. (N, T_k, d)

        e.g.,
        >> queries = tf.constant([[[1.],
                            [2.],
                            [0.]]], tf.float32) # (1, 3, 1)
        >> keys = tf.constant([[[4.],
                         [0.]]], tf.float32)  # (1, 2, 1)
        >> inputs = tf.constant([[[4., 0.],
                                   [8., 0.],
                                   [0., 0.]]], tf.float32)
        >> mask(inputs, queries, keys, "key")
        array([[[ 4.0000000e+00, -4.2949673e+09],
            [ 8.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
        >> inputs = tf.constant([[[1., 0.],
                                 [1., 0.],
                                  [1., 0.]]], tf.float32)
        >> mask(inputs, queries, keys, "query")
        array([[[1., 0.],
            [1., 0.],
            [0., 0.]]], dtype=float32)
        """
        padding_num = -2 ** 32 + 1
        if type_ in ("k", "key", "keys"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(input_tensor=tf.abs(keys), axis=-1))  # (N, T_k)
            masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
            masks = tf.tile(masks, [1, tf.shape(input=queries)[1], 1])  # (N, T_q, T_k)
            # Apply masks to inputs
            paddings = tf.ones_like(input_) * padding_num
            outputs = tf.compat.v1.where(tf.equal(masks, 0), paddings, input_)  # (N, T_q, T_k)
        elif type_ in ("q", "query", "queries"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(input_tensor=tf.abs(queries), axis=-1))  # (N, T_q)
            masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
            masks = tf.tile(masks, [1, 1, tf.shape(input=keys)[1]])  # (N, T_q, T_k)
            # Apply masks to inputs
            outputs = input_ * masks
        else:
            raise ValueError("Check if you entered type correctly!")

        return outputs

    def call(self, inputs):
        """
        multi head attention
        :param Q: query.  shape:[batch,sequence_length,d_model]
        :param K_s: keys. shape:[batch,sequence_length,d_model].
        :param V_s:values.shape:[batch,sequence_length,d_model].
        :param h: h times
        :return: result of scaled dot product attention. shape:[sequence_length,d_model]
        """
        # 1. linearly project the queries,keys and values h times(with different,learned linear projections to d_k,d_k,d_v dimensions)
        Q, K_s, V_s = inputs
        Q_projected = self.Q_proj_layer(Q)          # [batch,sequence_length,d_model]
        K_s_projected = self.K_proj_layer(K_s)      # [batch,sequence_length,d_model]
        V_s_projected = self.V_proj_layer(V_s)      # [batch,sequence_length,d_model]

        # 2. scaled dot product attention for each projected version of Q,K,V
        dot_product = self.scaled_dot_product_attention_batch(Q_projected, K_s_projected, V_s_projected)  # [batch*h,sequence_length,d_k]

        # 3. reshape
        dot_product = tf.concat(tf.split(dot_product, self.h, axis=0), axis=2)

        # 4. linear projection
        output = self.output_layer(dot_product)
        return output  # [batch,sequence_length,d_model]
    
    def scaled_dot_product_attention_batch(self, Q, K_s, V_s, scope="scaled_dot_product_attention"):
        """
        scaled dot product attention
        :param Q:  query.  shape:[batch,sequence_length,d_model]
        :param K_s: keys.  shape:[batch,sequence_length,d_model]
        :param V_s:values. shape:[batch,sequence_length,d_model]
        :return: result of scaled dot product attention. shape:[batch,h,sequence_length,d_k]
        """
        # 1. split Q,K,V
        Q_heads = tf.concat(tf.split(Q, self.h, axis=2), axis=0)    # [batch*h,sequence_length,d_k]
        K_heads = tf.concat(tf.split(K_s, self.h, axis=2), axis=0)  # [batch*h,sequence_length,d_k]
        V_heads = tf.concat(tf.split(V_s, self.h, axis=2), axis=0)  # [batch*h,sequence_length,d_k]

        # 2. dot product of Q,K
        dot_product = tf.matmul(Q_heads, K_heads, transpose_b=True)  # [batch*h,sequence_length,sequence_length]
        dot_product = dot_product * (1.0 / tf.sqrt(tf.cast(self.d_model, tf.float32)))  # [batch*h,sequence_length,sequence_length]

        # 3. add mask if it is none
        # if self.mask is not None:
        #     mask_expand = tf.expand_dims(tf.expand_dims(self.mask, axis=0), axis=0) # [1,1,sequence_length,sequence_length]
        #     dot_product = dot_product + mask_expand                                 # [batch,h,sequence_length,sequence_length]
        # key masking
        dot_product = self.masking(dot_product, Q_heads, K_heads, type_="key")

        # 4.get possibility
        weights = tf.nn.softmax(dot_product)  # [batch*h,sequence_length,sequence_length]
        # query masking
        weights = self.masking(weights, Q_heads, K_heads, type_="query")
        weights = tf.nn.dropout(weights, self.dropout_prob)  # [batch*h,sequence_length,sequence_length]

        # 5. final output
        output = tf.matmul(weights, V_heads)  # [batch*h,sequence_length,d_k]
        return output

class FeedFoward(tf.keras.layers.Layer):
    """
    Feed-Forward Networks
    In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
    connected feed-forward network, which is applied to each position separately and identically. This
    consists of two linear transformations with a ELU activation in between.

    FFN(x) = max(0,xW1+b1)W2+b2

    While the linear transformations are the same across different positions, they use different parameters
    from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
    The dimensionality of input and output is d_model= 512, and the inner-layer has dimensionality d_ff= 2048.
    """

    def __init__(self, layer_index, d_model, d_ff, **kwargs):
        """
        :param x: shape should be:[batch,sequence_length,d_model]
        :param layer_index:  index of layer
        :return: shape:[sequence_length,d_model]
        """
        super(FeedFoward, self).__init__(name=f'FeedFoward_{layer_index}', **kwargs)

        self.inner_layer = tf.keras.layers.Dense(d_ff, activation="relu", kernel_initializer='he_normal', name="inner_layer")
        self.outer_layer = tf.keras.layers.Dense(d_model, kernel_initializer='he_normal', name="outer_layer")

    def call(self, inputs):
        """
        x:       [batch,sequence_length,d_model]
        :return: [batch,sequence_length,d_model]
        """
        # Inner layer
        outputs = self.inner_layer(inputs)

        # Outer layer
        outputs = self.outer_layer(outputs)

        return outputs  # [batch,sequence_length,d_model]


class LayerNormResidualConnection(tf.keras.layers.Layer):
    """
    We employ a residual connection around each of the two sub-layers, followed by layer normalization.
    That is, the output of each sub-layer is LayerNorm(x+ Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.
    """

    def __init__(self, layer_index, d_model, mode, dropout_prob=0.1, use_residual_conn=True, **kwargs):
        super(LayerNormResidualConnection, self).__init__(name=f'LayerNormResidualConnection_{mode}_{layer_index}', **kwargs)

        self.layer_index = layer_index
        self.dropout_prob = dropout_prob
        self.use_residual_conn = use_residual_conn
        self.norm_scale = tf.Variable(
            tf.ones([d_model]), name="layer_norm_scale",
        )  # [filter]
        self.norm_bias = tf.Variable(
            tf.zeros([d_model]), name="layer_norm_bias",
        )  # [filter]
        # self.batch_norm_layer = tf.keras.layers.BatchNormalization()

    def _layer_normalization(self, x):
        """
        x should be:[batch_size,sequence_length,d_model]
        :return:
        """
        # 1. normalize input by using  mean and variance according to las
        # dimension
        mean = tf.reduce_mean(
            x, axis=-1, keepdims=True
        )  # [batch_size,sequence_length,1]
        variance = tf.reduce_mean(
            tf.square(x - mean), axis=-1, keepdims=True
        )  # [batch_size,sequence_length,1]
        norm_x = (x - mean) * tf.sqrt(
            variance + 1e-6
        )  # [batch_size,sequence_length,d_model]

        # 2. re-scale normalized input back
        output = (
            norm_x * self.norm_scale + self.norm_bias
        )  # [batch_size,sequence_length,d_model]
        return output  # [batch_size,sequence_length,d_model]

    # call residual connection and layer normalization
    def call(self, inputs):
        x, y = inputs
        if self.use_residual_conn:
            x_residual = x + tf.nn.dropout(y, self.dropout_prob)
            x_layer_norm = self._layer_normalization(x_residual) # self.batch_norm_layer(x_residual, training=False)
        else:
            x_layer_norm = self._layer_normalization(x) # self.batch_norm_layer(x, training=False)
        return x_layer_norm

class Encoder(tf.keras.layers.Layer):
    """
    base class has some common fields and functions.
    """

    def __init__(self, d_model, d_ff, sequence_length, h, batch_size, num_layer=6, dropout_prob=None, use_residual_conn=True, **kwargs):
        super(Encoder, self).__init__(name=f'Encoder', **kwargs)

        self.d_model = d_model
        self.d_ff = d_ff
        self.sequence_length = sequence_length
        self.h = h
        self.num_layer = num_layer
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.use_residual_conn = use_residual_conn
        
        self.V_s = [self.add_weight(name=f'V_{layer_index}', shape=[self.batch_size, self.sequence_length, self.d_model], initializer='he_normal') for layer_index in range(self.num_layer)]
        self.encode_stack = [{"MultiHeadAttention": MultiHeadAttention(layer_index, self.d_model, self.sequence_length, self.h, dropout_prob=dropout_prob),
                              "MHA_Resid": LayerNormResidualConnection(layer_index, self.d_model, "MHA_Resid", dropout_prob=dropout_prob, use_residual_conn=use_residual_conn),
                              "FeedForward": FeedFoward(layer_index, self.d_model, self.d_ff),
                              "FF_Resid": LayerNormResidualConnection(layer_index, self.d_model, 'FF_Resid', dropout_prob=dropout_prob, use_residual_conn=use_residual_conn)}
                              for layer_index in range(self.num_layer)]

    def call(self, inputs):
        Q, K_s = inputs
        for layer_index in range(self.num_layer):
            Q, K_s = self.encoder_single_layer(Q, K_s, layer_index)
        return Q, K_s
    
    def encoder_single_layer(self, Q, K_s, layer_index):
        """
        singel layer for encoder.each layers has two sub-layers:
        the first is multi-head self-attention mechanism; the second is fully connected feed-forward network.
        for each sublayer. use LayerNorm(x+Sublayer(x)). input and output of last dimension: d_model
        :param Q: shape should be:       [batch_size*sequence_length,d_model]
        :param K_s: shape should be:     [batch_size*sequence_length,d_model]
        :return:output: shape should be:[batch_size*sequence_length,d_model]
        """
        # 1.1 the first is multi-head self-attention mechanism
        multi_head_attention_output = self.encode_stack[layer_index]['MultiHeadAttention']([Q, K_s, self.V_s[layer_index]])

        # 1.2 use LayerNorm(x+Sublayer(x)). all dimension=512.  [batch_size,sequence_length,d_model]
        multi_head_attention_output = self.encode_stack[layer_index]['MHA_Resid']([K_s, multi_head_attention_output])

        # 2.1 the second is fully connected feed-forward network.
        feed_forward_output = self.encode_stack[layer_index]['FeedForward'](multi_head_attention_output)

        # 2.2 use LayerNorm(x+Sublayer(x)). all dimension=512.
        feed_forward_output = self.encode_stack[layer_index]['FF_Resid']([multi_head_attention_output, feed_forward_output])
        return feed_forward_output, feed_forward_output