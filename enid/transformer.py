# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Tranformer (Attention is all you need, https://arxiv.org/abs/1706.03762) 
# Authors:     Yage Wang
# Created:     5.29.2019
###############################################################################

import tensorflow as tf

class MultiHeadAttention(object):
    """
    Multi head attention.
    1.linearly project the queries, keys and values h times(with different, learned linear projections to d_k, d_k, d_v dimensions)
    2.scaled dot product attention for each projected version of Q,K,V
    3.concatenated result
    4.linear projection to get final result
    """
    def __init__(self, Q, K_s, V_s, d_model, sequence_length, h, mask=None, dropout_keep_prob=1.0):

        self.Q = Q
        self.K_s = K_s
        self.V_s = V_s
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.h = h
        self.mask = mask
        self.dropout_keep_prob = dropout_keep_prob

    @staticmethod
    def _linear(x, units, scope=None):
        """ linear projection (weight_shape: input size, output size) """
        with tf.variable_scope(scope or "linear"):
            layer = tf.keras.layers.Dense(units)
            return layer.apply(x)

    def multi_head_attention_fn(self, scope="multi_head_attention"):
        """
        multi head attention
        :param Q: query.  shape:[batch,sequence_length,d_model]
        :param K_s: keys. shape:[batch,sequence_length,d_model].
        :param V_s:values.shape:[batch,sequence_length,d_model].
        :param h: h times
        :return: result of scaled dot product attention. shape:[sequence_length,d_model]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            print(self.Q, self.K_s, self.V_s)
            # 1. linearly project the queries,keys and values h times(with different,learned linear projections to d_k,d_k,d_v dimensions)
            Q_projected   = self._linear(self.Q,   self.d_model, scope="Q")     # [batch,sequence_length,d_model]
            K_s_projected = self._linear(self.K_s, self.d_model, scope="K")     # [batch,sequence_length,d_model]
            V_s_projected = self._linear(self.V_s, self.d_model, scope="V")     # [batch,sequence_length,d_model]

            # 2. scaled dot product attention for each projected version of Q,K,V
            dot_product = self.scaled_dot_product_attention_batch(Q_projected, K_s_projected, V_s_projected) # [batch,h,sequence_length,d_k]

            # 3. concatenated
            dot_product = tf.reshape(dot_product, shape=(-1, self.sequence_length, self.d_model))

            # 4. linear projection
            output = self._linear(dot_product, self.d_model, scope="output")         # [batch,sequence_length,d_model]
            return output  #[batch,sequence_length,d_model]

    def scaled_dot_product_attention_batch(self, Q, K_s, V_s, scope="scaled_dot_product_attention"):
        """
        scaled dot product attention
        :param Q:  query.  shape:[batch,sequence_length,d_model]
        :param K_s: keys.  shape:[batch,sequence_length,d_model]
        :param V_s:values. shape:[batch,sequence_length,d_model]
        :param mask:       shape:[sequence_length,sequence_length]
        :return: result of scaled dot product attention. shape:[batch,h,sequence_length,d_k]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # 1. split Q,K,V
            Q_heads = tf.stack(tf.split(Q, self.h, axis=2), axis=1)                 # [batch,h,sequence_length,d_k]
            K_heads = tf.stack(tf.split(K_s, self.h, axis=2), axis=1)               # [batch,h,sequence_length,d_k]
            V_heads = tf.stack(tf.split(V_s, self.h, axis=2), axis=1)               # [batch,h,sequence_length,d_k]

            # 2. dot product of Q,K
            dot_product = tf.matmul(Q_heads, K_heads, transpose_b=True)                    # [batch,h,sequence_length,sequence_length]
            dot_product = dot_product * (1.0 / tf.sqrt(tf.cast(self.d_model, tf.float32))) # [batch,h,sequence_length,sequence_length]

            # 3. add mask if it is none
            if self.mask is not None:
                mask_expand = tf.expand_dims(tf.expand_dims(self.mask, axis=0), axis=0) # [1,1,sequence_length,sequence_length]
                dot_product = dot_product + mask_expand                                 # [batch,h,sequence_length,sequence_length]

            # 4.get possibility
            weights = tf.nn.softmax(dot_product)                                        # [batch,h,sequence_length,sequence_length]
            weights = tf.nn.dropout(weights, self.dropout_keep_prob)                    # [batch,h,sequence_length,sequence_length]
            
            # 5. final output
            output = tf.matmul(weights, V_heads)                                        # [batch,h,sequence_length,d_k]
            return output

class FeedFoward(object): #TODO make it parallel
    """
    Feed-Forward Networks
    In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
    connected feed-forward network, which is applied to each position separately and identically. This
    consists of two linear transformations with a ReLU activation in between.

    FFN(x) = max(0,xW1+b1)W2+b2

    While the linear transformations are the same across different positions, they use different parameters
    from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
    The dimensionality of input and output is d_model= 512, and the inner-layer has dimensionality d_ff= 2048.
    """
    def __init__(self, x, layer_index, d_model, d_ff):
        """
        :param x: shape should be:[batch,sequence_length,d_model]
        :param layer_index:  index of layer
        :return: shape:[sequence_length,d_model]
        """
        shape_list = x.get_shape().as_list()
        assert(len(shape_list)==3)
        self.x = x
        self.layer_index = layer_index
        self.d_model = d_model
        self.d_ff = d_ff
        self.initializer = tf.random_normal_initializer(stddev=0.1)

    @staticmethod
    def _linear(x, units, activation, scope=None):
        """ linear projection (weight_shape: input size, output size) """
        with tf.variable_scope(scope or "linear"):
            if activation: layer = tf.keras.layers.Dense(units, activation=tf.nn.relu)
            else: layer = tf.keras.layers.Dense(units)
            return layer.apply(x)

    def feed_forward_fn(self):
        """
        x:       [batch,sequence_length,d_model]
        :return: [batch,sequence_length,d_model]
        """
        # Inner layer
        outputs = self._linear(self.x, self.d_ff, True, scope="inner_layer")

        # Outer layer
        outputs = self._linear(outputs, self.d_model, False, scope="outer_layer")
        
        return outputs #[batch,sequence_length,d_model]

class LayerNormResidualConnection(object):
    """
    We employ a residual connection around each of the two sub-layers, followed by layer normalization.
    That is, the output of each sub-layer is LayerNorm(x+ Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.
    """
    def __init__(self, x, y, layer_index, dropout_keep_prob=0.9, use_residual_conn=True):
        self.x = x
        self.y = y
        self.layer_index = layer_index
        self.dropout_keep_prob = dropout_keep_prob
        self.use_residual_conn = use_residual_conn

    #call residual connection and layer normalization
    def layer_norm_residual_connection(self):
        if self.use_residual_conn:
            x_residual = self.residual_connection()
            x_layer_norm = self.layer_normalization(x_residual)
        else:
            x_layer_norm = self.layer_normalization(self.x)
        return x_layer_norm

    def residual_connection(self):
        output = self.x + tf.nn.dropout(self.y, self.dropout_keep_prob)
        return output

    # layer normalize the tensor x, averaging over the last dimension.
    def layer_normalization(self, x, scope="layer_normalization"):
        """
        x should be:[batch_size,sequence_length,d_model]
        :return:
        """
        filter_ = x.get_shape()[-1] # last dimension of x. e.g. 512
        with tf.variable_scope(scope + "_" + str(self.layer_index)):
            # 1. normalize input by using  mean and variance according to last dimension
            mean = tf.reduce_mean(x, axis=-1, keep_dims=True) #[batch_size,sequence_length,1]
            variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keep_dims=True) #[batch_size,sequence_length,1]
            norm_x = (x - mean) * tf.rsqrt(variance + 1e-6) #[batch_size,sequence_length,d_model]

            # 2. re-scale normalized input back
            scale = tf.get_variable("layer_norm_scale", [filter_], initializer=tf.ones_initializer) #[filter]
            bias  = tf.get_variable("layer_norm_bias",  [filter_], initializer=tf.ones_initializer) #[filter]
            output = norm_x * scale + bias #[batch_size,sequence_length,d_model]
            return output #[batch_size,sequence_length,d_model]

class Encoder(object):
    """
    base class has some common fields and functions.
    """
    def __init__(self, Q, K_s, d_model, d_ff, sequence_length, h, batch_size, num_layer=6, mask=None,
                 dropout_keep_prob=None, use_residual_conn=True, initializer=tf.initializers.random_normal(stddev=0.1)):
        """
        :param d_model:
        :param d_k:
        :param d_v:
        :param sequence_length:
        :param h:
        :param batch_size:
        :param embedded_words: shape:[batch_size,sequence_length,embed_size]
        """
        self.Q = Q
        self.K_s = K_s
        self.d_model = d_model
        self.d_ff = d_ff
        self.sequence_length = sequence_length
        self.h = h
        self.num_layer = num_layer
        self.batch_size = batch_size
        self.mask = mask
        self.dropout_keep_prob = dropout_keep_prob
        self.use_residual_conn = use_residual_conn
        self.initializer = initializer

    def encoder_multiple_layers(self):
        for layer_index in range(self.num_layer):
            self.Q, self.K_s = self.encoder_single_layer(self.Q, self.K_s, layer_index)
        return self.Q, self.K_s

    def encoder_single_layer(self, Q, K_s, layer_index):
        """
        singel layer for encoder.each layers has two sub-layers:
        the first is multi-head self-attention mechanism; the second is fully connected feed-forward network.
        for each sublayer. use LayerNorm(x+Sublayer(x)). input and output of last dimension: d_model
        :param Q: shape should be:       [batch_size*sequence_length,d_model]
        :param K_s: shape should be:     [batch_size*sequence_length,d_model]
        :return:output: shape should be:[batch_size*sequence_length,d_model]
        """
        #1.1 the first is multi-head self-attention mechanism
        multi_head_attention_output = self.sub_layer_multi_head_attention(layer_index, Q, K_s, mask=self.mask, dropout_keep_prob=self.dropout_keep_prob) #[batch_size,sequence_length,d_model]
        
        #1.2 use LayerNorm(x+Sublayer(x)). all dimension=512.  [batch_size,sequence_length,d_model]
        multi_head_attention_output = self.sub_layer_layer_norm_residual_connection(K_s, multi_head_attention_output, layer_index, "mhatt_resid",
                                                                                    dropout_keep_prob=self.dropout_keep_prob, use_residual_conn=self.use_residual_conn)

        #2.1 the second is fully connected feed-forward network.
        feed_forward_output = self.sub_layer_feed_forward(multi_head_attention_output, layer_index)

        #2.2 use LayerNorm(x+Sublayer(x)). all dimension=512.
        feed_forward_output = self.sub_layer_layer_norm_residual_connection(multi_head_attention_output, feed_forward_output, "ff_resid",
                                                                            layer_index, dropout_keep_prob=self.dropout_keep_prob)
        return  feed_forward_output, feed_forward_output

    def sub_layer_multi_head_attention(self, layer_index, Q, K_s, mask=None, dropout_keep_prob=None)  :# COMMON FUNCTION
        """
        multi head attention as sub layer
        :param layer_index: index of layer number
        :param Q: shape should be: [batch_size,sequence_length,embed_size]
        :param k_s: shape should be: [batch_size,sequence_length,embed_size]
        :param type: encoder,decoder or encoder_decoder_attention
        :param mask: when use mask,illegal connection will be mask as huge big negative value.so it's possiblitity will become zero.
        :return: output of multi head attention.shape:[batch_size,sequence_length,d_model]
        """
        with tf.variable_scope("encoder_sub_layer_multi_head_attention_" + str(layer_index)):
            # below is to handle attention for encoder and decoder with difference length:
            #1. get V as learned parameters
            V_s = tf.get_variable("V_s", shape=(self.batch_size, self.sequence_length, self.d_model), initializer=self.initializer)

            #2. call function of multi head attention to get result
            multi_head_attention_class = MultiHeadAttention(Q, K_s, V_s, self.d_model, self.sequence_length,
                                                            self.h, mask=mask, dropout_keep_prob=dropout_keep_prob)

            sub_layer_multi_head_attention_output = multi_head_attention_class.multi_head_attention_fn()  # [batch_size*sequence_length,d_model]
        return sub_layer_multi_head_attention_output  # [batch_size,sequence_length,d_model]

    def sub_layer_feed_forward(self, x, layer_index)  :# COMMON FUNCTION
        """
        :param x: shape should be:[batch_size,sequence_length,d_model]
        :param layer_index: index of layer number
        :param type: encoder,decoder or encoder_decoder_attention
        :return: [batch_size,sequence_length,d_model]
        """
        with tf.variable_scope("encoder_sub_layer_feed_forward_" + str(layer_index)):
            feed_forward = FeedFoward(x, layer_index, self.d_model, self.d_ff)
            feed_forward_output = feed_forward.feed_forward_fn()
        return feed_forward_output

    def sub_layer_layer_norm_residual_connection(self, layer_input, layer_output, layer_index, scope, dropout_keep_prob=None, use_residual_conn=True): # COMMON FUNCTION
        """
        layer norm & residual connection
        :param input: [batch_size,equence_length,d_model]
        :param output:[batch_size,sequence_length,d_model]
        :return:
        """
        with tf.variable_scope("encoder_sub_layer_norm_residual_" + str(layer_index) + '_' + str(scope)):
            layer_norm_residual_conn = LayerNormResidualConnection(layer_input, layer_output, layer_index, dropout_keep_prob=dropout_keep_prob,
                                                                   use_residual_conn=use_residual_conn)
            output = layer_norm_residual_conn.layer_norm_residual_connection()
            return output  # [batch_size,sequence_length,d_model]