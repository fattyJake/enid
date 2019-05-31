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
    def _linear(x, weight_shape, bias=True, scope=None):
        """ linear projection (weight_shape: input size, output size) """
        with tf.variable_scope(scope or "linear"):
            w = tf.get_variable("kernel", shape=weight_shape)
            x = tf.matmul(x, w)
            if bias:
                b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
                return tf.nn.bias_add(x, b)
            else:
                return x

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
            # 1. linearly project the queries,keys and values h times(with different,learned linear projections to d_k,d_k,d_v dimensions)
            Q_projected   = self._linear(self.Q,   weight_shape=self.d_model, scope="Q")     # [batch,sequence_length,d_model]
            K_s_projected = self._linear(self.K_s, weight_shape=self.d_model, scope="K")     # [batch,sequence_length,d_model]
            V_s_projected = self._linear(self.V_s, weight_shape=self.d_model, scope="V")     # [batch,sequence_length,d_model]

            # 2. scaled dot product attention for each projected version of Q,K,V
            dot_product = self.scaled_dot_product_attention_batch(Q_projected, K_s_projected, V_s_projected) # [batch,h,sequence_length,d_k]

            # 3. concatenated
            dot_product = tf.reshape(dot_product, shape=(-1, self.sequence_length, self.d_model))

            # 4. linear projection
            output = self._linear(dot_product, weight_shape=self.d_model, scope="output")         # [batch,sequence_length,d_model]
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

class PositionWiseFeedFoward(object): #TODO make it parallel
    """
    Position-wise Feed-Forward Networks
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
    def _linear(x, weight_shape, activation, scope=None):
        """ linear projection (weight_shape: input size, output size) """
        with tf.variable_scope(scope or "linear"):
            w = tf.get_variable("kernel", shape=weight_shape)
            x = tf.matmul(x, w)
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
            if activation:
                return tf.nn.relu(tf.nn.bias_add(x, b))
            else:
                return tf.nn.bias_add(x, b)

    def position_wise_feed_forward_fn(self, scope="feedforward"):
        """
        x:       [batch,sequence_length,d_model]
        :return: [batch,sequence_length,d_model]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = self._linear(self.x, self.d_ff, True, scope="inner_layer")

            # Outer layer
            outputs = self._linear(self.x, self.d_ff, False, scope="outer_layer")
        
        return outputs #[batch,sequence_length,d_model]

class LayerNormResidualConnection(object):
    """
    We employ a residual connection around each of the two sub-layers, followed by layer normalization.
    That is, the output of each sub-layer is LayerNorm(x+ Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.
    """
    def __init__(self, x, y, layer_index, type_, residual_dropout=0.1, use_residual_conn=True):
        self.x = x
        self.y = y
        self.layer_index = layer_index
        self.type_ = type_
        self.residual_dropout = residual_dropout
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
        output = self.x + tf.nn.dropout(self.y, 1.0 - self.residual_dropout)
        return output

    # layer normalize the tensor x, averaging over the last dimension.
    def layer_normalization(self, x, scope="layer_normalization"):
        """
        x should be:[batch_size,sequence_length,d_model]
        :return:
        """
        filter_ = x.get_shape()[-1] # last dimension of x. e.g. 512
        print("layer_normalization:==================>variable_scope:","layer_normalization"+str(self.layer_index)+self.type)
        with tf.variable_scope(scope + str(self.layer_index) + self.type_):
            # 1. normalize input by using  mean and variance according to last dimension
            mean = tf.reduce_mean(x, axis=-1, keep_dims=True) #[batch_size,sequence_length,1]
            variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keep_dims=True) #[batch_size,sequence_length,1]
            norm_x = (x - mean) * tf.rsqrt(variance + 1e-6) #[batch_size,sequence_length,d_model]

            # 2. re-scale normalized input back
            scale = tf.get_variable("layer_norm_scale", [filter_], initializer=tf.ones_initializer) #[filter]
            bias = tf.get_variable("layer_norm_bias", [filter_], initializer=tf.ones_initializer) #[filter]
            output = norm_x * scale + bias #[batch_size,sequence_length,d_model]
            return output #[batch_size,sequence_length,d_model]

class BaseClass(object):
    """
    base class has some common fields and functions.
    """
    def __init__(self,d_model,d_k,d_v,sequence_length,h,batch_size,num_layer=6,type_='encoder',decoder_sent_length=None,initializer=tf.initializers.random_normal()):
        """
        :param d_model:
        :param d_k:
        :param d_v:
        :param sequence_length:
        :param h:
        :param batch_size:
        :param embedded_words: shape:[batch_size,sequence_length,embed_size]
        """
        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.sequence_length=sequence_length
        self.h=h
        self.num_layer=num_layer
        self.batch_size=batch_size
        self.type=type
        self.decoder_sent_length=decoder_sent_length
        self.initializer = initializer

    def sub_layer_postion_wise_feed_forward(self ,x ,layer_index,type)  :# COMMON FUNCTION
        """
        :param x: shape should be:[batch_size,sequence_length,d_model]
        :param layer_index: index of layer number
        :param type: encoder,decoder or encoder_decoder_attention
        :return: [batch_size,sequence_length,d_model]
        """
        with tf.variable_scope("sub_layer_postion_wise_feed_forward" + type + str(layer_index)):
            postion_wise_feed_forward = PositionWiseFeedFoward(x, layer_index)
            postion_wise_feed_forward_output = postion_wise_feed_forward.position_wise_feed_forward_fn()
        return postion_wise_feed_forward_output

    def sub_layer_multi_head_attention(self ,layer_index ,Q ,K_s,type,mask=None,dropout_keep_prob=None)  :# COMMON FUNCTION
        """
        multi head attention as sub layer
        :param layer_index: index of layer number
        :param Q: shape should be: [batch_size,sequence_length,embed_size]
        :param k_s: shape should be: [batch_size,sequence_length,embed_size]
        :param type: encoder,decoder or encoder_decoder_attention
        :param mask: when use mask,illegal connection will be mask as huge big negative value.so it's possiblitity will become zero.
        :return: output of multi head attention.shape:[batch_size,sequence_length,d_model]
        """
        with tf.variable_scope("base_mode_sub_layer_multi_head_attention_" + type+str(layer_index)):
            # below is to handle attention for encoder and decoder with difference length:
            #length=self.decoder_sent_length if (type!='encoder' and self.sequence_length!=self.decoder_sent_length) else self.sequence_length #TODO this may be useful
            length=self.sequence_length
            #1. get V as learned parameters
            V_s = tf.get_variable("V_s", shape=(self.batch_size,length,self.d_model),initializer=self.initializer)
            #2. call function of multi head attention to get result
            multi_head_attention_class = MultiHeadAttention(Q, K_s, V_s, self.d_model, self.d_k, self.d_v, self.sequence_length,
                                                            self.h,type=type,mask=mask,dropout_keep_prob=(1.0-dropout_keep_prob))
            sub_layer_multi_head_attention_output = multi_head_attention_class.multi_head_attention_fn()  # [batch_size*sequence_length,d_model]
        return sub_layer_multi_head_attention_output  # [batch_size,sequence_length,d_model]

    def sub_layer_layer_norm_residual_connection(self,layer_input ,layer_output,layer_index,type,dropout_keep_prob=None,use_residual_conn=True): # COMMON FUNCTION
        """
        layer norm & residual connection
        :param input: [batch_size,equence_length,d_model]
        :param output:[batch_size,sequence_length,d_model]
        :return:
        """
        print("@@@========================>layer_input:",layer_input,";layer_output:",layer_output)
        #assert layer_input.get_shape().as_list()==layer_output.get_shape().as_list()
        #layer_output_new= layer_input+ layer_output
        layer_norm_residual_conn=LayerNormResidualConnection(layer_input,layer_output,layer_index,type,residual_dropout=(1-dropout_keep_prob),use_residual_conn=use_residual_conn)
        output = layer_norm_residual_conn.layer_norm_residual_connection()
        return output  # [batch_size,sequence_length,d_model]