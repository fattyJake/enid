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
    def __init__(self, Q, K_s, V_s, d_model, d_k, d_v, sequence_length, h, mask=None, dropout_keep_prob=1.0):

        self.Q = Q
        self.K_s = K_s
        self.V_s = V_s
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
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
                return tf.bias_add(x, b)
            else:
                return x

    def multi_head_attention_fn(self):
        """
        multi head attention
        :param Q: query.  shape:[batch,sequence_length,d_model]
        :param K_s: keys. shape:[batch,sequence_length,d_model].
        :param V_s:values.shape:[batch,sequence_length,d_model].
        :param h: h times
        :return: result of scaled dot product attention. shape:[sequence_length,d_model]
        """
        # 1. linearly project the queries,keys and values h times(with different,learned linear projections to d_k,d_k,d_v dimensions)
        Q_projected   = tf.layers.dense(self.Q, units=self.d_model)     # [batch,sequence_length,d_model]
        K_s_projected = tf.layers.dense(self.K_s, units=self.d_model)   # [batch,sequence_length,d_model]
        V_s_projected = tf.layers.dense(self.V_s, units=self.d_model)   # [batch,sequence_length,d_model]
        # 2. scaled dot product attention for each projected version of Q,K,V
        dot_product = self.scaled_dot_product_attention_batch(Q_projected, K_s_projected, V_s_projected) # [batch,h,sequence_length,d_k]
        # 3. concatenated
        print("dot_product:=================================>",dot_product) #dot_product:(128, 8, 6, 64)
        batch_size,h,length,d_k=dot_product.get_shape().as_list()
        print("self.sequence_length:",self.sequence_length) #5
        dot_product=tf.reshape(dot_product,shape=(-1,length,self.d_model))
        # 4. linear projection
        output=tf.layers.dense(dot_product,units=self.d_model) # [batch,sequence_length,d_model]
        return output  #[batch,sequence_length,d_model]

    def scaled_dot_product_attention_batch_mine(self,Q,K_s,V_s): #my own implementation of scaled dot product attention.
        """
        scaled dot product attention
        :param Q:  query.  shape:[batch,sequence_length,d_model]
        :param K_s: keys.  shape:[batch,sequence_length,d_model]
        :param V_s:values. shape:[batch,sequence_length,d_model]
        :param mask:       shape:[batch,sequence_length]
        :return: result of scaled dot product attention. shape:[batch,h,sequence_length,d_k]
        """
        # 1. split Q,K,V
        Q_heads = tf.stack(tf.split(Q,self.h,axis=2),axis=1)         # [batch,h,sequence_length,d_k]
        K_heads = tf.stack(tf.split(K_s, self.h, axis=2), axis=1)    # [batch,h,sequence_length,d_k]
        V_heads = tf.stack(tf.split(V_s, self.h, axis=2), axis=1)    # [batch,h,sequence_length,d_k]
        dot_product=tf.multiply(Q_heads,K_heads)                     # [batch,h,sequence_length,d_k]
        # 2. dot product
        dot_product=dot_product*(1.0/tf.sqrt(tf.cast(self.d_model,tf.float32))) # [batch,h,sequence_length,d_k]
        dot_product=tf.reduce_sum(dot_product,axis=-1,keep_dims=True) # [batch,h,sequence_length,1]
        # 3. add mask if it is none
        if self.mask is not None:
            mask = tf.expand_dims(self.mask, axis=-1)  # [batch,sequence_length,1]
            mask = tf.expand_dims(mask, axis=1)  # [batch,1,sequence_length,1]
            dot_product=dot_product+mask   # [batch,h,sequence_length,1]
        # 4. get possibility
        p=tf.nn.softmax(dot_product)                                  # [batch,h,sequence_length,1]
        # 5. final output
        output=tf.multiply(p,V_heads)                                 # [batch,h,sequence_length,d_k]
        return output                                                 # [batch,h,sequence_length,d_k]

    def scaled_dot_product_attention_batch(self, Q, K_s, V_s):# scaled dot product attention: implementation style like tensor2tensor from google
        """
        scaled dot product attention
        :param Q:  query.  shape:[batch,sequence_length,d_model]
        :param K_s: keys.  shape:[batch,sequence_length,d_model]
        :param V_s:values. shape:[batch,sequence_length,d_model]
        :param mask:       shape:[sequence_length,sequence_length]
        :return: result of scaled dot product attention. shape:[batch,h,sequence_length,d_k]
        """
        # 1. split Q,K,V
        Q_heads = tf.stack(tf.split(Q,self.h,axis=2),axis=1)                    # [batch,h,sequence_length,d_k]
        K_heads = tf.stack(tf.split(K_s, self.h, axis=2), axis=1)               # [batch,h,sequence_length,d_k]
        V_heads = tf.stack(tf.split(V_s, self.h, axis=2), axis=1)               # [batch,h,sequence_length,d_k]
        # 2. dot product of Q,K
        dot_product = tf.matmul(Q_heads,K_heads,transpose_b=True)                 # [batch,h,sequence_length,sequence_length]
        dot_product = dot_product*(1.0/tf.sqrt(tf.cast(self.d_model,tf.float32))) # [batch,h,sequence_length,sequence_length]
        # 3. add mask if it is none
        print("scaled_dot_product_attention_batch.===============================================================>mask is not none?",self.mask is not None)
        if self.mask is not None:
            mask_expand=tf.expand_dims(tf.expand_dims(self.mask,axis=0),axis=0) # [1,1,sequence_length,sequence_length]
            #dot_product:(128, 8, 6, 6);mask_expand:(1, 1, 5, 5)
            print("scaled_dot_product_attention_batch.===============================================================>dot_product:",dot_product,";mask_expand:",mask_expand)
            dot_product=dot_product+mask_expand                                 # [batch,h,sequence_length,sequence_length]
        # 4.get possibility
        weights=tf.nn.softmax(dot_product)                                      # [batch,h,sequence_length,sequence_length]
        # drop out weights
        weights=tf.nn.dropout(weights,1.0-self.dropout_keep_prob)                    # [batch,h,sequence_length,sequence_length]
        # 5. final output
        output=tf.matmul(weights,V_heads)                                       # [batch,h,sequence_length,d_model]
        return output

"""
Position-wise Feed-Forward Networks
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
connected feed-forward network, which is applied to each position separately and identically. This
consists of two linear transformations with a ReLU activation in between.

FFN(x) = max(0,xW1+b1)W2+b2

While the linear transformations are the same across different positions, they use different parameters
from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
The dimensionality of input and output is d_model= 512, and the inner-layer has dimensionalityd_ff= 2048.
"""
class PositionWiseFeedFoward(object): #TODO make it parallel
    """
    position-wise feed forward networks. formula as below:
    FFN(x)=max(0,xW1+b1)W2+b2
    """
    def __init__(self,x,layer_index,d_model=512,d_ff=2048):
        """
        :param x: shape should be:[batch,sequence_length,d_model]
        :param layer_index:  index of layer
        :return: shape:[sequence_length,d_model]
        """
        shape_list=x.get_shape().as_list()
        assert(len(shape_list)==3)
        self.x=x
        self.layer_index=layer_index
        self.d_model=d_model
        self.d_ff=d_ff
        self.initializer = tf.random_normal_initializer(stddev=0.1)

    def position_wise_feed_forward_fn(self):
        """
        x:       [batch,sequence_length,d_model]
        :return: [batch,sequence_length,d_model]
        """
        output=None
        #1.conv1
        input=tf.expand_dims(self.x,axis=3) #[batch,sequence_length,d_model,1]
        # conv2d.input:       [None,sentence_length,embed_size,1]. filter=[filter_size,self.embed_size,1,self.num_filters]
        # output with padding:[None,sentence_length,1,1]
        filter1 = tf.get_variable("filter1"+str(self.layer_index) , shape=[1, self.d_model, 1, 1],initializer=self.initializer)
        ouput_conv1=tf.nn.conv2d(input,filter1,strides=[1,1,1,1],padding="VALID",name="conv1") #[batch,sequence_length,1,1]
        print("output_conv1:",ouput_conv1)

        #2.conv2
        filter2 = tf.get_variable("filter2"+str(self.layer_index), [1, 1, 1, self.d_model], initializer=self.initializer)
        output_conv2=tf.nn.conv2d(ouput_conv1,filter2,strides=[1,1,1,1],padding="VALID",name="conv2") #[batch,sequence_length,1,d_model]
        output=tf.squeeze(output_conv2) #[batch,sequence_length,d_model]
        return output #[batch,sequence_length,d_model]

"""
We employ a residual connection around each of the two sub-layers, followed by layer normalization.
That is, the output of each sub-layer is LayerNorm(x+ Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. """
class LayerNormResidualConnection(object):
    def __init__(self,x,y,layer_index,type,residual_dropout=0.1,use_residual_conn=True):
        self.x=x
        self.y=y
        self.layer_index=layer_index
        self.type=type
        self.residual_dropout=residual_dropout
        self.use_residual_conn=use_residual_conn

    #call residual connection and layer normalization
    def layer_norm_residual_connection(self):
        print("LayerNormResidualConnection.use_residual_conn:",self.use_residual_conn)
        ##if self.use_residual_conn:
        #    x_residual=self.residual_connection()
        #    x_layer_norm=self.layer_normalization(x_residual)
        #else:
        x_layer_norm = self.layer_normalization(self.x)
        return x_layer_norm

    def residual_connection(self):
        output=self.x + tf.nn.dropout(self.y, 1.0 - self.residual_dropout)
        return output

    # layer normalize the tensor x, averaging over the last dimension.
    def layer_normalization(self,x):
        """
        x should be:[batch_size,sequence_length,d_model]
        :return:
        """
        filter=x.get_shape()[-1] #last dimension of x. e.g. 512
        print("layer_normalization:==================>variable_scope:","layer_normalization"+str(self.layer_index)+self.type)
        with tf.variable_scope("layer_normalization"+str(self.layer_index)+self.type):
            # 1. normalize input by using  mean and variance according to last dimension
            mean=tf.reduce_mean(x,axis=-1,keep_dims=True) #[batch_size,sequence_length,1]
            variance=tf.reduce_mean(tf.square(x-mean),axis=-1,keep_dims=True) #[batch_size,sequence_length,1]
            norm_x=(x-mean)*tf.rsqrt(variance+1e-6) #[batch_size,sequence_length,d_model]
            # 2. re-scale normalized input back
            scale=tf.get_variable("layer_norm_scale",[filter],initializer=tf.ones_initializer) #[filter]
            bias=tf.get_variable("layer_norm_bias",[filter],initializer=tf.ones_initializer) #[filter]
            output=norm_x*scale+bias #[batch_size,sequence_length,d_model]
            return output #[batch_size,sequence_length,d_model]

class BaseClass(object):
    """
    base class has some common fields and functions.
    """
    def __init__(self,d_model,d_k,d_v,sequence_length,h,batch_size,num_layer=6,type='encoder',decoder_sent_length=None):
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