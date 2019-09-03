# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Attention (Attention is all you need, https://arxiv.org/abs/1706.03762)
# Authors:     Yage Wang
# Created:     5.29.2019
###############################################################################

import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, level, sequence_length, output_dim, **kwargs):
        super(Attention, self).__init__(name=f"Attention_{level}", **kwargs)

        # Create a trainable weight variable for this layer.

        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.W_a = self.add_weight(
            name=f"W_{level}",
            shape=[output_dim, output_dim],
            initializer="uniform",
        )
        self.b_a = self.add_weight(
            name=f"b_{level}", shape=[output_dim], initializer="uniform"
        )
        self.context_vecotor = self.add_weight(
            name=f"context_vecotor_{level}",
            shape=[output_dim],
            initializer="uniform",
        )

    @tf.function
    def call(self, hidden_state):
        """
        @param hidden_state: [batch_size*num_sentences,sentence_length,d_model]
        @return representation [batch_size*num_sentences,d_model]
        """

        token_hidden_state_2 = tf.reshape(
            hidden_state, shape=[-1, self.output_dim]
        )
        token_hidden_representation = tf.tanh(
            tf.matmul(token_hidden_state_2, self.W_a) + self.b_a
        )
        token_hidden_representation = tf.reshape(
            token_hidden_representation,
            shape=[-1, self.sequence_length, self.output_dim],
        )
        token_hidden_state_context_similiarity = tf.multiply(
            token_hidden_representation, self.context_vecotor
        )
        token_attention_logits = tf.reduce_sum(
            token_hidden_state_context_similiarity, axis=2
        )  # [batch_size*num_sentences,sentence_length]
        token_p_attention = tf.nn.softmax(
            token_attention_logits, name="token_attention"
        )  # [batch_size*num_sentences,sentence_length]
        token_p_attention_expanded = tf.expand_dims(
            token_p_attention, axis=2
        )  # [batch_size*num_sentences,sentence_length,1]
        representation = tf.multiply(
            token_p_attention_expanded, hidden_state
        )  # [batch_size*num_sentences,sentence_length, d_model]
        representation = tf.reduce_sum(
            input_tensor=representation, axis=1
        )  # [batch_size*num_sentences, d_model]

        return representation
