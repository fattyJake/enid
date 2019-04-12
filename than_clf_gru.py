# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Time-Aware Hierarchical Attention Model class with embedding
# Authors:     Yage Wang
# Created:     9.26.2018
###############################################################################

import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from .tlstm import TLSTMCell
from datetime import datetime
import numpy as np

class T_HAN(object):
    """
    A Time-Aware-HAN for claim classification
    Uses an embedding layer, followed by a token-level bi-GRU with attention, a sentence-level time-aware-lstm with attention and sofrmax layer

    Parameters
    ----------
    max_sequence_length: int
        fixed padding latest number of time buckets

    max_sentence_length: int
        fixed padding number of tokens each time bucket

    num_classes: int
        the number of y classes

    hidden_size: int
        number of T_LSTM units

    pretrain_embedding: 2-D numpy array (vocab_size, embedding_size)
        random initialzed embedding matrix

    learning_rate: float
        initial learning rate for Adam Optimizer

    decay_steps: int
        step frequency to decay the learning rate. e.g. if 5000, model will reduce learning rate by decay_rate every 5000 trained batches

    decay_rate: float
        percentage of learning rate decay rate

    dropout_keep_prob: float
        percentage of neurons to keep from dropout regularization each layer

    l2_reg_lambda: float, default 0
        L2 regularization lambda for fully-connected layer to prevent potential overfitting

    objective: str, default 'ce'
        the objective function (loss) model trains on; if 'ce', use cross-entropy, if 'auc', use AUROC as objective

    initializer: tf tensor initializer object, default tf.orthogonal_initializer()
        initializer for fully connected layer weights

    Examples
    --------
    >>> from enid.than_clf import T_HAN
    >>> rnn = T_HAN(max_sequence_length=50, max_sentence_length=20,
            hidden_size=128, num_classes=2, pretrain_embedding=emb,
            learning_rate=0.05, decay_steps=5000, decay_rate=0.9,
            dropout_keep_prob=0.8, l2_reg_lambda=0.0,
            objective='ce')
    """
    def __init__(
        self, max_sequence_length, max_sentence_length, num_classes, hidden_size, pretrain_embedding,
        learning_rate, decay_steps, decay_rate, dropout_keep_prob, l2_reg_lambda=0.0, objective='ce',
        initializer=tf.orthogonal_initializer()):

        """init all hyperparameter here"""
        tf.reset_default_graph()

        self.num_classes = num_classes
        self.pretrain_embedding = pretrain_embedding
        self.max_sequence_length = max_sequence_length
        self.max_sentence_length = max_sentence_length
        self.hidden_size = hidden_size
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate
        self.initializer = initializer

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        # add placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.max_sequence_length, self.max_sentence_length], name="input_x") # X [instance_size, num_bucket, sentence_length]
        self.input_t = tf.placeholder(tf.int32, [None, self.max_sequence_length],                           name="input_t") # T [instance_size, num_bucket]
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes],                                   name="input_y") # y [instance_size, num_classes]
        
        with tf.name_scope("embedding"):
            self.emb_size = self.pretrain_embedding.shape[1]
            embedding_matrix = tf.concat([self.pretrain_embedding, tf.zeros((1, self.emb_size))], axis=0)
            self.Embedding = tf.Variable(embedding_matrix, trainable=True, dtype=tf.float32, name='embedding')
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size, self.num_classes], initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        # 1. get emebedding of tokens
        self.input = tf.nn.embedding_lookup(self.Embedding, self.input_x) # [instance_size, num_bucket, sentence_length, embedding_size]
        self.batch_size = tf.shape(self.input)[0]
        
        # 2. token level attention
        self.input = tf.reshape(self.input, shape=[-1, self.max_sentence_length, self.emb_size])

        self.tlstm_cell_fw = TLSTMCell(self.hidden_size, False, dropout_keep_prob_in=self.dropout_keep_prob,
                                       dropout_keep_prob_h=self.dropout_keep_prob, dropout_keep_prob_out=self.dropout_keep_prob,
                                       dropout_keep_prob_gate=self.dropout_keep_prob, dropout_keep_prob_forget=self.dropout_keep_prob)
        self.tlstm_cell_bw = TLSTMCell(self.hidden_size, False, dropout_keep_prob_in=self.dropout_keep_prob,
                                       dropout_keep_prob_h=self.dropout_keep_prob, dropout_keep_prob_out=self.dropout_keep_prob,
                                       dropout_keep_prob_gate=self.dropout_keep_prob, dropout_keep_prob_forget=self.dropout_keep_prob)

        self.hidden_state, _ = tf.nn.bidirectional_dynamic_rnn(self.tlstm_cell_fw, self.tlstm_cell_bw, self.input, dtype=tf.float32, time_major=False)
        self.hidden_state = tf.concat(self.hidden_state, axis=2) # [batch_size*num_sentences, num_tokens, hidden_size*2]

        sentence_representation = self._attention_token_level(self.hidden_state)  # output:[batch_size*num_sentences,hidden_size*2]
        sentence_representation = tf.reshape(sentence_representation, shape=[-1, self.max_sequence_length, self.hidden_size * 2]) # shape:[batch_size,num_sentences,hidden_size*2]
        
        # 3. time-aware lstm of sentence sequence
        # make scan_time [batch_size x seq_length x 1]
        scan_time = tf.reshape(self.input_t, [tf.shape(self.input_t)[0], tf.shape(self.input_t)[1], 1])
        concat_input = tf.concat([tf.cast(scan_time, tf.float32), sentence_representation], 2) # [batch_size x seq_length x num_filters_total+1]

        self.tlstm_cell = TLSTMCell(self.hidden_size, True, dropout_keep_prob_in=self.dropout_keep_prob,
                                    dropout_keep_prob_h=self.dropout_keep_prob, dropout_keep_prob_out=self.dropout_keep_prob,
                                    dropout_keep_prob_gate=self.dropout_keep_prob, dropout_keep_prob_forget=self.dropout_keep_prob)
        
        self.hidden_state_sentence, _ = tf.nn.dynamic_rnn(self.tlstm_cell, concat_input, dtype=tf.float32, time_major=False)
        self.instance_representation = self._attention_sentence_level(self.hidden_state_sentence)
        with tf.name_scope("output"):
            self.logits = tf.matmul(self.instance_representation, self.W_projection) + self.b_projection # [batch_size, self.num_classes]. main computation graph is here.
            self.probs = tf.nn.softmax(self.logits, name="probs")

        assert objective in ['ce', 'auc'], 'AttributeError: objective only acccept "ce" or "auc", got {}'.format(str(objective))
        if objective == 'ce':  self.loss_val = self._loss(self.l2_reg_lambda)
        if objective == 'auc': self.loss_val = self._loss_roc_auc(self.l2_reg_lambda)
        self.train_op = self._train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]

        # performance
        with tf.name_scope("performance"):
            _, self.auc = tf.metrics.auc(self.input_y, self.probs, num_thresholds=3000, curve="ROC", name="auc")
        
        self.loss_sum = tf.summary.scalar("loss_train", self.loss_val)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)
        self.attention_sum = tf.summary.histogram("attentions", self.instance_representation)
        self.merged_sum = tf.summary.merge_all()

    # def _init_weights(self, input_size, output_dim, name, std=0.1, reg=None):
    #     return tf.get_variable(name,shape=[input_size, output_dim],initializer=self.initializer, regularizer = reg)

    # def _init_bias(self, output_dim, name):
    #     return tf.get_variable(name,shape=[output_dim],initializer=tf.constant_initializer(0.0))

    # def _map_elapse_time(self, t):
    #     c1 = tf.constant(1, dtype=tf.float32)
    #     c2 = tf.constant(2.7183, dtype=tf.float32)
    #     T = tf.div(c1, tf.log(t + c2), name='Log_elapse_time') # according to paper, used for large time delta like days
    #     Ones = tf.ones([1, self.hidden_size], dtype=tf.float32)
    #     T = tf.matmul(T, Ones)
    #     return T

    # def _batch_norm(self, x, epsilon=1e-3, decay=0.999):
    #     '''Assume 2d [batch, values] tensor'''

    #     size = x.get_shape().as_list()[1]

    #     scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
    #     offset = tf.get_variable('offset', [size])

    #     pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
    #     pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
    #     batch_mean, batch_var = tf.nn.moments(x, [0])

    #     train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    #     train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

    #     def batch_statistics():
    #         with tf.control_dependencies([train_mean_op, train_var_op]):
    #             return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

    #     return batch_statistics()

    # def _TLSTM_Unit(self, prev_hidden_memory, concat_input):
    #     prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

    #     batch_size = tf.shape(concat_input)[0]
    #     x = tf.slice(concat_input, [0,1], [batch_size, self.hidden_size*2])
    #     t = tf.slice(concat_input, [0,0], [batch_size,1])

    #     # Dealing with time irregularity
    #     # Map elapse time in days or months
    #     T = self._map_elapse_time(t)

    #     # Decompose the previous cell if there is a elapse time
    #     C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
    #     C_ST_dis = tf.multiply(T, C_ST)
    #     # if T is 0, then the weight is one
    #     prev_cell = prev_cell - C_ST + C_ST_dis

    #     i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi) # Input gate
    #     f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf) # Forget Gate
    #     o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog) # Output Gate

    #     C = tf.nn.tanh(self._batch_norm(tf.matmul(x, self.Wc)) + self._batch_norm(tf.matmul(prev_hidden_state, self.Uc)) + self.bc) # Candidate Memory Cell
    #     Ct = f * prev_cell + i * C # Current Memory cell
    #     current_hidden_state = o * tf.nn.tanh(Ct) # Current Hidden state

    #     return tf.stack([current_hidden_state, Ct])

    # def _gru_forward_token_level(self, embedded_tokens):
    #     """
    #     @param embedded_tokens: [batch_size*num_sentences, sentence_length, embed_size]
    #     @return [batch_size*num_sentences, sentence_length, hidden_size]
    #     """
    #     with tf.variable_scope("gru_weights_token_level_forward"):
    #         self.wf_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
    #         self.wf_cell = tf.nn.rnn_cell.DropoutWrapper(self.wf_cell,
    #                                                      input_keep_prob=self.dropout_keep_prob,
    #                                                      output_keep_prob=self.dropout_keep_prob)
    #         init_state = self.wf_cell.zero_state(batch_size=self.batch_size*self.max_sequence_length, dtype=tf.float32)  # [batch_size, hidden_size]
    #         output, state = tf.nn.dynamic_rnn(self.wf_cell, embedded_tokens, initial_state=init_state, time_major=False)
        
    #     return output

    # def _gru_backward_token_level(self, embedded_tokens):
    #     """
    #     @param embedded_tokens: [batch_size*num_sentences, sentence_length, embed_size]
    #     @return [batch_size*num_sentences, sentence_length, hidden_size]
    #     """
    #     embedded_tokens_reverse = tf.reverse(embedded_tokens, [2])
    #     with tf.variable_scope("gru_weights_token_level_backward"):
    #         self.wb_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
    #         self.wb_cell = tf.nn.rnn_cell.DropoutWrapper(self.wb_cell,
    #                                                      input_keep_prob=self.dropout_keep_prob,
    #                                                      output_keep_prob=self.dropout_keep_prob)
    #         init_state = self.wb_cell.zero_state(batch_size=self.batch_size*self.max_sequence_length, dtype=tf.float32)  # [batch_size, hidden_size]
    #         output, state = tf.nn.dynamic_rnn(self.wb_cell, embedded_tokens_reverse, initial_state=init_state, time_major=False)
    #     output = tf.reverse(output, [2])
    #     return output

    # def _time_aware_lstm_sentence_level(self, sentence_input):
    #     """
    #     @param sentence_input: [batch_size, num_sentences, hidden_size*2]
    #     @return sentence_states [batch_size, num_sentences, hidden_size]
    #     """
    #     batch_size = tf.shape(sentence_input)[0]
    #     scan_input_ = tf.transpose(sentence_input, perm=[2, 0, 1])
    #     scan_input = tf.transpose(scan_input_) # scan input is [seq_length x batch_size x hidden_size*2]
    #     scan_time = tf.transpose(self.input_t) # scan_time [seq_length x batch_size]
    #     initial_hidden = tf.zeros([batch_size, self.hidden_size], tf.float32)
    #     ini_state_cell = tf.stack([initial_hidden, initial_hidden])

    #     # make scan_time [seq_length x batch_size x 1]
    #     scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
    #     concat_input = tf.concat([tf.cast(scan_time, tf.float32), scan_input],2) # [seq_length x batch_size x hidden_size*2+1]
    #     packed_hidden_states = tf.scan(self._TLSTM_Unit, concat_input, initializer=ini_state_cell)
    #     return packed_hidden_states[:, 0, :, :]

    def _attention_token_level(self, hidden_state):
        """
        @param hidden_state: [batch_size*num_sentences,sentence_length,hidden_size*2]
        @return representation [batch_size*num_sentences,hidden_size*2]
        """
        with tf.name_scope("token_level_attention"):
            self.W_w_attention_token = tf.get_variable("W_w_attention_token",
                                                      shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                      initializer=self.initializer)
            self.W_b_attention_token = tf.get_variable("W_b_attention_token", shape=[self.hidden_size * 2])
            self.context_vecotor_token = tf.get_variable("what_is_the_informative_token", shape=[self.hidden_size * 2],
                                                        initializer=tf.random_normal_initializer(stddev=0.1))
        
        hidden_state_2 = tf.reshape(hidden_state, shape=[-1, self.hidden_size * 2])
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2, self.W_w_attention_token) + self.W_b_attention_token)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.max_sentence_length, self.hidden_size * 2])
        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_token)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)  # [batch_size*num_sentences,sentence_length]
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keepdims=True)  # [batch_size*num_sentences,1]
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)  # [batch_size*num_sentences,sentence_length]
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # [batch_size*num_sentences,sentence_length,1]
        sentence_representation = tf.multiply(p_attention_expanded, hidden_state)  # [batch_size*num_sentences,sentence_length, hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)  # [batch_size*num_sentences, hidden_size*2]
        return sentence_representation

    def _attention_sentence_level(self, hidden_state_sentence):
        """
        @param hidden_state_sentence: [batch_size, num_sentences, hidden_size]
        @return representation:[batch_size, hidden_size]
        """
        with tf.name_scope("sentence_level_attention"):
            self.W_w_attention_sentence = tf.get_variable("W_w_attention_sentence",
                                                          shape=[self.hidden_size, self.hidden_size],
                                                          initializer=self.initializer)
            self.W_b_attention_sentence = tf.get_variable("W_b_attention_sentence", shape=[self.hidden_size])
            self.context_vecotor_sentence = tf.get_variable("what_is_the_informative_sentence",
                                                            shape=[self.hidden_size], initializer=tf.random_normal_initializer(stddev=0.1))
        
        hidden_state_2 = tf.reshape(hidden_state_sentence, shape=[-1, self.hidden_size])
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2, self.W_w_attention_sentence) + self.W_b_attention_sentence)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.max_sequence_length, self.hidden_size])
        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_sentence)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keepdims=True)
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        instance_representation = tf.multiply(p_attention_expanded, hidden_state_sentence)
        instance_representation = tf.reduce_sum(instance_representation, axis=1)
        return instance_representation

    def _train(self):
        """
        based on the loss, use Adam to update parameter
        """
        #learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=self.learning_rate, optimizer="Adam")
        return train_op

    def _loss(self, l2_reg_lambda):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # losses = tf.nn.weighted_cross_entropy_with_logits(targets=self.input_y, logits=self.logits, pos_weight=self.pos_weight)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_reg_lambda
            loss = tf.identity(loss + l2_losses, name='loss')
        return loss

    def _loss_roc_auc(self, l2_reg_lambda):
        """
        ROC AUC Score.
        Approximates the Area Under Curve score, using approximation based on
        the Wilcoxon-Mann-Whitney U statistic.
        Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
        Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
        Measures overall performance for a full range of threshold levels.
        """
        pos = tf.boolean_mask(self.logits, tf.cast(self.input_y, tf.bool))
        neg = tf.boolean_mask(self.logits, ~tf.cast(self.input_y, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma
        masked = tf.boolean_mask(difference, difference < 0.0)
        loss = tf.reduce_sum(tf.pow(-masked, p))
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_reg_lambda
        loss = tf.identity(loss + l2_losses, name='loss')
        return loss

def train_han(model, t_train, x_train, y_train, dev_sample_percentage, num_epochs, batch_size, evaluate_every, model_path, debug=False):
    """
    Training module for T_HAN objectives
    
    Parameters
    ----------
    model: object of T_HAN
        initialized Time-Aware HAN model

    t_train: 2-D numpy array, shape (num_exemplars, num_bucket)
        variable indices all buckets and sections

    x_train: 2-D numpy array, shape (num_exemplars, num_bucket)
        variable indices all variables

    y_train: 2-D numpy array, shape (num_exemplars, num_classes)
        whole training ground truth
        
    dev_sample_percentage: float
        percentage of x_train seperated from training process and used for validation

    num_epochs: int
        number of epochs of training, one epoch means finishing training entire training set

    batch_size: int
        size of training batches, this won't affect training speed significantly; smaller batch leads to more regularization

    evaluate_every: int
        number of steps to perform a evaluation on development (validation) set and print out info

    model_path: str
        the path to store the model

    Examples
    --------
    >>> from enid.than_clf import train_han
    >>> train_han(model=rnn, t_train=T, x_train=X,
                y_train=y, dev_sample_percentage=0.01,
                num_epochs=20, batch_size=64,
                evaluate_every=100, model_path='./model/')
    """

    # get number of input exemplars
    training_size = y_train.shape[0]

    dev_sample_index = -1 * int(dev_sample_percentage * float(training_size))
    t_train, t_dev = t_train[:dev_sample_index], t_train[dev_sample_index:]
    x_train, x_dev = x_train[:dev_sample_index], x_train[dev_sample_index:]
    y_train, y_dev = y_train[:dev_sample_index], y_train[dev_sample_index:]
    training_size = y_train.shape[0]

    # initialize TensorFlow graph
    graph = tf.get_default_graph()
    with graph.as_default():

        # configurate TensorFlow session, enable GPU accelerated if possible
        config=tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth=True
        sess =  tf.Session(config=config)
        saver = tf.train.Saver()

        # initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if debug: sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        print('start time:', datetime.now())
        # create model root path if not exists
        if not os.path.exists(model_path): os.mkdir(model_path)

        writer_train = tf.summary.FileWriter(os.path.join(model_path, 'train'))
        writer_val = tf.summary.FileWriter(os.path.join(model_path, 'vali'))
        writer_train.add_graph(sess.graph)
        
        # get current epoch
        curr_epoch = sess.run(model.epoch_step)
        for epoch in range(curr_epoch, num_epochs):
            print('Epoch', epoch+1, '...')
            counter = 0

            # loop batch training
            for start, end in zip(range(0, training_size, batch_size), range(batch_size, training_size, batch_size)):
                epoch_x = x_train[start:end]
                epoch_t = t_train[start:end]
                epoch_y = y_train[start:end]

                # create model inputs
                feed_dict = {model.input_x: epoch_x, model.input_t: epoch_t, model.input_y: epoch_y}

                # train one step
                curr_loss, _, merged_sum = sess.run([model.loss_val, model.train_op, model.merged_sum], feed_dict)
                writer_train.add_summary(merged_sum, global_step=sess.run(model.global_step))
                counter = counter+1

                # evaluation
                if counter % evaluate_every == 0:
                    #train_accu = model.auc.eval(feed_dict, session=sess)
                    dev_loss, dev_accu = _do_eval(sess, model, x_dev, t_dev, y_dev, batch_size, writer_val)
                    print('Step: {: <5}  |  Loss: {:2.9f}  |  Development Loss: {:2.8f}  |  Development AUROC: {:2.8f}'.format(counter, curr_loss, dev_loss, dev_accu))
            sess.run(model.epoch_increment)

            # write model into disk at the end of each 10 epoch     if epoch > 9 and epoch % 10 == 9: 
            saver.save(sess, os.path.join(model_path, 'model'), global_step=model.global_step)
            print('='*100)
        sess.close()
        print('End time:', datetime.now())

def test_han(model_path, step=None, prob_norm='softmax', just_graph=False, **kwargs):
    """
    Testing module for T_HAN models
    
    Parameters
    ----------
    model_path: str
        the path to store the model
    
    step: int
        if not None, load specific model with given step

    prob_norm: str, default 'softmax'
        method to convert final layer scores into probabilities, either 'softmax' or 'sigmoid'

    just_graph: boolean, default False
        if False, just return tf graphs; if True, take input test data and return y_pred

    **kwargs: dict, optional
        Keyword arguments for test module, full documentation of parameters can be found in notes

    Notes
    ----------
    If just_graph is False, test_rnn should take input test data as follows:

    t_test: 2-D numpy array, shape (num_exemplars, num_bucket)
        variable indices all buckets and sections

    x_test: 2-D numpy array, shape (num_exemplars, num_bucket)
        variable indices all variables

    Returns
    ----------
    If just_graph=True:

        sess: tf.Session object
            tf Session for running TensorFlow operations

        t: tf.placeholder
            placeholder for t_test

        x: tf.placeholder
            placeholder for x_test

        y_pred: tf.placeholder
            placeholder for t_test

    If just_graph=False:

        y_probs: 1-D numpy array, shape (num_exemplar,)
            predicted target values based on trained model

    Examples
    --------
    >>> from enid.than_clf import test_han
    >>> sess, t, x, y_pred = test_han('./model', prob_norm='sigmoid', just_graph=True)
    >>> sess.run(y_pred, {x: x_test, t: t_test})
    array([[4.8457133e-03, 9.9515426e-01],
           [4.6948572e-03, 9.9530518e-01],
           [3.1738445e-02, 9.6826160e-01],
           ...,
           [1.0457519e-03, 9.9895418e-01],
           [5.6348788e-04, 9.9943644e-01],
           [5.9802778e-04, 9.9940193e-01]], dtype=float32)
    >>> sess.close()

    >>> from enid.than_clf import test_han
    >>> test_han('./model', t_test=T_test, x_test=X_test)
    array([9.9515426e-01,
           4.6948572e-03,
           3.1738445e-02,,
           ...,
           9.9895418e-01,
           5.6348788e-04,
           9.9940193e-01], dtype=float32)
    """

    # clear TensorFlow graph
    tf.reset_default_graph()
    sess = tf.Session()
    sess.as_default()

    # Recreate the network graph. At this step only graph is created.
    if step:
        saver = tf.train.import_meta_graph(os.path.join(model_path.rstrip('/'), 'model-'+str(step)+'.meta'))
        saver.restore(sess, os.path.join(model_path.rstrip('/'), 'model-'+str(step)))
    else:
        saver = tf.train.import_meta_graph(os.path.join(model_path.rstrip('/'), 'model.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()

    # restore graph names for predictions
    assert prob_norm in ['softmax', 'sigmoid'], 'AttributeError: prob_norm only acccept "softmax" or "sigmoid", got {}'.format(str(prob_norm))
    if prob_norm == 'softmax': y_pred = graph.get_tensor_by_name("output/probs:0")
    if prob_norm == 'sigmoid':
        y_score = graph.get_tensor_by_name("output/scores:0")
        y_pred = tf.sigmoid(y_score)
    x = graph.get_tensor_by_name("input_x:0")
    t = graph.get_tensor_by_name("input_t:0")

    if just_graph: return sess, t, x, y_pred
    else:
        number_examples = kwargs['t_test'].shape[0]
        if number_examples < 128:
            y_probs = sess.run(y_pred, {x: kwargs['x_test'], t: kwargs['t_test']})[:,0]
        else:
            y_probs = np.empty((0))
            for start, end in zip(range(0,number_examples,128), range(128,number_examples,128)):
                feed_dict = {x: kwargs['x_test'][start:end],
                             t: kwargs['t_test'][start:end]}
                probs = sess.run(y_pred, feed_dict)[:,0]
                y_probs = np.concatenate([y_probs, probs])
            feed_dict = {x: kwargs['x_test'][end:],
                         t: kwargs['t_test'][end:]}
            probs = sess.run(y_pred, feed_dict)[:,0]
            y_probs = np.concatenate([y_probs, probs])
        # cali = pickle.load(open(os.path.join(os.path.dirname(__file__), 'pickle_files','er_calibration'), 'rb'))
        sess.close()
        return y_probs

############################# PRIVATE FUNCTIONS ###############################

def _do_eval(sess, model, eval_x, eval_t, eval_y, batch_size, writer_val):
    """
    Evaluate development in batch (if direcly force sess run entire development set, may raise OOM error)
    """
    number_examples = len(eval_x)
    eval_loss, eval_counter = 0.0, 0
    eval_probs = np.empty((0, model.num_classes))
    for start, end in zip(range(0,number_examples,batch_size), range(batch_size,number_examples,batch_size)):
        feed_dict = {model.input_x: eval_x[start:end],
                     model.input_t: eval_t[start:end],
                     model.input_y: eval_y[start:end]}
        curr_eval_loss, curr_probs, merged_sum = sess.run([model.loss_val, model.probs, model.merged_sum], feed_dict)
        writer_val.add_summary(merged_sum, global_step=sess.run(model.global_step))
        
        eval_loss, eval_probs, eval_counter = eval_loss+curr_eval_loss, np.concatenate([eval_probs, curr_probs]), eval_counter+1
    feed_dict = {model.input_x: eval_x[end:],
                 model.input_t: eval_t[end:],
                 model.input_y: eval_y[end:]}
    curr_probs = sess.run(model.probs, feed_dict)
    eval_probs = np.concatenate([eval_probs, curr_probs])
    eval_acc = sess.run(model.auc, {model.input_y: eval_y, model.probs: eval_probs})
    return eval_loss/float(eval_counter), eval_acc