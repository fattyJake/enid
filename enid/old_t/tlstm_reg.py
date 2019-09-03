# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Time-Aware Recurrent Neural Network class with embedding
# Authors:     Yage Wang
# Created:     8.14.2018
###############################################################################

import os
import pickle
import tensorflow as tf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from enid.data_helper import Vectorizer, _get_variables


class T_LSTM(object):
    """
    A Time-Aware-LSTM for claim regressor
    Uses an embedding layer, followed by a time-aware-lstm, attention and sofrmax layer

    Parameters
    ----------
    max_sequence_length: int
        fixed padding latest number of time buckets

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

    initializer: tf tensor initializer object, default tf.random_normal_initializer(stddev=0.1)
        initializer for fully connected layer weights

    Examples
    --------
    >>> from enid.tlstm import T_LSTM
    >>> rnn = T_LSTM(max_sequence_length=200, hidden_size=128,
            embedding_dict=embedding_dict,
            w_embedding_dict=w_embedding_dict, learning_rate=0.05,
            decay_steps=5000, decay_rate=0.9,
            dropout_keep_prob=0.8, l2_reg_lambda=0.0)
    """

    def __init__(
            self,
            max_sequence_length,
            hidden_size,
            pretrain_embedding,
            learning_rate,
            decay_steps,
            decay_rate,
            dropout_keep_prob,
            l2_reg_lambda=0.0,
            objective='ce',
            initializer=tf.orthogonal_initializer()):
        """init all hyperparameter here"""
        tf.reset_default_graph()

        # set hyperparamter
        self.pretrain_embedding = pretrain_embedding
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate
        self.initializer = initializer

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(
            self.epoch_step, tf.add(
                self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        # add placeholder (X, quantity, time and label)
        self.input_x = tf.placeholder(
            tf.int32, [
                None, self.max_sequence_length], name="input_x")  # X [instance_size, num_bucket]
        self.input_t = tf.placeholder(
            tf.int32, [
                None, self.max_sequence_length], name="input_t")  # T [instance_size, num_bucket]
        self.input_y = tf.placeholder(
            tf.float32, [
                None, 1], name="input_y")  # y [instance_size, 1]

        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            #embedding_matrix = tf.truncated_normal((self.variable_size, self.embedding_size), stddev=1/np.sqrt(self.embedding_size))
            embedding_matrix = tf.concat([self.pretrain_embedding, tf.zeros(
                (1, self.pretrain_embedding.shape[1]))], axis=0)
            self.Embedding = tf.Variable(
                embedding_matrix,
                trainable=False,
                dtype=tf.float32,
                name='embedding')

        # main computation graph here: 1. embeddding layer, 2.T-LSTM layer, 3.concat, 4.FC layer 5.softmax
        # 1.get emebedding of tokens
        self.input = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # 2. Time-Aware-LSTM layer
        self.input_size = self.input.get_shape().as_list()[2]
        with tf.name_scope("weight"):
            self.Wi = self._init_weights(
                self.input_size,
                self.hidden_size,
                name='Input_Hidden_weight',
                reg=None)
            self.Ui = self._init_weights(
                self.hidden_size,
                self.hidden_size,
                name='Input_State_weight',
                reg=None)
            self.bi = self._init_bias(
                self.hidden_size, name='Input_Hidden_bias')

            self.Wf = self._init_weights(
                self.input_size,
                self.hidden_size,
                name='Forget_Hidden_weight',
                reg=None)
            self.Uf = self._init_weights(
                self.hidden_size,
                self.hidden_size,
                name='Forget_State_weight',
                reg=None)
            self.bf = self._init_bias(
                self.hidden_size, name='Forget_Hidden_bias')

            self.Wog = self._init_weights(
                self.input_size,
                self.hidden_size,
                name='Output_Hidden_weight',
                reg=None)
            self.Uog = self._init_weights(
                self.hidden_size,
                self.hidden_size,
                name='Output_State_weight',
                reg=None)
            self.bog = self._init_bias(
                self.hidden_size, name='Output_Hidden_bias')

            self.Wc = self._init_weights(
                self.input_size,
                self.hidden_size,
                name='Cell_Hidden_weight',
                reg=None)
            self.Uc = self._init_weights(
                self.hidden_size,
                self.hidden_size,
                name='Cell_State_weight',
                reg=None)
            self.bc = self._init_bias(
                self.hidden_size, name='Cell_Hidden_bias')

            self.W_decomp = self._init_weights(
                self.hidden_size,
                self.hidden_size,
                name='Decomposition_Hidden_weight',
                reg=None)
            self.b_decomp = self._init_bias(
                self.hidden_size, name='Decomposition_Hidden_bias_enc')

            self.W_final = self._init_weights(
                self.hidden_size,
                1,
                name='Output_Layer_weight',
                reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.b_final = self._init_bias(1, name='Output_Layer_bias')

        with tf.name_scope("tlstm"):
            batch_size = tf.shape(self.input)[0]
            scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
            # scan input is [seq_length x batch_size x input_size]
            scan_input = tf.transpose(scan_input_)
            # scan_time [seq_length x batch_size]
            scan_time = tf.transpose(self.input_t)
            initial_hidden = tf.zeros(
                [batch_size, self.hidden_size], tf.float32)
            ini_state_cell = tf.stack([initial_hidden, initial_hidden])

            # make scan_time [seq_length x batch_size x 1]
            scan_time = tf.reshape(
                scan_time, [
                    tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
            # [seq_length x batch_size x input_size+1]
            concat_input = tf.concat(
                [tf.cast(scan_time, tf.float32), scan_input], 2)
            packed_hidden_states = tf.scan(
                self._TLSTM_Unit, concat_input, initializer=ini_state_cell)
            all_states = packed_hidden_states[:, 0, :, :]
            all_states = tf.identity(all_states, name='hidden_states')

            # attention layer
            attention_output = self._attention(
                all_states, self.hidden_size, time_major=True, return_alphas=False)

        with tf.name_scope("output"):
            self.predictions = tf.nn.xw_plus_b(
                attention_output, self.W_final, self.b_final, name='predictions')

        self.loss_val = tf.identity(
            self._loss(
                self.l2_reg_lambda),
            name='loss')
        self.train_op = self._train()

        # performance
        with tf.name_scope("performance"):
            _, self.mae = tf.metrics.mean_absolute_error(
                self.input_y, self.predictions, name="mae")

        self.loss_sum = tf.summary.scalar("loss_mae_train", self.loss_val)
        self.learning_rate_sum = tf.summary.scalar(
            "learning_rate", self.learning_rate)
        self.prediction_sum = tf.summary.histogram(
            "predictions", self.predictions)
        self.attention_sum = tf.summary.histogram(
            "attentions", attention_output)
        self.merged_sum = tf.summary.merge_all()

    def _init_weights(self, input_size, output_dim, name, std=0.1, reg=None):
        return tf.nn.dropout(
            tf.get_variable(
                name,
                shape=[
                    input_size,
                    output_dim],
                initializer=self.initializer,
                regularizer=reg),
            self.dropout_keep_prob)

    def _init_bias(self, output_dim, name):
        return tf.get_variable(
            name,
            shape=[output_dim],
            initializer=tf.constant_initializer(0.1))

    def _map_elapse_time(self, t):
        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)
        # T = tf.multiply(self.wt, t) + self.bt
        # according to paper, used for large time delta like days
        T = tf.div(c1, tf.log(t + c2), name='Log_elapse_time')
        Ones = tf.ones([1, self.hidden_size], dtype=tf.float32)
        T = tf.matmul(T, Ones)
        return T

    def _TLSTM_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_size])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        # Dealing with time irregularity
        # Map elapse time in days or months
        T = self._map_elapse_time(t)

        # Decompose the previous cell if there is a elapse time
        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis

        i = tf.sigmoid(
            tf.matmul(
                x,
                self.Wi) +
            tf.matmul(
                prev_hidden_state,
                self.Ui) +
            self.bi)  # Input gate
        f = tf.sigmoid(
            tf.matmul(
                x,
                self.Wf) +
            tf.matmul(
                prev_hidden_state,
                self.Uf) +
            self.bf)  # Forget Gate
        o = tf.sigmoid(
            tf.matmul(
                x,
                self.Wog) +
            tf.matmul(
                prev_hidden_state,
                self.Uog) +
            self.bog)  # Output Gate

        C = tf.nn.tanh(
            tf.matmul(
                x,
                self.Wc) +
            tf.matmul(
                prev_hidden_state,
                self.Uc) +
            self.bc)  # Candidate Memory Cell
        Ct = f * prev_cell + i * C  # Current Memory cell
        current_hidden_state = o * tf.nn.tanh(Ct)  # Current Hidden state

        return tf.stack([current_hidden_state, Ct])

    def _get_output(self, state):
        output = tf.nn.tanh(tf.matmul(state, self.Wo) + self.bo)
        output = tf.matmul(output, self.W_softmax) + self.b_softmax
        return output

    def _attention(
            self,
            inputs,
            attention_size,
            time_major=False,
            return_alphas=False):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
         for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Variables notation is also inherited from the article

        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
                    In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor of shape:
                            `[max_time, batch_size, cell.output_size]`.
                    In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
                        If time_major == False (default),
                            outputs_fw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_bw.output_size]`.
                        If time_major == True,
                            outputs_fw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_bw.output_size]`.
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
        """

        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN
            # outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, [1, 0, 2])

        # D value - hidden size of the RNN layer
        hidden_size = inputs.shape[2].value

        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal(
            [hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced
        # with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        # alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has
        # (B,D) shape
        output = tf.reduce_mean(
            inputs * tf.expand_dims(vu, -1), 1, name='attention_output')

        if not return_alphas:
            return output
        else:
            return output  # , alphas

    def _train(self):
        """
        based on the loss, use Adam to update parameter
        """
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.decay_rate,
            staircase=True)
        train_op = tf.contrib.layers.optimize_loss(
            self.loss_val,
            global_step=self.global_step,
            learning_rate=learning_rate,
            optimizer="Adam")
        return train_op

    def _loss(self, l2_reg_lambda):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, 1]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as
            # `logits` with the softmax cross entropy loss.
            loss = tf.losses.mean_squared_error(
                predictions=self.predictions, labels=self.input_y)
            l2_losses = tf.add_n([tf.nn.l2_loss(
                v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_reg_lambda
        return loss + l2_losses


def train_rnn(
        model,
        t_train,
        x_train,
        y_train,
        dev_sample_percentage,
        num_epochs,
        batch_size,
        evaluate_every,
        model_path):
    """
    Training module for T_LSTM objectives

    Parameters
    ----------
    model: object of T_LSTM
        initialized Phased LSTM model

    t_train: 2-D numpy array, shape (num_exemplars, num_bucket)
        variable indices all buckets and sections

    x_train: 2-D numpy array, shape (num_exemplars, num_bucket)
        variable indices all variables

    y_train: 2-D numpy array, shape (num_exemplars, 1)
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
    >>> from enid.tlstm import train_rnn
    >>> train_rnn(model=rnn, t_train=T, x_train=X,
                y_train=y, dev_sample_percentage=0.01,
                num_epochs=20, batch_size=64,
                evaluate_every=100, model_path='./plstm_model/')
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
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            # initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            print('start time:', datetime.now())
            # create model root path if not exists
            if not os.path.exists(model_path):
                os.mkdir(model_path)

            writer = tf.summary.FileWriter(os.path.join(model_path))
            writer.add_graph(sess.graph)

            # get current epoch
            curr_epoch = sess.run(model.epoch_step)
            for epoch in range(curr_epoch, num_epochs):
                print('Epoch', epoch + 1, '...')
                counter = 0

                # loop batch training
                for start, end in zip(
                    range(
                        0, training_size, batch_size), range(
                        batch_size, training_size, batch_size)):
                    epoch_x = x_train[start:end]
                    epoch_t = t_train[start:end]
                    epoch_y = y_train[start:end]

                    # create model inputs
                    feed_dict = {
                        model.input_x: epoch_x,
                        model.input_t: epoch_t,
                        model.input_y: epoch_y}

                    # train one step
                    curr_loss, _, merged_sum = sess.run(
                        [model.loss_val, model.train_op, model.merged_sum], feed_dict)
                    writer.add_summary(
                        merged_sum, global_step=sess.run(
                            model.global_step))
                    counter = counter + 1

                    # evaluation
                    if counter % evaluate_every == 0:
                        train_accu = model.mae.eval(feed_dict)
                        dev_accu = _do_eval(
                            sess, model, x_dev, t_dev, y_dev, batch_size)
                        print(
                            'Step: {: <6}  |  Loss: {:2.10f}  |  Training MAE: {:2.10f}  |  Development MAE: {:2.10f}'.format(
                                counter, curr_loss, train_accu, dev_accu))
                sess.run(model.epoch_increment)

                # write model into disk at the end of each 10 epoch     if
                # epoch > 9 and epoch % 10 == 9:
                saver.save(
                    sess,
                    os.path.join(
                        model_path,
                        'model'),
                    global_step=model.global_step)
                print('=' * 100)
            print('End time:', datetime.now())


def test_rnn(model_path, step=None, just_graph=False, **kwargs):
    """
    Testing module for T_LSTM models

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
    >>> from enid.tlstm import test_rnn
    >>> sess, t, x, y_pred = test_rnn('./plstm_model', prob_norm='sigmoid', just_graph=True)
    >>> sess.run(y_pred, {x: x_test, t: t_test})
    array([[4.8457133e-03, 9.9515426e-01],
           [4.6948572e-03, 9.9530518e-01],
           [3.1738445e-02, 9.6826160e-01],
           ...,
           [1.0457519e-03, 9.9895418e-01],
           [5.6348788e-04, 9.9943644e-01],
           [5.9802778e-04, 9.9940193e-01]], dtype=float32)
    >>> sess.close()

    >>> from enid.tlstm import test_rnn
    >>> test_rnn('./plstm_model', t_test=T_test, x_test=X_test)
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
    cost_norm = pickle.load(
        open(
            os.path.join(
                os.path.dirname(__file__),
                'pickle_files',
                'cost_norm_q'),
            'rb'))

    # Recreate the network graph. At this step only graph is created.
    if step:
        saver = tf.train.import_meta_graph(
            os.path.join(
                model_path.rstrip('/'),
                'model-' + str(step) + '.meta'))
        saver.restore(
            sess,
            os.path.join(
                model_path.rstrip('/'),
                'model-' +
                str(step)))
    else:
        saver = tf.train.import_meta_graph(
            os.path.join(model_path.rstrip('/'), 'model.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()

    # restore graph names for predictions
    y_pred = graph.get_tensor_by_name("output/predictions:0")
    x = graph.get_tensor_by_name("input_x:0")
    t = graph.get_tensor_by_name("input_t:0")

    if just_graph:
        return sess, t, x, y_pred
    else:
        number_examples = kwargs['t_test'].shape[0]
        output = np.empty((0, 1))
        for start, end in zip(
            range(
                0, number_examples, 128), range(
                128, number_examples, 128)):
            feed_dict = {x: kwargs['x_test'][start:end],
                         t: kwargs['t_test'][start:end]}
            preds = sess.run(y_pred, feed_dict)
            output = np.concatenate([output, preds])
        feed_dict = {x: kwargs['x_test'][end:],
                     t: kwargs['t_test'][end:]}
        preds = sess.run(y_pred, feed_dict)
        output = np.concatenate([output, preds])
        output = cost_norm.inverse_transform(output)[:, 0]
        sess.close()
        return output


def interpret(model_path, step, t_, x_, label):
    if len(t_.shape) == 1:
        t_, x_ = np.expand_dims(t_, 0), np.expand_dims(x_, 0)

    tf.reset_default_graph()
    sess = tf.Session()
    sess.as_default()

    # Recreate the network graph. At this step only graph is created.
    if step:
        saver = tf.train.import_meta_graph(
            os.path.join(
                model_path.rstrip('/'),
                'model-' + str(step) + '.meta'))
        saver.restore(
            sess,
            os.path.join(
                model_path.rstrip('/'),
                'model-' +
                str(step)))
    else:
        saver = tf.train.import_meta_graph(
            os.path.join(model_path.rstrip('/'), 'model.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()

    states = graph.get_tensor_by_name("tlstm/vu:0")
    x = graph.get_tensor_by_name("input_x:0")
    t = graph.get_tensor_by_name("input_t:0")

    code_desc = pickle.load(
        open(
            os.path.join(
                os.path.dirname(__file__),
                'pickle_files',
                'codes'),
            'rb'))
    output = sess.run(states, {x: x_, t: t_})[:, -50:].T
    vec = Vectorizer()
    tokens = [_get_variables(c, vec) for c in x_[0][-50:]]
    tokens = [t + '  ' + code_desc[t] if t in code_desc else t for t in tokens]

    fig, ax = plt.subplots(figsize=(2, 12), dpi=120)
    sns.heatmap(
        output,
        xticklabels=[label],
        cmap='coolwarm',
        yticklabels=tokens,
        vmin=-1,
        vmax=2,
        ax=ax)
    sess.close()

############################# PRIVATE FUNCTIONS ###############################


def _do_eval(sess, model, eval_x, eval_t, eval_y, batch_size):
    """
    Evaluate development in batch (if direcly force sess run entire development set, may raise OOM error)
    """
    number_examples = len(eval_x)
    eval_acc, eval_counter = 0.0, 0
    for start, end in zip(
        range(
            0, number_examples, batch_size), range(
            batch_size, number_examples, batch_size)):
        feed_dict = {model.input_x: eval_x[start:end],
                     model.input_t: eval_t[start:end],
                     model.input_y: eval_y[start:end]}
        curr_eval_acc = model.mae.eval(feed_dict)

        eval_acc, eval_counter = eval_acc + curr_eval_acc, eval_counter + 1
    return eval_acc / float(eval_counter)
