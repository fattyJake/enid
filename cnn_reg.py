# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Convolutional Neural Network class with embedding
# Authors:     Yage Wang (credit to Y. Kim https://arxiv.org/abs/1408.5882)
# Created:     9.5.2018
###############################################################################

import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import pickle

class CNN(object):
    """
    A CNN for sequence classification
    Uses an embedding layer, followed by a convolutional, max-pooling and sofrmax layer
    """

    def __init__(
        self, sequence_length, pretrain_embedding, filter_sizes, num_filters, learning_rate,
        decay_steps, decay_rate, dropout_keep_prob, l2_reg_lambda=0.0, objective='ce'):
        """
        Convolutional Neural Network model graph

        Parameters
        ----------        
        sequence_length: int
            the dimensionality of original vector input

        num_classes: int
            the number of y classes

        vocab_size: int
            the unique number of total member observations

        embedding_size: int
            the dimensionality after embedding for Conv_Layer input

        pre_trained_w2v: 2-D numpy array, shape (vocab_size, embedding_size) or NoneType
            pre-trained word embeddings for embedding_lookup; if None, initialize and train inside the model

        filter_sizes: list of int, recommend [3,4,5] according to paper
            each representing the window size of each convolution filter (kernel)

        num_filters: int
            number of filters for each convolutional layer

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

        Examples
        --------
        >>> from georges.text_cnn import TextCNN
        >>> cnn = TextCNN(sequence_length=128, num_classes=2,
                    vocab_size=1000000, embedding_size=200,
                    pre_trained_w2v=None, filter_sizes=[3,4,5],
                    num_filters=128, learning_rate=0.001,
                    decay_steps=5000, decay_rate=0.9,
                    dropout_keep_prob=0.5, l2_reg_lambda=0.02,
                    objective='ce')
        """

        # set hyperparamter
        tf.reset_default_graph()

        self.sequence_length = sequence_length
        self.pretrain_embedding = pretrain_embedding
        self.embedding_size = pretrain_embedding.shape[1]
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        # Placeholders for input, output
        self.input_x = tf.placeholder(tf.int32,   [None, self.sequence_length], name="input_x") # X [instance_size, num_bucket]
        self.input_y = tf.placeholder(tf.float32, [None, 1],                    name="input_y") # y [instance_size, 1]

        # Keep track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"): # embedding matrix
            #embedding_matrix = tf.truncated_normal((self.variable_size, self.embedding_size), stddev=1/np.sqrt(self.embedding_size))
            embedding_matrix = tf.concat([self.pretrain_embedding, tf.zeros((1, self.pretrain_embedding.shape[1]))], axis=0)
            self.Embedding = tf.Variable(embedding_matrix, trainable=False, dtype=tf.float32, name='embedding')

            self.embedded_vals = tf.nn.embedding_lookup(self.Embedding, self.input_x)
            self.embedded_vals_expanded = tf.expand_dims(self.embedded_vals, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_vals_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.predictions = tf.nn.xw_plus_b(self.h_drop, W, b, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.loss_val = tf.identity(self._loss(self.l2_reg_lambda), name='loss')
            self.train_op = self._train()

        # Accuracy
        with tf.name_scope("performance"):
            _, self.mae = tf.metrics.mean_absolute_error(self.input_y, self.predictions, name="mae")

        self.loss_sum = tf.summary.scalar("loss_mae_train", self.loss_val)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)
        self.prediction_sum = tf.summary.histogram("predictions", self.predictions)
        self.merged_sum = tf.summary.merge_all()

    def _train(self):
        """
        based on the loss, use Adam to update parameter
        """
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam")
        return train_op

    def _loss(self, l2_reg_lambda):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, 1]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            loss = tf.losses.mean_squared_error(predictions=self.predictions, labels=self.input_y)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_reg_lambda
        return loss + l2_losses

def train_cnn(model, x_train, y_train, dev_sample_percentage, num_epochs, batch_size, evaluate_every, model_path, vocab_processor=None, pre_trained_embedding=None):
    """
    Training module for TextCNN objectives
    
    Parameters
    ----------
    model: object of TextCNN
        initialized CNN model

    x_train: 2-D numpy array, shape (n_exemplars, sequence_length)
        whole training input

    y_train: 2-D numpy array, shape (n_exemplars, num_classes)
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
    >>> from georges.text_cnn import train_cnn
    >>> train_cnn(model=cnn,
            x_train=X_train, y_train=y_train,
            dev_sample_percentage=0.005, num_epochs=20,
            batch_size=64, evaluate_every=200,
            model_path='./cnn_model/')
    """

    # get number of input exemplars
    training_size = y_train.shape[0]

    dev_sample_index = -1 * int(dev_sample_percentage * float(training_size))
    x_train, x_dev = x_train[:dev_sample_index], x_train[dev_sample_index:]
    y_train, y_dev = y_train[:dev_sample_index], y_train[dev_sample_index:]
    training_size = y_train.shape[0]

    # initialize TensorFlow graph
    graph = tf.get_default_graph()
    with graph.as_default():

        # configurate TensorFlow session, enable GPU accelerated if possible
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            # initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            print('start time:', datetime.now())
            # create model root path if not exists
            if not os.path.exists(model_path): os.mkdir(model_path)

            writer = tf.summary.FileWriter(os.path.join(model_path))
            writer.add_graph(sess.graph)

            # get current epoch
            curr_epoch = sess.run(model.epoch_step)
            for epoch in range(curr_epoch, num_epochs):
                print('Epoch', epoch+1, '...')
                counter = 0

                # loop batch training
                for start, end in zip(range(0, training_size, batch_size), range(batch_size, training_size, batch_size)):
                    epoch_x = x_train[start:end]
                    epoch_y = y_train[start:end]

                    # create model inputs
                    feed_dict = {model.input_x: epoch_x}
                    feed_dict[model.input_y] = epoch_y

                    # train one step
                    curr_loss, _, merged_sum = sess.run([model.loss_val, model.train_op, model.merged_sum], feed_dict)
                    writer.add_summary(merged_sum, global_step=sess.run(model.global_step))
                    counter = counter+1

                    # evaluation
                    if counter % evaluate_every == 0:
                        train_accu = model.mae.eval(feed_dict)
                        dev_accu = _do_eval(sess, model, x_dev, y_dev, batch_size)
                        print('Step: {: <6}  |  Loss: {:2.10f}  |  Training MAE: {:2.10f}  |  Development MAE: {:2.10f}'.format(counter, curr_loss, train_accu, dev_accu))
                sess.run(model.epoch_increment)

                # write model into disk at the end of each epoch
                saver.save(sess, os.path.join(model_path, 'model'), global_step=model.global_step)
                print('='*100)

            print('End time:', datetime.now())

def test_cnn(model_path, step=None, just_graph=False, **kwargs):
    """
    Testing module for TextCNN models
    
    Parameters
    ----------
    model_path: str
        the path to store the model

    prob_norm: str, default 'softmax'
        method to convert final layer scores into probabilities, either 'softmax' or 'sigmoid'

    Examples
    --------
    >>> from georges.text_cnn import test_cnn
    >>> sess, x, y_pred = test_cnn('./cnn_model', prob_norm='sigmoid')
    >>> sess.run(y_pred, {x: vectors})
    array([[4.8457133e-03, 9.9515426e-01],
           [4.6948572e-03, 9.9530518e-01],
           [3.1738445e-02, 9.6826160e-01],
           ...,
           [1.0457519e-03, 9.9895418e-01],
           [5.6348788e-04, 9.9943644e-01],
           [5.9802778e-04, 9.9940193e-01]], dtype=float32)
    >>> sess.close()
    """

    # clear TensorFlow graph
    tf.reset_default_graph()
    sess = tf.Session()
    sess.as_default()
    cost_norm = pickle.load(open(os.path.join(os.path.dirname(__file__),'pickle_files','cost_norm_q'), 'rb'))

    # Recreate the network graph. At this step only graph is created.
    if step:
        saver = tf.train.import_meta_graph(os.path.join(model_path.rstrip('/'), 'model-'+str(step)+'.meta'))
        saver.restore(sess, os.path.join(model_path.rstrip('/'), 'model-'+str(step)))
    else:
        saver = tf.train.import_meta_graph(os.path.join(model_path.rstrip('/'), 'model.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()

    # restore graph names for predictions
    y_pred = graph.get_tensor_by_name("output/predictions:0")
    x = graph.get_tensor_by_name("input_x:0")

    if just_graph: return sess, x, y_pred
    else:
        number_examples = kwargs['x_test'].shape[0]
        output = np.empty((0, 1))
        for start, end in zip(range(0,number_examples,128), range(128,number_examples,128)):
            feed_dict = {x: kwargs['x_test'][start:end]}
            preds = sess.run(y_pred, feed_dict)
            output = np.concatenate([output, preds])
        feed_dict = {x: kwargs['x_test'][end:]}
        preds = sess.run(y_pred, feed_dict)
        output = np.concatenate([output, preds])
        output = cost_norm.inverse_transform(output)[:, 0]
        sess.close()
        return output

def _do_eval(sess, model, eval_x, eval_y, batch_size):
    """
    Evaluate development in batch (if direcly force sess run entire development set, may raise OOM error)
    """
    number_examples = len(eval_x)
    eval_acc, eval_counter = 0.0, 0
    for start, end in zip(range(0,number_examples,batch_size), range(batch_size,number_examples,batch_size)):
        feed_dict = {model.input_x: eval_x[start:end],
                     model.input_y: eval_y[start:end]}
        curr_eval_acc = model.mae.eval(feed_dict)
        
        eval_acc,eval_counter = eval_acc+curr_eval_acc, eval_counter+1
    return eval_acc/float(eval_counter)