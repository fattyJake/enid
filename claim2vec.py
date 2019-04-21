# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Claim2Vec
# Authors:     Yage Wang
# Created:     4.15.2019
###############################################################################

import os
import math
import random
import pickle

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

class Claim2Vec(object):
    """
    Medical claim embedding training mimic Word2Vec skipgram
    Unlike Word2Vec assuming nearest token are related, due to claims are only attached with
    date but no timestamp, the model assume all the claims within a day are related

    Parameters
    ----------
    dictionary : dict
        fixed indexing of embeddings with format of {index int: claim str}

    batch_size : int, optional (default 128)
        number of tokens of each training batch

    embedding_size : int, optional (default 256)
        dimentionality of embedding vectors

    learning_rate : float, optional (default 0.1)
        initial leaerning rate of gradient descent optimizer

    decay_steps : int, optional (default 50000)
        step frequency to decay the learning rate. e.g. if 5000, model will reduce learning rate by decay_rate every 5000 trained batches

    decay_rate : float, optional (default 0.95)
        percentage of learning rate decay rate

    num_sampled : int, optional (default 64)
        size of negative samling

    valid_size : float, optional (default 16)
        Random set of words to evaluate similarity on.

    Examples
    --------
    >>> from enid.claim2vec import Claim2Vec
    >>> c2v = Claim2Vec(d)
    """
    def __init__(self, dictionary, batch_size=128, embedding_size=256, learning_rate=0.1,
                 decay_steps=50000, decay_rate=0.95, num_sampled=64, valid_size=16):

        self.dictionary = dictionary
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.valid_size = valid_size
        self.vocabulary_size = len(dictionary)

        self.data_index = 0
        self.group_index = 0
        self.epoch = 0
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.step = tf.Variable(0, trainable=False, name="Global_Step")

            # Input data.
            with tf.name_scope('inputs'):
                self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
                self.valid_dataset = tf.placeholder(tf.int32, shape=[self.valid_size])

            # Ops and variables pinned to the CPU because of missing GPU implementation
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name='embedding')
                self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
               self.nce_weights = tf.Variable(
                            tf.truncated_normal(
                                    [self.vocabulary_size, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
            with tf.name_scope('biases'):
                self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                                tf.nn.nce_loss(
                                    weights=self.nce_weights,
                                    biases=self.nce_biases,
                                    labels=self.train_labels,
                                    inputs=self.embed,
                                    num_sampled=self.num_sampled,
                                    num_classes=self.vocabulary_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', self.loss)

            # Construct the SGD optimizer
            with tf.name_scope('optimizer'):
                lr = tf.train.exponential_decay(self.learning_rate,
                                                global_step=self.step,
                                                decay_steps=self.decay_steps,
                                                decay_rate=self.decay_rate)
                self.optimizer = tf.train.GradientDescentOptimizer(lr)
                self.add_global = self.step.assign_add(1)
                with tf.control_dependencies([self.add_global]):
                    self.train_op = self.optimizer.minimize(self.loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            self.normalized_embeddings = tf.identity(self.embeddings / self.norm, name='final_embedding')
            self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
            self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True)

            # Merge all summaries.
            self.merged = tf.summary.merge_all()

            # Add variable initializer.
            self.init = tf.global_variables_initializer()

            # Create a saver.
            self.saver = tf.train.Saver()

    def train(self, data, num_epoch, log_dir, evaluate_every=2000):
        """
        Train Claim2Vec object

        Parameters
        ----------
        data : list
            training index data with format of [[1,573,203], [16389, 8792], ... [0, 4, 8394, 20094]]
            the index should be consistent with dictionary

        num_epoch : int
            number of epoch to go over whole dataset

        log_dir : str
            directory of model logs

        evaluate_every : int, optional (default: 2000)
            how many steps the model evluates loss (evaluate sampled validation set on every evaluate_every*10 steps)

        Examples
        --------
        >>> from enid.claim2vec import Claim2Vec
        >>> c2v = Claim2Vec(d)
        >>> c2v.train(dataset, 5, 'c2v_logs', 5000)
        """
        data = [l for l in data if len(l) > 1]
    
        # configurate TensorFlow session, enable GPU accelerated if possible
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        session =  tf.Session(config=config, graph=self.graph)
        
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(log_dir, session.graph)
    
        # We must initialize all variables before we use them.
        self.init.run(session=session)
        print('Initialized')
    
        average_loss = 0
        while self.epoch < num_epoch:
            batch_inputs, batch_labels = self._generate_batch(data)
            feed_dict = {self.train_inputs: batch_inputs,
                         self.train_labels: batch_labels}
    
            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
            # Feed metadata variable to session for visualizing the graph in TensorBoard.
            _, summary, loss_val = session.run(
                    [self.train_op, self.merged, self.loss],
                    feed_dict=feed_dict)
            average_loss += loss_val
            step = session.run(self.step)
    
            # Add returned summaries to writer in each step.
            writer.add_summary(summary, global_step=step)
    
            if step % evaluate_every == 0:
                if step > 0:
                    average_loss /= evaluate_every
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0
    
            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % (evaluate_every*10) == 0:
                valid_examples = np.random.choice(len(self.dictionary)-1, self.valid_size, replace=False)
                sim = self.similarity.eval(feed_dict={self.valid_dataset: valid_examples}, session=session)
                for i in xrange(self.valid_size):
                    valid_word = self.dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = self.dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)

        final_embeddings = self.normalized_embeddings.eval(session=session)
        pickle.dump(final_embeddings, open(os.path.join(log_dir, 'embeddings.pkl'), 'wb'))
    
        # Write corresponding labels for the embeddings.
        with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
            for i in xrange(self.vocabulary_size):
                f.write(self.dictionary[i] + '\n')
    
        # Save the model for checkpoints.
        self.saver.save(session, os.path.join(log_dir, 'model.ckpt'))
    
        # Create a configuration for visualizing embeddings with the labels in TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = self.embeddings.name
        embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)
    
        writer.close()
        session.close()

    ###########################  PRIVATE FUNCTIONS  ###########################

    # Function to generate a training batch for the skip-gram model.
    def _generate_batch(self, data):
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        
        if self.data_index == len(data[self.group_index]):
            self.data_index = 0
            self.group_index += 1
        
        # locate group
        data_ = data[self.group_index]
        i = 0
        while True:
            input_ = data_[self.data_index]
            if len(data_)-1 > self.batch_size-i: context_words = random.sample([w for w in data_ if w != input_], self.batch_size-i)
            else: context_words = [w for w in data_ if w != input_]
            for context_word in context_words:
                batch[i] = input_
                labels[i, 0] = context_word
                i += 1
                if i >= self.batch_size:
                    self.data_index += 1
                    return batch, labels
            self.data_index += 1
            if self.data_index == len(data_):
                if self.group_index < len(data)-1: self.group_index += 1
                else:
                    self.group_index = 0
                    self.epoch += 1
                data_ = data[self.group_index]
                self.data_index = 0

#def generate_batch(batch_size, num_skips, skip_window):
#    global data_index, group_index, epoch
#    assert batch_size % num_skips == 0
#    assert num_skips <= 2 * skip_window
#    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
#    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
#    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
#    
#    # locate group
#    data_ = data[group_index]
#    if data_index + span > len(data_):
#        data_index = 0
#    buffer.extend(data_[data_index:data_index + span])
#    data_index += span
#    for i in range(batch_size // num_skips):
#        context_words = [w for w in range(span) if w != skip_window]
#        words_to_use = random.sample(context_words, num_skips)
#        for j, context_word in enumerate(words_to_use):
#            batch[i * num_skips + j] = buffer[skip_window]
#            labels[i * num_skips + j, 0] = buffer[context_word]
#        if data_index == len(data_):
#            if group_index < len(data)-1: group_index += 1
#            else:
#                group_index = 0
#                epoch += 1
#            data_ = data[group_index]
#            while len(data_) < span:
#                if group_index < len(data)-1: group_index += 1
#                else:
#                    group_index = 0
#                    epoch += 1
#                data_ = data[group_index]
#            buffer.extend(data_[0:span])
#            data_index = span
#        else:
#            buffer.append(data_[data_index])
#            data_index += 1
#    # Backtrack a little bit to avoid skipping words in the end of a batch
#    data_index = (data_index + len(data_) - span) % len(data_)
#    return batch, labels
