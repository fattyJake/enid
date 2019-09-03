# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Claim2Vec
# Authors:     Yage Wang
# Created:     4.15.2019
###############################################################################

import os
import math
import collections
import pickle

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from .claim2vec_op import SkipGram


class Claim2Vec(object):
    """
    Medical claim embedding training mimic Word2Vec skipgram
    Unlike Word2Vec assuming nearest token are related, due to claims are only
    attached with date but no timestamp, the model assume all the claims within
    a day are related

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
        step frequency to decay the learning rate. e.g. if 5000, model will
        reduce learning rate by decay_rate every 5000 trained batches

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

    def __init__(
            self,
            dictionary,
            batch_size=128,
            embedding_size=256,
            learning_rate=0.1,
            decay_steps=50000,
            decay_rate=0.95,
            num_sampled=64,
            valid_size=16):

        self.dictionary = dictionary
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.valid_size = valid_size
        self.vocabulary_size = len(dictionary)

        # self.data_index = 0
        # self.group_index = 0
        # self.epoch = 0
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.step = tf.Variable(0, trainable=False, name="Global_Step")

            # Input data.
            with tf.name_scope('inputs'):
                self.train_inputs = tf.placeholder(
                    tf.int32, shape=[self.batch_size])
                self.train_labels = tf.placeholder(
                    tf.int32, shape=[self.batch_size, 1])
                self.valid_dataset = tf.placeholder(
                    tf.int32, shape=[self.valid_size])

            # Ops and variables pinned to the CPU because of missing GPU implementation
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                self.embeddings = tf.Variable(tf.random_uniform(
                    [self.vocabulary_size, self.embedding_size], -1.0, 1.0),
                    name='embedding')
                self.embed = tf.nn.embedding_lookup(
                    self.embeddings, self.train_inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                self.nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [self.vocabulary_size, self.embedding_size],
                        stddev=1.0 / math.sqrt(self.embedding_size)))
            with tf.name_scope('biases'):
                self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative
            # labels each time we evaluate the loss.
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
                self.lr_sum = tf.summary.scalar("learning_rate", lr)
                self.optimizer = tf.train.GradientDescentOptimizer(lr)
                self.add_global = self.step.assign_add(1)
                with tf.control_dependencies([self.add_global]):
                    self.train_op = self.optimizer.minimize(self.loss)

            # Compute the cosine similarity between minibatch examples and all
            # embeddings.
            self.norm = tf.sqrt(
                tf.reduce_sum(
                    tf.square(
                        self.embeddings),
                    1,
                    keepdims=True))
            self.normalized_embeddings = tf.identity(
                self.embeddings / self.norm, name='final_embedding')
            self.valid_embeddings = tf.nn.embedding_lookup(
                self.normalized_embeddings, self.valid_dataset)
            self.similarity = tf.matmul(
                self.valid_embeddings,
                self.normalized_embeddings,
                transpose_b=True)

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
            training index data with format of [[1,573,203], [16389,8792], ...
            [0,4,8394,20094]]
            the index should be consistent with dictionary

        num_epoch : int
            number of epoch to go over whole dataset

        log_dir : str
            directory of model logs

        evaluate_every : int, optional (default: 2000)
            how many steps the model evluates loss (evaluate sampled validation
            set on every evaluate_every*10 steps)

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
        session = tf.Session(config=config, graph=self.graph)

        # load skipgram opt
        sg = SkipGram(self.batch_size)

        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(log_dir, session.graph)

        # We must initialize all variables before we use them.
        self.init.run(session=session)
        print('Initialized')

        average_loss = 0
        while sg.get_epoch() < num_epoch:
            batch_inputs, batch_labels = sg(data)  # self._generate_batch(data)
            feed_dict = {self.train_inputs: batch_inputs,
                         self.train_labels: batch_labels}

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
                # The average loss is an estimate of the loss over the last
                # 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500
            # steps)
            if step % (evaluate_every * 10) == 0:
                valid_examples = np.random.choice(
                    len(self.dictionary) - 1, self.valid_size, replace=False)
                sim = self.similarity.eval(
                    feed_dict={
                        self.valid_dataset: valid_examples},
                    session=session)
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
        pickle.dump(
            final_embeddings,
            open(
                os.path.join(
                    log_dir,
                    'embeddings.pkl'),
                'wb'))

        # Write corresponding labels for the embeddings.
        with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
            for i in xrange(self.vocabulary_size):
                f.write(self.dictionary[i] + '\n')

        # Save the model for checkpoints.
        self.saver.save(session, os.path.join(log_dir, 'model.ckpt'))

        writer.close()
        session.close()

