# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Time-Aware Hierarchical Attention Model class with embedding
# Authors:     Yage Wang
# Created:     9.26.2018
###############################################################################

import os
import pickle
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from .transformer import Encoder
from .tlstm import TLSTMCell
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score

class T_HAN(object):

    def __init__(self, mode, **kwargs):
        """
        A Time-Aware-HAN for claim classification
        Uses an embedding layer, followed by a token-level inception CNN with attention, a sentence-level time-aware-lstm with attention and sofrmax layer

        Parameters
        ----------
        mode: str
            'train' or 'deploy' mode

        Train Mode Parameters
        ----------
        max_sequence_length: int
            fixed padding latest number of time buckets

        max_sentence_length: int
            fixed padding number of tokens each time bucket

        num_classes: int
            the number of y classes

        batch_size: int
            size of training batches, this won't affect training speed significantly; smaller batch leads to more regularization

        hidden_size: int
            number of T_LSTM units

        learning_rate: float
            initial learning rate for Adam Optimizer

        grad_clip_thres: float
            gradient clip threshold, see https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them

        decay_steps: int
            step frequency to decay the learning rate. e.g. if 5000, model will reduce learning rate by decay_rate every 5000 trained batches

        decay_rate: float
            percentage of learning rate decay rate

        dropout_keep_prob: float
            percentage of neurons to keep from dropout regularization each layer

        l2_reg_lambda: float, optional (default: .0)
            L2 regularization lambda for fully-connected layer to prevent potential overfitting

        objective: str, optional (default: 'ce')
            the objective function (loss) model trains on; if 'ce', use cross-entropy, if 'auc', use AUROC as objective

        initializer: tf tensor initializer object, optional (default: tf.orthogonal_initializer())
            initializer for fully connected layer weights

        Deploy Mode Parameters
        ----------
        model_path: str
            the path to store the model

        step: int, optional (defult None)
            if not None, load specific model with given step

        Examples
        --------
        >>> from enid.than_clf import T_HAN
        >>> model_1 = T_HAN('train', max_sequence_length=50, max_sentence_length=20,
                hidden_size=128, num_classes=2, learning_rate=0.05,
                decay_steps=5000, decay_rate=0.9,
                dropout_keep_prob=0.8, l2_reg_lambda=0.0,
                objective='ce')
        >>> model_1.train(t_train=T, x_train=X,
                        y_train=y, dev_sample_percentage=0.01,
                        num_epochs=20, batch_size=64,
                        evaluate_every=100, model_path='./model/')
        >>> model_2 = T_HAN('deploy', model_path='./model')
        >>> model_2.deploy(t_test=T_test, x_test=X_test)
        array([9.9515426e-01,
               4.6948572e-03,
               3.1738445e-02,,
               ...,
               9.9895418e-01,
               5.6348788e-04,
               9.9940193e-01], dtype=float32)
        """
        tf.reset_default_graph()
        assert mode in ['train', 'deploy'], f'AttributeError: mode only acccept "train" or "deploy", got {mode} instead.'
        self.mode = mode

        if self.mode == 'train':

            self.pretrain_embedding = pickle.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), r"pickle_files", r"embeddings")), "rb"))
            self.num_classes = kwargs['num_classes']
            self.max_sequence_length = kwargs['max_sequence_length']
            self.max_sentence_length = kwargs['max_sentence_length']
            self.batch_size = kwargs.get('batch_size', 64)
            self.d_model = kwargs.get('d_model', 256)
            self.d_ff = kwargs.get('d_ff', 1024)
            self.h = kwargs.get('h', 8)
            self.encoder_layers = kwargs.get('encoder_layers', 6)
            self.hidden_size = kwargs.get('hidden_size', 128)
            self.dropout_keep_prob = kwargs.get('dropout_keep_prob', 0.8)
            self.l2_reg_lambda = kwargs.get('l2_reg_lambda', 0.0)
            self.learning_rate = kwargs.get('learning_rate', 0.0001)
            self.grad_clip_thres = kwargs.get('grad_clip_thres', None)
            self.decay_steps = kwargs.get('decay_steps', 5000)
            self.decay_rate = kwargs.get('decay_rate', 0.9)
            self.initializer = kwargs.get('initializer', tf.initializers.he_normal())
            self.objective = kwargs.get('objective', 'ce')

            self.graph = tf.get_default_graph()
            with self.graph.as_default():
                self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
                self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
                self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

                # add placeholder
                self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.max_sequence_length, self.max_sentence_length], name="input_x") # X [instance_size, num_bucket, sentence_length]
                self.input_t = tf.placeholder(tf.int32, [self.batch_size, self.max_sequence_length],                           name="input_t") # T [instance_size, num_bucket]
                self.input_y = tf.placeholder(tf.int32, [self.batch_size, self.num_classes],                                   name="input_y") # y [instance_size, num_classes]
                
                with tf.name_scope("embedding"):
                    self.emb_size = self.pretrain_embedding.shape[1]
                    embedding_matrix = tf.concat([self.pretrain_embedding, tf.zeros((1, self.emb_size))], axis=0)
                    self.Embedding = tf.Variable(embedding_matrix, trainable=True, dtype=tf.float32, name='embedding')

                # 1. get emebedding of tokens
                self.input = tf.nn.embedding_lookup(self.Embedding, self.input_x) # [batch_size, num_bucket, sentence_length, embedding_size]
                self.input = tf.reshape(self.input, shape=[self.batch_size*self.max_sequence_length, self.max_sentence_length, self.emb_size])
                self.input = tf.multiply(self.input, tf.sqrt(tf.cast(self.d_model, dtype=tf.float32)))
                # input_mask = tf.get_variable("input_mask", [self.max_sentence_length, 1], initializer=self.initializer)
                # self.input = tf.add(self.input, input_mask) #[batch_size,sentence_length,embed_size].

                # 2. encoder
                encoder_class = Encoder(self.input, self.input, self.d_model, self.d_ff, self.max_sentence_length, self.h, self.batch_size*self.max_sequence_length,
                                        self.encoder_layers, dropout_keep_prob=self.dropout_keep_prob, use_residual_conn=True)
                Q_encoded, _ = encoder_class.encoder_multiple_layers() #[batch_size*sequence_length, sentence_length_length, d_model]

                sentence_representation = self._attention_token_level(Q_encoded) # [batch_size*num_sentences, d_model]
                sentence_representation = tf.reshape(sentence_representation, shape=[-1, self.max_sequence_length, self.d_model]) # shape:[batch_size,sequence_lenth,d_model]
                # Q_encoded = tf.reshape(Q_encoded, shape=(self.batch_size*self.max_sequence_length, -1)) #[batch_size*sequence_length, sentence_length_length, d_model]
                # with tf.variable_scope("sentence_representation"):
                #     self.encoder_W_projection = tf.get_variable("encoder_W_projection", shape=[self.max_sentence_length*self.d_model, self.hidden_size], initializer=self.initializer)
                #     self.encoder_b_projection = tf.get_variable("encoder_b_projection", shape=[self.hidden_size])
                #     sentence_representation = tf.matmul(Q_encoded, self.encoder_W_projection) + self.encoder_b_projection #[batch_size*sequence_lenth,hidden_size]              
                #     sentence_representation = tf.reshape(sentence_representation, shape=[-1, self.max_sequence_length, self.hidden_size]) # shape:[batch_size,sequence_lenth,hidden_size]

                # make scan_time [batch_size x seq_length x 1]
                scan_time = tf.reshape(self.input_t, [tf.shape(self.input_t)[0], tf.shape(self.input_t)[1], 1])
                concat_input = tf.concat([tf.cast(scan_time, tf.float32), sentence_representation], 2) # [batch_size x seq_length x hidden_size+1]

                self.tlstm_cell = TLSTMCell(self.hidden_size, True, dropout_keep_prob_in=self.dropout_keep_prob,
                                            dropout_keep_prob_h=self.dropout_keep_prob, dropout_keep_prob_out=self.dropout_keep_prob,
                                            dropout_keep_prob_gate=self.dropout_keep_prob, dropout_keep_prob_forget=self.dropout_keep_prob)
                self.hidden_state_sentence, _ = tf.nn.dynamic_rnn(self.tlstm_cell, concat_input, dtype=tf.float32, time_major=False)

                self.instance_representation = self._attention_sentence_level(self.hidden_state_sentence)
                with tf.name_scope("output"):
                    self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size, self.num_classes], initializer=self.initializer)  # [embed_size,label_size]
                    self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])
                    self.logits = tf.matmul(self.instance_representation, self.W_projection) + self.b_projection  # [batch_size, self.num_classes]. main computation graph is here.
                    self.probs = tf.nn.softmax(self.logits, name="probs")

                assert self.objective in ['ce', 'auc'], 'AttributeError: objective only acccept "ce" or "auc", got {}'.format(str(self.objective))
                if self.objective == 'ce':  self.loss_val = self._loss(self.l2_reg_lambda)
                if self.objective == 'auc': self.loss_val = self._loss_roc_auc(self.l2_reg_lambda)
                self._train()
                self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[batch_size,]

                self.loss_sum = tf.summary.scalar("loss_train", self.loss_val)
                self.attention_sum = tf.summary.histogram("attentions", self.instance_representation)
                self.merged_sum = tf.summary.merge_all()

            config = tf.ConfigProto(log_device_placement=True)
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config, graph=self.graph)

        if self.mode == 'deploy':
            self.model_path = kwargs['model_path']
            self.step = kwargs['step']
            self.sess = tf.Session()
            self.sess.as_default()

            # Recreate the network graph. At this step only graph is created.
            if self.step:
                saver = tf.train.import_meta_graph(os.path.join(self.model_path.rstrip('/'), 'model-'+str(self.step)+'.meta'))
                saver.restore(self.sess, os.path.join(self.model_path.rstrip('/'), 'model-'+str(self.step)))
            else:
                saver = tf.train.import_meta_graph(os.path.join(self.model_path.rstrip('/'), 'model.meta'))
                saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
            self.graph = tf.get_default_graph()

            # restore graph names for predictions
            self.probs = self.graph.get_tensor_by_name("output/probs:0")
            self.input_x = self.graph.get_tensor_by_name("input_x:0")
            self.input_t = self.graph.get_tensor_by_name("input_t:0")

            self.batch_size = self.input_x.get_shape().as_list()[0]
            self.max_sequence_length, self.max_sentence_length = self.input_x.get_shape()[1:]
            self.max_sequence_length, self.max_sentence_length = int(self.max_sequence_length), int(self.max_sentence_length)

    def __del__(self):
        if hasattr(self, "sess"): self.sess.close()
        tf.reset_default_graph()

    def train(self, t_train, x_train, y_train, dev_sample_percentage, num_epochs, evaluate_every, model_path, debug=False):
        """
        Training module for T_HAN objectives
        
        Parameters
        ----------
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

        evaluate_every: int
            number of steps to perform a evaluation on development (validation) set and print out info

        model_path: str
            the path to store the model

        debug: boolean, default False
            if True, run TensorFlow Session as interactive debug session.
        """

        # get number of input exemplars
        training_size = y_train.shape[0]

        dev_sample_index = -1 * int(dev_sample_percentage * float(training_size))
        t_train, t_dev = t_train[:dev_sample_index], t_train[dev_sample_index:]
        x_train, x_dev = x_train[:dev_sample_index], x_train[dev_sample_index:]
        y_train, y_dev = y_train[:dev_sample_index], y_train[dev_sample_index:]
        training_size = y_train.shape[0]

        saver = tf.train.Saver()

        # initialize all variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        if debug: self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        print('start time:', datetime.now())
        # create model root path if not exists
        if not os.path.exists(model_path): os.mkdir(model_path)

        writer_train = tf.summary.FileWriter(os.path.join(model_path, 'train'))
        writer_val = tf.summary.FileWriter(os.path.join(model_path, 'vali'))
        writer_train.add_graph(self.sess.graph)
        
        # get current epoch
        curr_epoch = self.sess.run(self.epoch_step)
        try:
            for epoch in range(curr_epoch, num_epochs):
                print('Epoch', epoch+1, '...')
                counter = 0

                # loop batch training
                for start, end in zip(range(0, training_size, self.batch_size), range(self.batch_size, training_size, self.batch_size)):
                    epoch_x = x_train[start:end]
                    epoch_t = t_train[start:end]
                    epoch_y = y_train[start:end]

                    # create model inputs
                    feed_dict = {self.input_x: epoch_x, self.input_t: epoch_t, self.input_y: epoch_y}

                    # train one step
                    curr_loss, _, merged_sum = self.sess.run([self.loss_val, self.train_op, self.merged_sum], feed_dict)
                    writer_train.add_summary(merged_sum, global_step=self.sess.run(self.global_step))
                    counter = counter+1

                    # evaluation
                    if counter % evaluate_every == 0:
                        #train_accu, _ = model.auc.eval(feed_dict, session=sess)
                        dev_loss, dev_accu = self._do_eval(x_dev, t_dev, y_dev, writer_val)
                        print(f'Step: {counter: <6}  |  Loss: {curr_loss:10.7f}  |  Development Loss: {dev_loss:10.7f}  |  Development AUROC: {dev_accu: 10.7f}')
                self.sess.run(self.epoch_increment)

                # write model into disk at the end of each 10 epoch     if epoch > 9 and epoch % 10 == 9:
                saver.save(self.sess, os.path.join(model_path, 'model'), global_step=self.global_step)
                print('='*100)
        except KeyboardInterrupt:
            print("KeyboardInterrupt Error: saving model...")
            saver.save(self.sess, os.path.join(model_path, 'model'), global_step=self.global_step)

        print('End time:', datetime.now())

    def deploy(self, t_test, x_test):
        """
        Testing module for T_HAN models
        
        Parameters
        ----------
        t_test: 2-D numpy array, shape (num_exemplars, num_bucket)
            variable indices all buckets and sections

        x_test: 2-D numpy array, shape (num_exemplars, num_bucket)
            variable indices all variables

        Returns
        ----------
        y_probs: 1-D numpy array, shape (num_exemplar,)
                predicted target values based on trained model
        """
        number_examples = t_test.shape[0]
        fake_samples = self.batch_size - (number_examples % self.batch_size)
        if fake_samples > 0 and fake_samples < number_examples:
            t_test = np.concatenate([t_test, t_test[-fake_samples:]], axis=0)
            x_test = np.concatenate([x_test, x_test[-fake_samples:]], axis=0)
        if fake_samples > number_examples:
            sup_rec = int(self.batch_size / number_examples) + 1
            t_test, x_test = np.concatenate([t_test] * sup_rec, axis=0), np.concatenate([x_test] * sup_rec, axis=0)
            t_test, x_test = t_test[:self.batch_size], x_test[:self.batch_size]

        y_probs = np.empty((0))
        for start, end in zip(range(0, number_examples+fake_samples+1, self.batch_size), range(self.batch_size, number_examples+fake_samples+1, self.batch_size)):
            feed_dict = {self.input_x: x_test[start:end],
                         self.input_t: t_test[start:end]}
            probs = self.sess.run(self.probs, feed_dict)[:, 0]
            y_probs = np.concatenate([y_probs, probs])

        if fake_samples > 0: return y_probs[:number_examples]
        else: return y_probs

    ###########################  PRIVATE FUNCTIONS  ###########################

    def _attention_token_level(self, hidden_state):
        """
        @param hidden_state: [batch_size*num_sentences,sentence_length,d_model]
        @return representation [batch_size*num_sentences,d_model]
        """
        with tf.name_scope("token_level_attention"):
            self.W_w_attention_token = tf.get_variable("W_w_attention_token",
                                                      shape=[self.d_model, self.d_model],
                                                      initializer=self.initializer)
            self.W_b_attention_token = tf.get_variable("W_b_attention_token", shape=[self.d_model])
            self.context_vecotor_token = tf.get_variable("what_is_the_informative_token", shape=[self.d_model],
                                                        initializer=self.initializer)
        
        token_hidden_state_2 = tf.reshape(hidden_state, shape=[-1, self.d_model])
        token_hidden_representation = tf.nn.tanh(tf.matmul(token_hidden_state_2, self.W_w_attention_token) + self.W_b_attention_token)
        token_hidden_representation = tf.reshape(token_hidden_representation, shape=[-1, self.max_sentence_length, self.d_model])
        token_hidden_state_context_similiarity = tf.multiply(token_hidden_representation, self.context_vecotor_token)
        self.token_attention_logits = tf.reduce_sum(token_hidden_state_context_similiarity, axis=2)  # [batch_size*num_sentences,sentence_length]
        self.token_p_attention = tf.nn.softmax(self.token_attention_logits, name='token_attention')  # [batch_size*num_sentences,sentence_length]
        token_p_attention_expanded = tf.expand_dims(self.token_p_attention, axis=2)  # [batch_size*num_sentences,sentence_length,1]
        sentence_representation = tf.multiply(token_p_attention_expanded, hidden_state)  # [batch_size*num_sentences,sentence_length, d_model]
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)  # [batch_size*num_sentences, d_model]
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
                                                            shape=[self.hidden_size], initializer=self.initializer)
        
        hidden_state_2 = tf.reshape(hidden_state_sentence, shape=[-1, self.hidden_size])
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2, self.W_w_attention_sentence) + self.W_b_attention_sentence)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.max_sequence_length, self.hidden_size])
        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_sentence)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        instance_representation = tf.multiply(p_attention_expanded, hidden_state_sentence)
        instance_representation = tf.reduce_sum(instance_representation, axis=1)
        return instance_representation

    def _train(self):
        """
        based on the loss, use Adam to update parameter
        """
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.lr_sum = tf.summary.scalar("learning_rate", learning_rate)
        if self.grad_clip_thres:
            self.train_op = tf.train.AdamOptimizer(learning_rate)
            gradients, variables = zip(*self.train_op.compute_gradients(self.loss_val))
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_thres)
            self.train_op = self.train_op.apply_gradients(zip(gradients, variables), global_step=self.global_step)
        else: self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_val, global_step=self.global_step)
    # self.train_op = tf.contrib.layers.optimize_loss(self.loss_val, self.global_step, learning_rate=self.learning_rate,
    #                                                           optimizer='Adam', summaries=["gradients"])

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

    def _do_eval(self, eval_x, eval_t, eval_y, writer_val):
        """
        Evaluate development in batch (if direcly force sess run entire development set, may raise OOM error)
        """
        number_examples = eval_x.shape[0]
        fake_samples = number_examples % self.batch_size
        eval_x, eval_t, eval_y = eval_x[:number_examples-fake_samples], eval_t[:number_examples-fake_samples], eval_y[:number_examples-fake_samples]
        number_examples = number_examples - fake_samples

        eval_loss, eval_counter = 0.0, 0
        eval_probs = np.empty((0, self.num_classes))
        for start, end in zip(range(0,number_examples+1,self.batch_size), range(self.batch_size,number_examples+1,self.batch_size)):
            feed_dict = {self.input_x: eval_x[start:end],
                         self.input_t: eval_t[start:end],
                         self.input_y: eval_y[start:end]}
            curr_eval_loss, curr_probs, merged_sum = self.sess.run([self.loss_val, self.probs, self.merged_sum], feed_dict)
            writer_val.add_summary(merged_sum, global_step=self.sess.run(self.global_step))
            eval_loss, eval_probs, eval_counter = eval_loss+curr_eval_loss, np.concatenate([eval_probs, curr_probs]), eval_counter+1

        eval_acc = roc_auc_score(eval_y[:, 0], eval_probs[:, 0]) #self.sess.run(self.auc, {self.test_y: eval_y, self.test_p: eval_probs})
        return eval_loss/float(eval_counter), eval_acc
