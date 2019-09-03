# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Time-Aware Hierarchical Attention Model class with embedding
# Authors:     Yage Wang
# Created:     9.26.2018
###############################################################################

import os
import pickle
import json
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from .transformer import Encoder
from .attention import Attention
from .tlstm import TLSTM


def build_model(
    num_classes: int,
    max_sequence_length: int,
    max_sentence_length: int,
    batch_size: int = 64,
    d_model: int = 256,
    d_ff: int = 2048,
    h: int = 8,
    encoder_layers: int = 1,
    hidden_size: int = 128,
    dropout_prob: float = 0.1,
    l2_reg_lambda: float = 0.0,
    learning_rate: int = 0.0001,
):
    """
    Build Time-Aware-HAN model for claim classification
    Uses an embedding layer, followed by a token-level transformer encoding with attention, a sentence-level time-aware-lstm with attention and sofrmax layer

    Parameters
    ----------
    num_classes: int
        the number of y classes

    max_sequence_length: int
        fixed padding latest number of time buckets

    max_sentence_length: int
        fixed padding number of tokens each time bucket

    batch_size: int, optional (default: 64)
        size of training batches, this won't affect training speed significantly; smaller batch leads to more regularization

    d_model: int, optional (default: 256)
        sizes of transformer Q, K, V vectors

    d_ff: int, optional (default: 1024)
        size of transformer feed forward layer

    hL int, optional (default: 8)
        size of heads in transformer multi-head attention

    encoder_layers: int, optional (default: 6)
        number of transformer encoder layers

    hidden_size: int, optional (default: 128)
        number of T_LSTM units

    dropout_prob: float, optional (default: 0.2)
        percentage of neurons to drop each layer

    l2_reg_lambda: float, optional (default: .0)
        L2 regularization lambda for fully-connected layer to prevent potential overfitting

    learning_rate: int, optional (default: 1e-04)
        model learning rate

    Returns
    ----------
    trained_model: tf.keras.Model
        built model of THAN
    
    config: dict
        dictionary of THAN hyperparameters

    Examples
    --------
    >>> from enid.than_clf import build_model
    >>> model, config = build_model(2, 30, 40)
    >>> model.summary()
    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_x (InputLayer)            [(64, 40, 30)]       0
    __________________________________________________________________________________________________
    embedding (Embedding)           (64, 40, 30, 256)    16402176    input_x[0][0]
    __________________________________________________________________________________________________
    tf_op_layer_reshape_embedding ( [(2560, 30, 256)]    0           embedding[0][0]
    __________________________________________________________________________________________________
    tf_op_layer_mul (TensorFlowOpLa [(2560, 30, 256)]    0           tf_op_layer_reshape_embedding[0][
    __________________________________________________________________________________________________
    input_t (InputLayer)            [(64, 40)]           0
    __________________________________________________________________________________________________
    Encoder (Encoder)               ((2560, 30, 256), (2 20976896    tf_op_layer_mul[0][0]
                                                                     tf_op_layer_mul[0][0]
    __________________________________________________________________________________________________
    tf_op_layer_reshape_time (Tenso [(64, 40, 1)]        0           input_t[0][0]
    __________________________________________________________________________________________________
    Attention_token (Attention)     (2560, 256)          66048       Encoder[0][0]
    __________________________________________________________________________________________________
    tf_op_layer_Cast (TensorFlowOpL [(64, 40, 1)]        0           tf_op_layer_reshape_time[0][0]
    __________________________________________________________________________________________________
    tf_op_layer_reshape_encoder (Te [(64, 40, 256)]      0           Attention_token[0][0]
    __________________________________________________________________________________________________
    tf_op_layer_concat_xt (TensorFl [(64, 40, 257)]      0           tf_op_layer_Cast[0][0]
                                                                     tf_op_layer_reshape_encoder[0][0]
    __________________________________________________________________________________________________
    tlstm (TLSTM)                   (64, 40, 128)        213632      tf_op_layer_concat_xt[0][0]
    __________________________________________________________________________________________________
    Attention_sentence (Attention)  (64, 128)            16640       tlstm[0][0]
    __________________________________________________________________________________________________
    dense (Dense)                   (64, 2)              258         Attention_sentence[0][0]
    __________________________________________________________________________________________________
    softmax (Softmax)               (64, 2)              0           dense[0][0]
    ==================================================================================================
    Total params: 37,675,650
    Trainable params: 37,674,626
    Non-trainable params: 1,024
    __________________________________________________________________________________________________
    """
    tf.enable_eager_execution()
    K.clear_session()

    pretrain_embedding = pickle.load(
        open(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), r"pickle_files", r"embeddings"
                )
            ),
            "rb",
        )
    )
    vocab_size, emb_size = pretrain_embedding.shape
    pretrain_embedding = np.concatenate(
        [pretrain_embedding, np.zeros(shape=(1, emb_size), dtype="float32")],
        axis=0,
    )
    vocab_size += 1

    # building stacks
    input_t = tf.keras.Input(
        shape=(max_sequence_length,),
        batch_size=batch_size,
        dtype=tf.int32,
        name="input_t",
    )
    input_x = tf.keras.Input(
        shape=(max_sequence_length, max_sentence_length),
        batch_size=batch_size,
        dtype=tf.int32,
        name="input_x",
    )

    # 1. get emebedding of tokens
    embeddings = tf.keras.layers.Embedding(
        vocab_size,
        emb_size,
        embeddings_initializer=tf.keras.initializers.Constant(
            pretrain_embedding
        ),
        trainable=True,
    )(
        input_x
    )  # [batch_size, sequence_length, sentence_length_length, emb_size]
    embeddings = K.reshape(
        embeddings,
        shape=(
            batch_size * max_sequence_length,
            max_sentence_length,
            emb_size,
        ),
    )  # [batch_size*sequence_length, sentence_length_length, emb_size]
    embeddings = embeddings * K.sqrt(K.cast(d_model, tf.float32))

    # 2. token level encoder with attention
    Q_encoded, _ = Encoder(
        d_model,
        d_ff,
        max_sentence_length,
        h,
        batch_size * max_sequence_length,
        encoder_layers,
        dropout_prob=dropout_prob,
        use_residual_conn=True,
    )(
        [embeddings, embeddings]
    )  # [batch_size*sequence_length, sentence_length_length, d_model]

    sentence_representation = Attention(
        level="token", sequence_length=max_sentence_length, output_dim=d_model
    )(
        Q_encoded
    )  # [batch_size*num_sentences, d_model]
    sentence_representation = K.reshape(
        sentence_representation, (-1, max_sequence_length, d_model)
    )  # [batch_size, sequence_lenth, d_model]

    # 3. sentence level tlstm with attention
    scan_time = K.reshape(
        input_t, shape=(batch_size, max_sequence_length, 1)
    )  # [batch_size x seq_length x 1]
    concat_input = tf.keras.layers.Concatenate(axis=-1)(
        [tf.cast(scan_time, tf.float32), sentence_representation]
    )  # [batch_size, sequence_lenth, d_model+1]
    hidden_state_sentence = TLSTM(hidden_size, dropout_prob=dropout_prob)(
        concat_input
    )  # [batch_size, sequence_lenth, hidden_size]
    instance_representation = Attention(
        level="sentence",
        sequence_length=max_sequence_length,
        output_dim=hidden_size,
    )(hidden_state_sentence)

    # 4. output layer
    logits = tf.keras.layers.Dense(num_classes)(instance_representation)
    probabilities = tf.keras.layers.Softmax()(logits)

    model = tf.keras.Model(inputs=[input_t, input_x], outputs=probabilities)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.AUC(num_thresholds=50 * batch_size)],
    )
    model.batch_size = batch_size
    return model


def train_model(
    model,
    t_train,
    x_train,
    y_train,
    num_epochs: int = 5,
    model_path: str = "model",
    dev_sample_percentage: float = 0.01,
    evaluate_every: int = 200,
):
    """
    Train compiled THAN model

    Parameters
    ----------
    model: tf.keras.Model
        built model of THAN

    t_train: 2-D numpy array, shape (num_exemplars, num_bucket)
        variable indices all buckets and sections

    x_train: 2-D numpy array, shape (num_exemplars, num_bucket)
        variable indices all variables

    y_train: 2-D numpy array, shape (num_exemplars, num_classes)
        whole training ground truth

    num_epochs: int
        number of epochs of training, one epoch means finishing training entire training set

    model_path: str
        the path to store the model

    dev_sample_percentage: float
        percentage of x_train seperated from training process and used for validation

    evaluate_every: int
        number of steps to perform a evaluation on development (validation) set and print out info
    
    Returns
    ----------
    trained_model: tf.keras.Model
        built model of THAN

    Examples
    --------
    >>> from enid.than_clf import build_model, train_model
    >>> model = build_model(2, 30, 40)
    >>> train_model(model, t_train, x_train, y_train, num_epoch=2)
    Epoch 1/2
    723648/723648 [==============================] - 4336s 6ms/sample - loss: 0.4735 - categorical_crossentropy: 0.4735 - val_loss: 0.4703 - val_categorical_crossentropy: 0.4703
    Epoch 2/2
    723648/723648 [==============================] - 4375s 6ms/sample - loss: 0.4566 - categorical_crossentropy: 0.4566 - val_loss: 0.4736 - val_categorical_crossentropy: 0.4736
    """

    training_size = int(y_train.shape[0] / model.batch_size) * model.batch_size
    dev_size = model.batch_size * int(
        dev_sample_percentage * float(training_size) / model.batch_size
    )
    train_index = np.arange(training_size)
    np.random.shuffle(train_index)
    train_index, dev_index = train_index[:-dev_size], train_index[-dev_size:]
    t_train, t_dev = (
        np.take(t_train, train_index, axis=0),
        np.take(t_train, dev_index, axis=0),
    )
    x_train, x_dev = (
        np.take(x_train, train_index, axis=0),
        np.take(x_train, dev_index, axis=0),
    )
    y_train, y_dev = (
        np.take(y_train, train_index, axis=0),
        np.take(y_train, dev_index, axis=0),
    )

    try:
        model.fit(
            x=[t_train, x_train],
            y=y_train,
            batch_size=model.batch_size,
            epochs=num_epochs,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=1
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(model_path, "logs"),
                    update_freq="batch",
                    histogram_freq=1,
                ),
            ],
            validation_data=([t_dev, x_dev], y_dev),
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt Error: saving model...")
        model.save(model_path)

    model.save(model_path)
    return model


def deploy_model(model, t_test, x_test):
    """
    Get output from trained THAN model

    Parameters
    ----------
    model: tf.keras.Model
        trained model of THAN

    t_test: 2-D numpy array, shape (num_exemplars, num_bucket)
        variable indices all buckets and sections

    x_test: 2-D numpy array, shape (num_exemplars, num_bucket)
        variable indices all variables

    Returns
    ----------
    y_probs: 1-D numpy array, shape (num_exemplar,)
            predicted target values based on trained model

    Examples
    --------
    >>> from enid.than_clf import load_model, deploy_model
    >>> model = load_model("model_1")
    >>> deploy_model(model, t_test, x_test)
    array([9.9515426e-01,
            4.6948572e-03,
            3.1738445e-02,,
            ...,
            9.9895418e-01,
            5.6348788e-04,
            9.9940193e-01], dtype=float32)
    """

    number_examples = t_test.shape[0]
    fake_samples = model.batch_size - (number_examples % model.batch_size)
    if fake_samples > 0 and fake_samples < number_examples:
        t_test = np.concatenate([t_test, t_test[-fake_samples:]], axis=0)
        x_test = np.concatenate([x_test, x_test[-fake_samples:]], axis=0)
    if fake_samples > number_examples:
        sup_rec = int(model.batch_size / number_examples) + 1
        t_test, x_test = (
            np.concatenate([t_test] * sup_rec, axis=0),
            np.concatenate([x_test] * sup_rec, axis=0),
        )
        t_test, x_test = t_test[: model.batch_size], x_test[: model.batch_size]

    y_probs = model.predict(x=[t_test, x_test], batch_size=model.batch_size)
    if fake_samples > 0:
        return y_probs[:number_examples, 0]
    else:
        return y_probs[:, 0]


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


class NBatchLogger(tf.keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """

    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params["metrics"]:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ""
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += " - %s: %.4f" % (k, val)
                else:
                    metrics_log += " - %s: %.4e" % (k, val)
            print("\nstep: {} ... {}".format(self.step, metrics_log))
            self.metric_cache.clear()
