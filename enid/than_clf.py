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
    learning_rate: float = 0.0001,
):
    """
    Build Time-Aware-HAN model for claim classification
    Uses an embedding layer, followed by a token-level transformer encoding
    with attention, a sentence-level time-aware-lstm with attention and sofrmax
    layer

    Parameters
    ----------
    num_classes: int
        the number of y classes

    max_sequence_length: int
        fixed padding latest number of time buckets

    max_sentence_length: int
        fixed padding number of tokens each time bucket

    batch_size: int, optional (default: 64)
        size of training batches, this won't affect training speed
        significantly; smaller batch leads to more regularization

    d_model: int, optional (default: 256)
        sizes of transformer Q, K, V vectors

    d_ff: int, optional (default: 1024)
        size of transformer feed forward layer

    h: int, optional (default: 8)
        size of heads in transformer multi-head attention

    encoder_layers: int, optional (default: 6)
        number of transformer encoder layers

    hidden_size: int, optional (default: 128)
        number of T_LSTM units

    dropout_prob: float, optional (default: 0.2)
        percentage of neurons to drop each layer

    l2_reg_lambda: float, optional (default: .0)
        L2 regularization lambda for fully-connected layer to prevent potential
        overfitting

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
    Model: "T_HAN"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    Encoder (Encoder)            multiple                  20976896
    _________________________________________________________________
    Attention_token (Attention)  multiple                  66048
    _________________________________________________________________
    tlstm (TLSTM)                multiple                  213632
    _________________________________________________________________
    Attention_sentence (Attentio multiple                  16640
    _________________________________________________________________
    dense (Dense)                multiple                  258
    =================================================================
    Total params: 37,675,650
    Trainable params: 37,674,626
    Non-trainable params: 1,024
    _________________________________________________________________
    """

    tf.keras.backend.clear_session()

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
    model = T_HAN(
        num_classes,
        pretrain_embedding,
        max_sequence_length,
        max_sentence_length,
        batch_size,
        d_model,
        d_ff,
        h,
        encoder_layers,
        hidden_size,
        dropout_prob,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.AUC(num_thresholds=50 * batch_size)],
    )
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
    patience: int = 20,
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
        number of epochs of training, one epoch means finishing training entire
        training set

    model_path: str
        the path to store the model

    dev_sample_percentage: float
        percentage of x_train seperated from training process and used for
        validation

    evaluate_every: int
        number of steps to perform a evaluation on development (validation) set
        and print out info

    patience: int
        number of steps to perform early stopping

    Returns
    ----------
    trained_model: tf.keras.Model
        built model of THAN

    Examples
    --------
    >>> from enid.than_clf import build_model, train_model
    >>> model = build_model(2, 30, 40)
    >>> model = train_model(model, t_train, x_train, y_train, num_epochs=2)
    Epoch 1/200
    723648/723648 [==============================] - 4336s 6ms/sample - ...
    Epoch 2/200
    723648/723648 [==============================] - 4375s 6ms/sample - ...
    ...
    """

    training_size = int(y_train.shape[0] / model.batch_size) * model.batch_size
    dev_size = model.batch_size * int(
        dev_sample_percentage * float(training_size) / model.batch_size
    )
    train_index = np.arange(training_size)
    np.random.shuffle(train_index)
    train_index, dev_index = train_index[:-dev_size], train_index[-dev_size:]
    training_size -= dev_size

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

    def training_batch_generate():
        i = 0
        while i < training_size - model.batch_size - 1:
            yield (
                (
                    t_train[i : i + model.batch_size],
                    x_train[i : i + model.batch_size],
                ),
                y_train[i : i + model.batch_size],
            )
            i += model.batch_size

    tf_training_set = tf.data.Dataset.from_generator(
        generator=training_batch_generate,
        output_types=((tf.int32, tf.int32), tf.int8),
        output_shapes=(
            (
                tf.TensorShape([model.batch_size, model.max_sequence_length]),
                tf.TensorShape(
                    [
                        model.batch_size,
                        model.max_sequence_length,
                        model.max_sentence_length,
                    ]
                ),
            ),
            tf.TensorShape([model.batch_size, model.num_classes]),
        ),
    ).repeat()

    tf_dev_set = tf.data.Dataset.from_tensor_slices(
        ((t_dev, x_dev), y_dev)
    ).batch(model.batch_size)

    try:
        model.fit(
            tf_training_set,
            epochs=num_epochs
            * int(training_size / model.batch_size / evaluate_every),
            verbose=2,
            callbacks=[
                # tf.keras.callbacks.EarlyStopping(
                #     monitor="val_loss", patience=patience
                # ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(model_path, "logs"),
                    update_freq="batch",
                    histogram_freq=1,
                ),
            ],
            validation_data=tf_dev_set,
            steps_per_epoch=evaluate_every,
            validation_freq=1,
        )
        save_model(model, model_path)

    except KeyboardInterrupt:
        print("KeyboardInterrupt Error: saving model...")
        save_model(model, model_path)

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
    if hasattr(model, "batch_size"):
        batch_size = model.batch_size
    else:
        batch_size = model.signatures["serving_default"].output_shapes[
            "output_1"
        ][0]

    number_examples = t_test.shape[0]
    fake_samples = batch_size - (number_examples % batch_size)
    if fake_samples > 0 and fake_samples < number_examples:
        t_test = np.concatenate([t_test, t_test[-fake_samples:]], axis=0)
        x_test = np.concatenate([x_test, x_test[-fake_samples:]], axis=0)
    if fake_samples > number_examples:
        sup_rec = int(batch_size / number_examples) + 1
        t_test, x_test = (
            np.concatenate([t_test] * sup_rec, axis=0),
            np.concatenate([x_test] * sup_rec, axis=0),
        )
        t_test, x_test = t_test[:batch_size], x_test[:batch_size]
    if fake_samples == batch_size:
        fake_samples = 0

    y_probs = np.empty((0))
    for start, end in zip(
        range(0, number_examples + fake_samples + 1, batch_size),
        range(batch_size, number_examples + fake_samples + 1, batch_size),
    ):
        probs = model.call([t_test[start:end], x_test[start:end]])[:, 0]
        y_probs = np.concatenate([y_probs, probs])
    if fake_samples > 0:
        return y_probs[:number_examples]
    else:
        return y_probs


def save_model(model, model_path):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # model.call.get_concrete_function(
    #     inputs=[
    #         tf.TensorSpec(
    #             shape=(model.batch_size, model.max_sequence_length),
    #             dtype=tf.int32
    #         ),
    #         tf.TensorSpec(
    #             shape=(
    #                 model.batch_size,
    #                 model.max_sequence_length,
    #                 model.max_sentence_length
    #             ),
    #             dtype=tf.int32
    #         )
    #     ]
    # )
    # tf.keras.models.save_model(model, model_path, save_format="tf")
    model.save(model_path, save_format='tf')


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


class T_HAN(tf.keras.Model):
    def __init__(
        self,
        num_classes: int,
        pretrain_embedding,
        max_sequence_length: int,
        max_sentence_length: int,
        batch_size: int = 64,
        d_model: int = 256,
        d_ff: int = 1024,
        h: int = 8,
        encoder_layers: int = 6,
        hidden_size: int = 128,
        dropout_prob: float = 0.2,
    ):
        """
        A Time-Aware-HAN Keras Subclassed model for claim classification
        """
        super(T_HAN, self).__init__(name="T_HAN")

        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        self.max_sentence_length = max_sentence_length
        self.batch_size = batch_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.encoder_layers = encoder_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        self.vocab_size, self.emb_size = pretrain_embedding.shape
        pretrain_embedding = np.concatenate(
            [
                pretrain_embedding,
                np.zeros(shape=(1, self.emb_size), dtype="float32"),
            ],
            axis=0,
        )
        self.vocab_size += 1
        self.Embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.emb_size,
            embeddings_initializer=tf.keras.initializers.Constant(
                pretrain_embedding
            ),
            trainable=True,
        )

        self.encoder = Encoder(
            self.d_model,
            self.d_ff,
            self.max_sentence_length,
            self.h,
            self.batch_size * self.max_sequence_length,
            self.encoder_layers,
            dropout_prob=self.dropout_prob,
            use_residual_conn=True,
        )
        self.token_level_attention = Attention(
            level="token",
            sequence_length=self.max_sentence_length,
            output_dim=self.d_model,
        )

        self.tlstm_layer = TLSTM(
            self.hidden_size, dropout_prob=self.dropout_prob
        )
        self.sentence_level_attention = Attention(
            level="sentence",
            sequence_length=self.max_sequence_length,
            output_dim=self.hidden_size,
        )

        self.output_projection_layer = tf.keras.layers.Dense(self.num_classes)
        self.temp = tf.Variable(initial_value=1., name="temp") # T temperature parameter

    def call(self, inputs, training=False):

        input_t, input_x = inputs

        # 1. get emebedding of tokens
        inputs = self.Embedding(
            input_x
        )  # [batch_size, num_bucket, sentence_length, embedding_size]
        inputs = tf.reshape(
            inputs,
            shape=[
                self.batch_size * self.max_sequence_length,
                self.max_sentence_length,
                self.emb_size,
            ],
        )
        inputs = tf.multiply(
            inputs, tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        )

        # 2. token level encoder with attention
        Q_encoded, _ = self.encoder(
            [inputs, inputs]
        )  # [batch_size*sequence_length, sentence_length_length, d_model]
        sentence_representation = self.token_level_attention(
            Q_encoded
        )  # [batch_size*num_sentences, d_model]
        sentence_representation = tf.reshape(
            sentence_representation,
            shape=[-1, self.max_sequence_length, self.d_model],
        )  # shape:[batch_size, sequence_lenth, d_model]

        # 3. sentence level tlstm with attention
        scan_time = tf.expand_dims(
            input_t, axis=-1
        )  # [batch_size x seq_length x 1]
        concat_input = tf.concat(
            [tf.cast(scan_time, tf.float32), sentence_representation], 2
        )  # [batch_size, sequence_lenth, d_model+1]
        hidden_state_sentence = self.tlstm_layer(
            concat_input
        )  # [batch_size, sequence_lenth, hidden_size]
        instance_representation = self.sentence_level_attention(
            hidden_state_sentence
        )

        # 4. output layer
        logits = self.output_projection_layer(instance_representation)
        logits_w_temp = tf.divide(logits, self.temp)
        probabilities = tf.nn.softmax(logits_w_temp)

        return probabilities  # [batch_size, self.num_classes].

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "max_sequence_length": self.max_sequence_length,
            "max_sentence_length": self.max_sentence_length,
            "batch_size": self.batch_size,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "h": self.h,
            "encoder_layers": self.encoder_layers,
            "hidden_size": self.hidden_size,
            "dropout_prob": self.dropout_prob,
        }
