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
import numpy as np
from .transformer import Encoder
from .attention import Attention
from .tlstm import TLSTM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    # learning_rate: int = 0.0001,
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

    h: int, optional (default: 8)
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
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate),
    #     loss=tf.keras.losses.CategoricalCrossentropy(),
    #     metrics=[tf.keras.metrics.AUC(num_thresholds=50 * batch_size)],
    # )
    return model


@tf.function
def training_step(model, optimizer, epoch_t, epoch_x, epoch_y):
    with tf.GradientTape() as tape:
        y_pred = model([epoch_t, epoch_x])
        loss = tf.keras.losses.binary_crossentropy(
            y_true=epoch_y[:, 1], y_pred=y_pred[:, 1],
        )

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        grads_and_vars=zip(grads, model.trainable_variables)
    )

    return loss


@tf.function
def validation_step(model, metrics, t_dev,  x_dev, y_dev, dev_size):
    y_dev_pred = tf.concat(
        [
            model([t_dev[start:end], x_dev[start:end]])
            for start, end in zip(
            range(0, dev_size + 1, model.batch_size),
            range(
                model.batch_size,
                dev_size + 1,
                model.batch_size,
            ),
        )
        ],
        axis=0,
    )
    dev_loss = tf.keras.losses.binary_crossentropy(
        y_true=y_dev[:, 1], y_pred=y_dev_pred[:, 1],
    )

    metrics.update_state(y_dev, y_dev_pred)
    return dev_loss


def train_model(
    model,
    t_train,
    x_train,
    y_train,
    learning_rate: int = 0.0001,
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
    >>> model = train_model(model, t_train, x_train, y_train, num_epochs=2)
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
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        train_summary_writer = tf.summary.create_file_writer(
            os.path.join(model_path, "train")
        )
        val_summary_writer = tf.summary.create_file_writer(
            os.path.join(model_path, "vali")
        )

        counter = 0
        metrics = tf.keras.metrics.AUC(num_thresholds=50 * model.batch_size)

        for epoch in range(1, num_epochs + 1):
            print("Epoch", epoch, "...")

            # loop batch training
            for start, end in zip(
                range(0, training_size, model.batch_size),
                range(model.batch_size, training_size, model.batch_size),
            ):
                epoch_x = x_train[start:end]
                epoch_t = t_train[start:end]
                epoch_y = y_train[start:end]
                loss = training_step(
                    model, optimizer, epoch_t, epoch_x, epoch_y
                )
                loss = loss.numpy().mean()
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=counter)

                if counter % evaluate_every == 0:
                    dev_loss = validation_step(
                        model, metrics, t_dev, x_dev, y_dev, dev_size
                    )
                    dev_loss = dev_loss.numpy().mean()
                    with val_summary_writer.as_default():
                        tf.summary.scalar(
                            'validation_loss',
                            dev_loss,
                            step=counter
                        )
                        tf.summary.scalar(
                            'validation_auc',
                            metrics.result(),
                            step=counter
                        )

                    print(
                        f"Step: {counter: <6}  |  Loss: {loss:10.7f}  |  "
                        + f"Development Loss: {dev_loss:10.7f}  |  Development"
                        + f" AUROC: {metrics.result().numpy(): 10.7f}"
                    )

                    metrics.reset_states()

                counter += 1
            
            save_model(model, model_path)
            print("=" * 100)

        # model.fit(
        #     x=[t_train, x_train],
        #     y=y_train,
        #     batch_size=model.batch_size,
        #     epochs=num_epochs,
        #     verbose=1,
        #     callbacks=[
        #         tf.keras.callbacks.EarlyStopping(
        #             monitor="val_loss", patience=1
        #         ),
        #         tf.keras.callbacks.TensorBoard(
        #             log_dir=os.path.join(model_path, "logs"),
        #             update_freq="batch",
        #             histogram_freq=1,
        #         ),
        #     ],
        #     validation_data=([t_dev, x_dev], y_dev),
        # )
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


def save_model(model, model_path):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    tf.saved_model.save(model, model_path)
    # json.dump(
    #     model.get_config(),
    #     open(os.path.join(model_path, "model_config.json"), "w"),
    # )
    # model.save_weights(os.path.join(model_path, "model_weights.h5"))


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    # config = json.load(open(os.path.join(model_path, "model_config.json")))
    # model = build_model(**config)
    # init_t = np.zeros(
    #     shape=[config["batch_size"], config["max_sequence_length"]],
    #     dtype="int32",
    # )
    # init_x = np.zeros(
    #     shape=[
    #         config["batch_size"],
    #         config["max_sequence_length"],
    #         config["max_sentence_length"],
    #     ],
    #     dtype="int32",
    # )
    # _ = model.call(inputs=[init_t, init_x])
    # model.load_weights(os.path.join(model_path, "model_weights.h5"))

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

        # self.tlstm_layer = tf.keras.layers.RNN(TLSTMCell(self.hidden_size, time_aware=True, dropout=self.dropout_prob), return_sequences=True, dynamic=True)
        self.tlstm_layer = TLSTM(
            self.hidden_size, dropout_prob=self.dropout_prob
        )
        self.sentence_level_attention = Attention(
            level="sentence",
            sequence_length=self.max_sequence_length,
            output_dim=self.hidden_size,
        )

        self.output_projection_layer = tf.keras.layers.Dense(self.num_classes)

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
        probabilities = tf.nn.softmax(logits)

        return probabilities  # [batch_size, self.num_classes]. main computation graph is here.

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
