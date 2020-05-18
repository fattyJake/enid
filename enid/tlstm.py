# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Time-Aware LSTMCell
# Authors:     Yage Wang
# Created:     4.10.2019
###############################################################################

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops


class TLSTM(Layer):
    """
    Time-Aware LSTM with several additional regularization and batch
    normalization; edit `LSTM` of tensorflow.

    The implementation is based on: http://arxiv.org/abs/1409.2329 and
    https://www.kdd.org/kdd2017/papers/view/patient-subtyping-via-time-aware-lstm-networks

    Parameters
    ----------
    units : int
        The number of units in the LSTM cell.

    dropout_prob : float, optional
        probability of varaibles to dropout

    Examples
    --------
    >>> from enid.tlstm import TLSTM
    >>> tlstm_layer = TLSTM(128, 0.2)
    >>> output, _ = tlstm_layer(input_)
    """

    def __init__(self, units, dropout_prob=0.0, kernel_regularizer=None, **kwargs):
        super(TLSTM, self).__init__(**kwargs)
        self.units = units
        self.dropout_prob = dropout_prob
        self.kernel_regularizer = kernel_regularizer

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.input_dim = input_shape[-1] - 1
        self.batch_size = input_shape[0]

        self.Wi = self.add_weight(
            name="Input_Hidden_weight",
            shape=[self.input_dim, self.units],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )
        self.Ui = self.add_weight(
            name="Input_State_weight",
            shape=[self.units, self.units],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )
        self.bi = self.add_weight(
            name="Input_Hidden_bias",
            shape=[self.units,],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )

        self.Wf = self.add_weight(
            name="Forget_Hidden_weight",
            shape=[self.input_dim, self.units],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )
        self.Uf = self.add_weight(
            name="Forget_State_weight",
            shape=[self.units, self.units],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )
        self.bf = self.add_weight(
            name="Forget_Hidden_bias",
            shape=[self.units,],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )

        self.Wo = self.add_weight(
            name="Output_Hidden_weight",
            shape=[self.input_dim, self.units],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )
        self.Uo = self.add_weight(
            name="Output_State_weight",
            shape=[self.units, self.units],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )
        self.bo = self.add_weight(
            name="Output_Hidden_bias",
            shape=[self.units,],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )

        self.Wc = self.add_weight(
            name="Cell_Hidden_weight",
            shape=[self.input_dim, self.units],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )
        self.Uc = self.add_weight(
            name="Cell_State_weight",
            shape=[self.units, self.units],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )
        self.bc = self.add_weight(
            name="Cell_Hidden_bias",
            shape=[self.units,],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )

        self.W_decomp = self.add_weight(
            name="Decomposition_Hidden_weight",
            shape=[self.units, self.units],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )
        self.b_decomp = self.add_weight(
            name="Decomposition_Hidden_bias_enc",
            shape=[self.units,],
            initializer="he_normal",
            regularizer=self.kernel_regularizer
        )

        self.built = True

    def _map_elapse_time(self, t):
        c1 = constant_op.constant(1, dtype=dtypes.float32)
        c2 = constant_op.constant(2.7183, dtype=dtypes.float32)
        T = math_ops.divide(
            c1, gen_math_ops.log(t / 10 + c2), name="Log_elapse_time"
        )
        Ones = array_ops.ones([1, self.units], dtype=dtypes.float32)
        T = math_ops.matmul(T, Ones)
        return T

    def _step(self, states, inputs):

        prev_hidden_state, prev_cell = array_ops.unstack(states)
        x = array_ops.slice(inputs, [0, 1], [self.batch_size, self.input_dim])
        t = array_ops.slice(inputs, [0, 0], [self.batch_size, 1])

        # Dealing with time irregularity
        # Map elapse time in days or months
        T = self._map_elapse_time(t)

        # Decompose the previous cell if there is a elapse time
        C_ST = gen_math_ops.tanh(
            math_ops.matmul(prev_cell, self.W_decomp) + self.b_decomp
        )
        C_ST_dis = math_ops.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input gate
        i = nn_ops.dropout(
            math_ops.sigmoid(
                math_ops.matmul(x, self.Wi)
                + math_ops.matmul(prev_hidden_state, self.Ui)
                + self.bi
            ),
            rate=self.dropout_prob,
        )
        # Forget Gate
        f = nn_ops.dropout(
            math_ops.sigmoid(
                math_ops.matmul(x, self.Wf)
                + math_ops.matmul(prev_hidden_state, self.Uf)
                + self.bf
            ),
            rate=self.dropout_prob,
        )
        # Output Gate
        o = nn_ops.dropout(
            math_ops.sigmoid(
                math_ops.matmul(x, self.Wo)
                + math_ops.matmul(prev_hidden_state, self.Uo)
                + self.bo
            ),
            rate=self.dropout_prob,
        )
        # Candidate Memory Cell
        C = nn_ops.dropout(
            gen_math_ops.tanh(
                math_ops.matmul(x, self.Wc)
                + math_ops.matmul(prev_hidden_state, self.Uc)
                + self.bc
            ),
            rate=self.dropout_prob,
        )
        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * math_ops.tanh(Ct)

        return array_ops.stack([current_hidden_state, Ct])

    def call(self, inputs):

        scan_input = array_ops.transpose(
            inputs, perm=[1, 0, 2]
        )  # scan input is [seq_length x batch_size x input_dim+1]
        initial_hidden = array_ops.zeros(
            [self.batch_size, self.units], dtypes.float32
        )
        ini_state_cell = array_ops.stack([initial_hidden, initial_hidden])

        packed_hidden_states = functional_ops.scan(
            self._step, scan_input, initializer=ini_state_cell, name="states"
        )
        return array_ops.transpose(packed_hidden_states[:, 0, :, :], [1, 0, 2])
