# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Time-Aware LSTMCell
# Authors:     Yage Wang
# Created:     4.10.2019
###############################################################################

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import control_flow_ops

_EPSILON = 10 ** -4


class TLSTMCell(rnn_cell_impl.RNNCell):
    """
    Time-Aware LSTM with several additional regularization and batch
    normalization; edit `BasicLSTMCell` of tensorflow.
    
    The implementation is based on: http://arxiv.org/abs/1409.2329 and
    https://www.kdd.org/kdd2017/papers/view/patient-subtyping-via-time-aware-lstm-networks
    
    We add forget_bias (default: 1) to the biases of the forget gate in order
    to reduce the scale
    of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not use
    peep-hole connections: it is the basic baseline. For advanced models,
    please use the full @{tf.nn.rnn_cell.LSTMCell}

    Parameters
    ----------
    num_units : int
        The number of units in the LSTM cell.

    time_aware : boolean
        If False, perform as basic LSTM cell; if True, perform time
        decomposition. Note that if True, corresponding cell input should be
        [batch_size x 1+input_size], where the first column is time-delta

    forget_bias : float, optional
        The bias added to forget gates (see above). Must set to `0.0` manually
        when restoring from CudnnLSTM-trained checkpoints.

    activation : function or string, optional (default: tf.tanh)
        Activation function of the inner states.

    reuse : boolean, optional
        Describing whether to reuse variables in an existing scope. If not
        `True`, and the existing scope already has the given variables, an
        error is raised.

    batch_norm : boolean, optional
        If True, apply layer normalization.

    norm_shift : float, optional
        Shift parameter for layer normalization.

    norm_gain : float
        Gain parameter for layer normalization.

    dropout_keep_prob_in : float, optional
        keep probability of variational dropout for input

    dropout_keep_prob_out : float, optional
        keep probability of variational dropout for output

    dropout_keep_prob_gate : float, optional
        keep probability of variational dropout for gating cell

    dropout_keep_prob_forget : float, optional
        keep probability of variational dropout for forget cell

    dropout_keep_prob_h : float, optional
        keep probability of recurrent dropout for gated state

    Examples
    --------
    >>> import tensorflow as tf
    >>> from enid.tlstm import TLSTMCell
    >>> tlstm_cell = TLSTMCell(128, True)
    >>> init_state = tlstm_cell.zero_state(batch_size=64, dtype=tf.float32)
    >>> output, _ = tf.nn.dynamic_rnn(tlstm_cell, input,
            initial_state=init_state)
    """

    def __init__(
        self,
        num_units,
        time_aware,
        forget_bias=1.0,
        activation=None,
        reuse=None,
        batch_norm: bool = True,
        norm_shift: float = 0.0,
        norm_gain: float = 1.0,
        dropout_keep_prob_in: float = 1.0,
        dropout_keep_prob_h: float = 1.0,
        dropout_keep_prob_out: float = 1.0,
        dropout_keep_prob_gate: float = 1.0,
        dropout_keep_prob_forget: float = 1.0,
        scope: str = "",
    ):
        """Initialize the basic LSTM cell."""

        super(TLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._time_aware = time_aware
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

        self._batch_norm = batch_norm
        self._g = norm_gain
        self._b = norm_shift

        self._keep_prob_i = dropout_keep_prob_in
        self._keep_prob_g = dropout_keep_prob_gate
        self._keep_prob_f = dropout_keep_prob_forget
        self._keep_prob_o = dropout_keep_prob_out
        self._keep_prob_h = dropout_keep_prob_h

        self._scope = scope

    @property
    def state_size(self):
        return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def _map_elapse_time(self, t):
        c1 = constant_op.constant(1, dtype=dtypes.float32)
        c2 = constant_op.constant(2.7183, dtype=dtypes.float32)
        T = math_ops.div(
            c1, gen_math_ops.log(t / 10 + c2), name="Log_elapse_time"
        )
        Ones = array_ops.ones([1, self._num_units], dtype=dtypes.float32)
        T = math_ops.matmul(T, Ones)
        return T

    def _batch_normalization(self, x, name_scope, trainable, decay=0.999):
        """Assume 2d [batch, values] tensor"""

        with vs.variable_scope(name_scope):
            size = x.get_shape().as_list()[1]

            scale = vs.get_variable(
                "scale", [size], initializer=init_ops.constant_initializer(0.1)
            )
            offset = vs.get_variable("offset", [size])

            pop_mean = vs.get_variable(
                "pop_mean",
                [size],
                initializer=init_ops.constant_initializer(0.0),
                trainable=False,
            )
            pop_var = vs.get_variable(
                "pop_var",
                [size],
                initializer=init_ops.constant_initializer(1.0),
                trainable=False,
            )
            batch_mean, batch_var = nn_impl.moments(x, [0])

            train_mean_op = state_ops.assign(
                pop_mean, pop_mean * decay + batch_mean * (1 - decay)
            )
            train_var_op = state_ops.assign(
                pop_var, pop_var * decay + batch_var * (1 - decay)
            )

            def batch_statistics():
                with control_dependencies([train_mean_op, train_var_op]):
                    return nn_impl.batch_normalization(
                        x, batch_mean, batch_var, offset, scale, _EPSILON
                    )

            def population_statistics():
                return nn_impl.batch_normalization(
                    x, pop_mean, pop_var, offset, scale, _EPSILON
                )

            return control_flow_ops.cond(
                constant_op.constant(trainable, dtype=dtypes.bool),
                batch_statistics,
                population_statistics,
            )

    @staticmethod
    def _linear(x, weight_shape, bias=True, scope=None):
        """ linear projection (weight_shape: input size, output size) """
        with vs.variable_scope(scope or "linear"):
            w = vs.get_variable("kernel", shape=weight_shape)
            x = math_ops.matmul(x, w)
            if bias:
                b = vs.get_variable(
                    "bias", initializer=[0.0] * weight_shape[-1]
                )
                return nn_ops.bias_add(x, b)
            else:
                return x

    def call(self, inputs, state):
        """
        Time Aware Long short-term memory cell (TLSTM).

        Parameters
        ----------
        inputs: 2-D tensor
            shape `[batch_size x input_size]`.
        
        state: LSTMStateTuple
            state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`

        Returns
        ----------
        A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """

        c, h = state  # memory cell, hidden unit

        if self._time_aware:
            input_size = inputs.get_shape().as_list()[1]
            t, inputs = array_ops.split(inputs, [1, input_size - 1], axis=1)

            # Dealing with time irregularity
            # Map elapse time in days or months
            T = self._map_elapse_time(t)

            # Decompose the previous cell if there is a elapse time
            C_ST = gen_math_ops.tanh(
                self._linear(
                    c,
                    [self._num_units, self._num_units],
                    bias=True,
                    scope=(
                        "decomposition"
                        + f"{'_'+self._scope if self._scope else ''}"
                    ),
                )
            )
            C_ST_dis = math_ops.multiply(T, C_ST)
            # if T is 0, then the weight is one
            c = c - C_ST + C_ST_dis

        if self._batch_norm:
            xh = self._linear(
                inputs,
                [inputs.get_shape()[-1], 4 * self._num_units],
                bias=False,
                scope=f"x_weight{'_'+self._scope if self._scope else ''}",
            )
            hh = self._linear(
                h,
                [h.get_shape()[-1], 4 * self._num_units],
                bias=False,
                scope=f"h_weight{'_'+self._scope if self._scope else ''}",
            )
            bias = vs.get_variable("bias", [4 * self._num_units])

            bn_xh = self._batch_normalization(xh, "xh", self.trainable)
            bn_hh = self._batch_normalization(hh, "hh", self.trainable)
            concat = bn_xh + bn_hh + bias
        else:
            args = array_ops.concat([inputs, h], 1)
            concat = self._linear(
                args, [args.get_shape()[-1], 4 * self._num_units]
            )

        i, j, f, o = array_ops.split(
            value=concat, num_or_size_splits=4, axis=1
        )
        g = self._activation(j)  # gating

        # variational dropout
        i = nn_ops.dropout(i, self._keep_prob_i)
        g = nn_ops.dropout(g, self._keep_prob_g)
        f = nn_ops.dropout(f, self._keep_prob_f)
        o = nn_ops.dropout(o, self._keep_prob_o)

        gated_in = math_ops.sigmoid(i) * g
        memory = c * math_ops.sigmoid(f + self._forget_bias)

        # recurrent dropout
        gated_in = nn_ops.dropout(gated_in, self._keep_prob_h)
        new_c = memory + gated_in

        # layer normalization for memory cell (original paper didn't use for
        # memory cell).
        if self._batch_norm:
            new_c = self._batch_normalization(new_c, "state", self.trainable)

        new_h = self._activation(new_c) * math_ops.sigmoid(o)
        new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)

        return new_h, new_state
