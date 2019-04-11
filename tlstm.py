# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Time-Aware LSTMCell
# Authors:     Yage Wang
# Created:     4.10.2019
###############################################################################

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl

_EPSILON = 10**-4

class TLSTMCell(rnn_cell_impl.RNNCell):
    """
    Time-Aware LSTM with several additional regularization; edit `BasicLSTMCell` of tensorflow.
    
    The implementation is based on: http://arxiv.org/abs/1409.2329 and
    https://www.kdd.org/kdd2017/papers/view/patient-subtyping-via-time-aware-lstm-networks
    
    We add forget_bias (default: 1) to the biases of the forget gate in order to reduce the scale
    of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not use peep-hole connections:
    it is the basic baseline. For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}

    Parameters
    ----------
    num_units : int
        The number of units in the LSTM cell.

    forget_bias : float, optional
        The bias added to forget gates (see above). Must set to `0.0` manually when restoring
        from CudnnLSTM-trained checkpoints.

    activation : function or string, optional (default: tf.tanh)
        Activation function of the inner states.

    reuse : boolean, optional
        Describing whether to reuse variables in an existing scope. If not `True`, and the
        existing scope already has the given variables, an error is raised.

    layer_norm : boolean, optional
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
    >>> tlstm_cell = TLSTMCell(128)
    >>> init_state = tlstm_cell.zero_state(batch_size=64, dtype=tf.float32)
    >>> output, _ = tf.nn.dynamic_rnn(tlstm_cell, input, initial_state=init_state)
    """
    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 activation=None,
                 reuse=None,
                 layer_norm: bool=False,
                 norm_shift: float=0.0,
                 norm_gain: float=1.0,  # layer normalization
                 dropout_keep_prob_in: float = 1.0,
                 dropout_keep_prob_h: float=1.0,
                 dropout_keep_prob_out: float=1.0,
                 dropout_keep_prob_gate: float=1.0,
                 dropout_keep_prob_forget: float=1.0
                 ):
        """Initialize the basic LSTM cell."""

        super(TLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift
        
        self._keep_prob_i = dropout_keep_prob_in
        self._keep_prob_g = dropout_keep_prob_gate
        self._keep_prob_f = dropout_keep_prob_forget
        self._keep_prob_o = dropout_keep_prob_out
        self._keep_prob_h = dropout_keep_prob_h

    @property
    def state_size(self):
        return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def _map_elapse_time(self, t):
        c1 = constant_op.constant(1, dtype=dtypes.float32)
        c2 = constant_op.constant(2.7183, dtype=dtypes.float32)
        T = math_ops.div(c1, gen_math_ops.log(t + c2), name='Log_elapse_time') # according to paper, used for large time delta like days
        Ones = array_ops.ones([1, self._num_units], dtype=dtypes.float32)
        T = math_ops.matmul(T, Ones)
        return T

    def _layer_normalization(self, inputs, scope=None):
        """
        :param inputs: (batch, shape)
        :param scope:
        :return : layer normalized inputs (batch, shape)
        """
        shape = inputs.get_shape()[-1:]
        with vs.variable_scope(scope or "layer_norm"):
            # Initialize beta and gamma for use by layer_norm.
            g = vs.get_variable("gain", shape=shape, initializer=init_ops.constant_initializer(self._g))  # (shape,)
            s = vs.get_variable("shift", shape=shape, initializer=init_ops.constant_initializer(self._b))  # (shape,)
        m, v = nn_impl.moments(inputs, [1], keep_dims=True)  # (batch,)
        normalized_input = (inputs - m) / math_ops.sqrt(v + _EPSILON)  # (batch, shape)
        return normalized_input * g + s

    @staticmethod
    def _linear(x, weight_shape, bias=True, scope=None):
        """ linear projection (weight_shape: input size, output size) """
        with vs.variable_scope(scope or "linear"):
            w = vs.get_variable("kernel", shape=weight_shape)
            x = math_ops.matmul(x, w)
            if bias:
                b = vs.get_variable("bias", initializer=[0.0] * weight_shape[-1])
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
        
        input_size = inputs.get_shape().as_list()[1]
        t, x = array_ops.split(inputs, [1, input_size-1], axis=1)

        # Dealing with time irregularity
        # Map elapse time in days or months
        T = self._map_elapse_time(t)

        # Decompose the previous cell if there is a elapse time
        with vs.variable_scope("tlstm_weight"):
            self.W_decomp = vs.get_variable('Decomposition_Hidden_weight', shape=[self._num_units, self._num_units])
            self.b_decomp = vs.get_variable('Decomposition_Hidden_bias_enc', shape=[self._num_units])

        C_ST = gen_math_ops.tanh(math_ops.matmul(c, self.W_decomp) + self.b_decomp)
        C_ST_dis = math_ops.multiply(T, C_ST)
        # if T is 0, then the weight is one
        c = c - C_ST + C_ST_dis

        args = array_ops.concat([x, h], 1)
        concat = self._linear(args, [args.get_shape()[-1], 4 * self._num_units])

        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
        if self._layer_norm:
            i = self._layer_normalization(i, "layer_norm_i")
            j = self._layer_normalization(j, "layer_norm_j")
            f = self._layer_normalization(f, "layer_norm_f")
            o = self._layer_normalization(o, "layer_norm_o")
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

        # layer normalization for memory cell (original paper didn't use for memory cell).
        # if self._layer_norm:
        #     new_c = self._layer_normalization(new_c, "state")

        new_c = memory + gated_in
        new_h = self._activation(new_c) * math_ops.sigmoid(o)
        new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
        return new_h, new_state