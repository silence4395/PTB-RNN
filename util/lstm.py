#!/usr/bin/python
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : lstm.py
## Authors    : zhluo@aries
## Create Time: 2017-07-20:10:54:55
## Description:
## 
##
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.python.framework import ops

# define switch class
class Switch(object):
  def __init__(self, value):
    self.value = value
    self.fall = False
  
  def __iter__(self):
    """Retrun the match method once, then stop"""
    yield self.match
    raise StopIteration
  
  def match(self, *args):
    """Indicate whether or not to enter a casr suite"""
    if self.fall or not args:
      return True
    elif self.value in args:
      self.fall = True
      return True
    else:
      return False

# define LSTM cell
class BasicLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
  that follows.
  """

  def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
        Must set to `0.0` manually when restoring from CudnnLSTM-trained
        checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMCell instead.
    """
    super(BasicLSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM).

    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size x 2 * self.state_size]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = math_ops.sigmoid
    lib_diy_0 = tf.load_op_library('util/usr_op/sigmoid_diy.so')
    lib_diy_1 = tf.load_op_library('util/usr_op/tanh/tanh_diy.so')
    
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

    concat = _linear([inputs, h], 4 * self._num_units, True)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
    
    # optional 'sigmoid_diy' 'tanh_diy' 'sigmoid_tanh_diy' 'origin'
    op = 'tanh_diy'
    
    for case in Switch(op):
      if case('origin'):
        new_c = (
          c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)
        break
      if case('sigmoid_diy'):
        new_c = (
          c * lib_diy_0.sigmoid_diy(f + self._forget_bias) + lib_diy_0.sigmoid_diy(i) * self._activation(j))
        new_h = self._activation(new_c) * lib_diy_0.sigmoid_diy(o)
        break
      if case('tanh_diy'):
        new_c = (
          c * sigmoid(f + self._forget_bias) + sigmoid(i) * lib_diy_1.tanh_diy(j))
        new_h = lib_diy_1.tanh_diy(new_c) * sigmoid(o)
        break
      if case('sigmoid_tanh_diy'):
        new_c = (
          c * lib_diy_0.sigmoid_diy(f + self._forget_bias) + lib_diy_0.sigmoid_diy(i) * lib_diy_1.tanh_diy(j))
        new_h = lib_diy_1.tanh_diy(new_c) * lib_diy_0.sigmoid_diy(o)
        break
      if case():
        print("[ Error ] please set LSTM function operation type")

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state
