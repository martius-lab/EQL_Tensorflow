"""
Representations of EQL functions and layers.
    - *EQL_Layer* are regularized tf.layers.Dense objects representing the function layers in the network. The
    intermediate EQL_Layers consist of multiple *EQL_fn* objects and have a helper method get_fns to retrieve the
    fn structure of each layer.
    - *EQL_fn* is a a group of *self.repeats* identical functions. It takes the input for all these functions as a list
    and returns a list of all the outputs. *self.tf_fn* and *self.sympy_fn* are tensorflow and sympy representations of
    the function used.
    - *reg_div* implements the regularized division and reg_div.__call__ is used like a normal tf function in an EQL_fn.
"""
import sympy as sp
import tensorflow as tf
from l0_regularization import l0_dense, l0_utils
from .utils import number_of_positional_arguments
from tensorflow.python.ops import init_ops


class EQL_fn(object):
    """EQL_fn is a group of *self.repeats* identical nodes, e.g. 10 sine functions."""

    def __init__(self, tf_fn, sympy_fn, repeats):
        """
        :param tf_fn: A Tensorflow operation or a class instance with __call__ acting as a tensorflow function.
        :param sympy_fn: A sympy operation matching tf_fn
        :param repeats: number of times the function is used in the layer
        """
        self.tf_fn = tf_fn
        self.sympy_fn = sympy_fn
        self.num_positional_args = number_of_positional_arguments(self.tf_fn)
        self.repeats = repeats

    def __call__(self, data):
        slices = tf.split(data, [self.repeats] * self.num_positional_args, axis=1)
        return self.tf_fn(*slices)

    def get_total_dimension(self):
        return self.repeats * self.num_positional_args


class reg_div(object):
    """Save regularized division, used as tf function for division layer."""

    def __init__(self, div_thresh_fn):
        """
        Initializing regularized division.
        :param div_thresh_fn: a fn that calculated the division threshold from a given train step tensor
        """
        self.div_thresh_fn = div_thresh_fn

    def __call__(self, numerator, denominator):
        """
        Acts as a normal tensorflow math function, performing save regularized division. Implemented as a class so that
        it can follow the tf.div signature. Adds division loss (threshold penalty) to loss collections.
        """
        step = tf.train.get_or_create_global_step()
        div_thresh = self.div_thresh_fn(step)
        mask = tf.cast(denominator > div_thresh, dtype=tf.float32)
        div = tf.reciprocal(tf.abs(denominator) + 1e-10)
        output = numerator * div * mask
        P_theta = tf.maximum((div_thresh - denominator), 0.0)  # equation 5 in paper
        tf.add_to_collection('Threshold_penalties', P_theta)
        return output


# Dict of function tuples consisting of matching tensorflow functions/function classes and sympy funcs
dict_of_ops = {'multiply': (tf.multiply, sp.Symbol.__mul__),
               'sin': (tf.sin, sp.sin),
               'cos': (tf.cos, sp.cos),
               'id': (tf.identity, sp.Id),
               'sub': (tf.subtract, sp.Symbol.__sub__),
               'log': (tf.log, sp.log),
               'exp': (tf.exp, sp.exp),
               'reg_div': (reg_div, sp.Symbol.__div__)}


def validate_op_dict(op_dict):
    """Checks if dictionary only includes keywords matching the keywords in dict_of_ops."""
    if not isinstance(op_dict, dict):
        raise ValueError("Operation dict has to be a dictionary.")
    if not op_dict:
        raise ValueError("No parameters given")
    for key in op_dict:
        if key not in dict_of_ops:
            raise ValueError('Unknown parameter {} passed'.format(key))


def op_dict_to_eql_op_list(op_dict):
    """Transforms a dict of fn_tuples specified by strings into list of EQL functions."""
    list_of_EQL_fn = []
    for key, value in op_dict.items():
        if key == 'reg_div':
            reg_division = dict_of_ops[key][0](value.div_thresh_fn)  # This is __init__ call to reg_div class.
            list_of_EQL_fn.append(EQL_fn(reg_division.__call__, dict_of_ops[key][1], repeats=value.repeats))
        else:
            list_of_EQL_fn.append(EQL_fn(dict_of_ops[key][0], dict_of_ops[key][1], repeats=value))
    sorted_list_of_EQL_fn = sorted(list_of_EQL_fn, key=lambda fn: str(fn.sympy_fn))
    return sorted_list_of_EQL_fn


class EQL_Layer(object):
    """
    CREATES THE EQL LAYERS
    Uses module 'tf.layers.dense' to perform the operation (inputs*weights+biases),
    data is split, chunks to given to different activation functions.
    Returns: Output tensor following concatenation of the chunks after passing them through corresponding activation functons
    """

    def __init__(self, op_dict, weight_init_scale, num_inputs, bias_init_value, L0_beta, is_training, seed=None):
        validate_op_dict(op_dict)
        self.list_of_ops = op_dict_to_eql_op_list(op_dict)
        self.matmul_output_dim = sum(item.get_total_dimension() for item in self.list_of_ops)
        self.w_init_scale = tf.sqrt(weight_init_scale / (num_inputs + self.matmul_output_dim))
        self.b_init_value = bias_init_value
        self.seed = seed
        self.L0_beta = L0_beta
        self.is_training = is_training

    def __call__(self, data, weight_reg):
        layer_output = self.get_matmul_output(data=data, weight_reg=weight_reg)
        indices = [item.get_total_dimension() for item in self.list_of_ops]
        slices = tf.split(layer_output, indices, axis=1)
        outputs = [op(tensor_slice) for op, tensor_slice in zip(self.list_of_ops, slices)]
        return tf.concat(outputs, axis=1)

    def get_matmul_output(self, data, weight_reg):
        """Method building the regularized matrix multiplication layer and returning the output for given data."""
        kernel_init = tf.random_normal_initializer(stddev=self.w_init_scale, seed=self.seed)
        bias_init = init_ops.constant_initializer(value=self.b_init_value)
        regularizer = l0_utils.l0_regularizer(scale=weight_reg, beta=self.L0_beta)
        layer = l0_dense.L0Dense(is_training=self.is_training, seed=self.seed, units=self.matmul_output_dim,
                                 kernel_initializer=kernel_init, bias_initializer=bias_init,
                                 kernel_regularizer=regularizer, bias_regularizer=regularizer)
        layer.build(data.get_shape())
        assign_kernel = tf.assign(layer.kernel, layer.kernel)
        assign_bias = tf.assign(layer.bias, layer.bias)
        with tf.control_dependencies([assign_kernel, assign_bias]):
            layer_output = layer(data)
            self.kernel, self.bias = layer.kernel, layer.bias
        return layer_output

    def get_current_weights(self):
        return self.kernel, self.bias

    def get_fns(self):
        """Method returning the functions used in layer as a list of (tf_fn, sympy fn, repeats, num_args) tuples."""
        fn_list = [(eql_fn.tf_fn, eql_fn.sympy_fn, eql_fn.repeats, eql_fn.num_positional_args) for eql_fn in
                   self.list_of_ops]
        return fn_list
