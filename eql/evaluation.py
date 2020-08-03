"""
Module for symbolic manipulation with formulas and evaluation.
    - Implements *EvaluationHook* which is used to generate symbolic expressions of the current formula represented
    by the network structure and to calculate the complexity of the current network.
    - Generation of the symbolic expression mainly consists of *symbolic_eql_layer* and *symbolic_matmul_and_bias*
    routines, which perform the symbolic representation of the EQL fns and the matrix multiplication/bias addition.
    - Symbolic expressions are saved as pngs of a latex representation and of a rendered graphviz graph.
    - The complexity calculation is performed in three steps:
    calculate_complexity -> complexity_of_layer -> complexity of node
"""
from functools import reduce
from os import path
import os
import pickle
import numpy as np
import numpy  # Needs a separate import for numba compilation
import sympy
import tensorflow as tf
from graphviz import Source
from sympy.printing.dot import dotprint
from tensorflow.python.training.session_run_hook import SessionRunHook
from subprocess import CalledProcessError
from sympy.printing.pycode import NumPyPrinter as Printer
from sympy.utilities.lambdify import lambdastr
from numba import jit

from .timeout import time_limit, TimeoutException
from .utils import generate_arguments, yield_with_repeats


class EvaluationHook(SessionRunHook):
    """Hook for saving evaluating the eql."""

    def __init__(self, store_path=None):
        self.weights = None
        self.store_path = store_path
        self.fns_list = None
        self.round_decimals = 3
        self.complexity = None
        self.calls = 0
        self.numba_expr = None
        self.sympy_expr = None

    def begin(self):
        self.iteration = 0

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):
        if self.iteration == 0:
            tens = [layer.get_current_weights() for layer in self.layers]
        else:
            tens = {}
        return tf.train.SessionRunArgs(fetches=tens)

    def after_run(self, run_context, run_values):
        if self.iteration == 0:
            self.weights = run_values.results
        self.iteration += 1

    def end(self, session):
        if self.store_path is not None:
            if self.fns_list is None:
                raise ValueError("Network structure not provided. Call init_network_structure first.")
            kernels, biases = zip(*self.weights)
            self.complexity = calculate_complexity(kernels, biases, self.fns_list)
            if self.generate_symbolic_expr:
                try:
                    self.sympy_expr, self.numba_expr = generate_symbolic_expression(kernels, biases, self.fns_list,  self.round_decimals)
                except TimeoutException:
                    print('Generating symbolic expression timed out, expression is too complicated. tf.Estimator will be used for predictions')
                try:
                    save_symbolic_expression(save_path=self.store_path, expr_list=self.sympy_expr, extra_info=str(self.reg_scale))
                except TimeoutException:
                    print('Saving symbolic expression timed out, expression is too complicated.')

    def init_network_structure(self, reg_scale, model, params):
        self.fns_list = [layer.get_fns() for layer in model.eql_layers]
        self.generate_symbolic_expr = params['generate_symbolic_expr']
        self.layers = model.eql_layers
        self.reg_scale = reg_scale

    def get_complexity(self):
        if self.complexity is not None:
            return self.complexity
        else:
            raise ValueError('Complexity not yet evaluated.')


@time_limit(20)
def proper_simplify(expr):
    """ Combine trig and normal simplification for sympy expression."""
    return sympy.simplify(sympy.trigsimp(expr))


def symbolic_matmul_and_bias(input_nodes_symbolic, weight_matrix, bias_vector):
    """ Computes a symbolic representations of nodes in a layer after matrix mul of the previous layer.
    :param input_nodes_symbolic: list of sympy expressions
    :param weight_matrix: 2D numpy array of shape (input_dim, output_dim)
    :param bias_vector: 1D numpy array of shape (output_dim)
    :return: list of sympy expressions at output nodes of length (output_dim)
    """

    def output_for_index(i):
        return bias_vector[i] + sum([w * x for w, x in zip(weight_matrix[:, i], input_nodes_symbolic)])

    return [output_for_index(i) for i in range(weight_matrix.shape[1])]


def symbolic_eql_layer(input_nodes_symbolic, output_fn_group_list):
    """ Computes a symbolic representation of a node given incoming weights and the output fn.
    :param input_nodes_symbolic: list of sympy expressions
    :param output_fn_group_list: list of (sympy function, repeats) tuples to be applied to input nodes.
    :return: list of sympy expressions at output nodes
    """
    _, output_fns, repeats, arg_nums = zip(*output_fn_group_list)
    arg_iterator = generate_arguments(input_nodes_symbolic, repeats, arg_nums)
    fn_iterator = yield_with_repeats(output_fns, repeats)
    return [fn(*items) for fn, items in zip(fn_iterator, arg_iterator)]


def get_symbol_list(number_of_symbols):
    """ Returns a list of sympy expression, each being an identity of a variable. To be used for input layer."""
    return sympy.symbols(['x_{}'.format(i + 1) for i in range(number_of_symbols)], real=True)


def expression_graph_as_png(expr, output_file, view=True):
    """ Save a PNG of rendered graph (graphviz) of the symbolic expression.
    :param expr: sympy expression
    :param output_file: string with .png extension
    :param view: set to True if system default PNG viewer should pop up
    :return: None
    """
    assert output_file.endswith('.png')
    graph = Source(dotprint(expr))
    graph.format = 'png'
    graph.render(output_file.rpartition('.png')[0], view=view, cleanup=True)


def expr_to_latex_png(expr, output_file):
    """Saves a png of a latex representation of a symbolic expression."""
    sympy.preview(expr, viewer='file', filename=output_file)


def expr_to_latex(expr):
    """Returns latex representation (as string) of a symbolic expression."""
    return sympy.latex(expr)

#TODO: Fix it
@time_limit(20)
def round_sympy_expr(expr, decimals):
    """Returns the expression with every float rounded to the given number of decimals."""
    if isinstance(expr, list):
        return [round_sympy_expr(exp, decimals) for exp in expr]
    rounded_expr = expr
    for a in sympy.preorder_traversal(expr):
        if isinstance(a, sympy.Float):
            rounded_expr = rounded_expr.subs(a, round(a, decimals))
    return rounded_expr


@time_limit(60)
def generate_symbolic_expression(kernels, biases, fns_list, round_decimals):
    """
    Saves a symbolic expression of network as pngs showing the equation as a tree and as a latex equation.
    ALso generates a fast numba expression
    :param kernels: list of 2D numpy arrays
    :param biases: list of 1D numpy arrays
    :param fns_list: list of lists of (tf_fn, sp_fn, repeats, num_args) tuples
    :param round_decimals: integer specifying to which decimal the expression is rounded
    """
    tf.logging.info('Generating symbolic expression.')
    in_nodes = get_symbol_list(kernels[0].shape[0])
    res = in_nodes
    for kernel, bias, fns in zip(kernels, biases, fns_list):
        res = symbolic_matmul_and_bias(res, kernel, bias)
        res = symbolic_eql_layer(res, fns)
    try:
        res = proper_simplify(res)
        # res = round_sympy_expr(res, round_decimals)
    except TimeoutException:
        print('Simplification of result failed.')
    numba_res = generate_numba_fn(sympy_fn=res, variables=in_nodes)
    return res, numba_res


def save_symbolic_expression(save_path, expr_list, extra_info):
    """
    Saves symbolic expression as latex and graph representation.
    :param save_path: Path specifying where to save the symbolic representations.
    :param expr_list: List of sympy expressions.
    :param extra_info: Extra information to save, e.g. the last used regularization constant.
    """
    for i, expr in enumerate(expr_list):
        # with open(path.join(save_path, extra_info + '_f.dat'), 'wb') as f:
        #     pickle.dump(expr, f)
        try:
            if not os.path.exists(os.path.join(save_path, 'latex')):
                os.makedirs(os.path.join(save_path, 'latex'))
            if not os.path.exists(os.path.join(save_path, 'graph')):
                os.makedirs(os.path.join(save_path, 'graph'))
            expr_to_latex_png(expr, os.path.join(save_path, ('latex/latex_y{}_{}.png').format(str(i), extra_info)))
            expression_graph_as_png(expr, os.path.join(save_path, ('graph/graph_y{}_{}.png').format(str(i), extra_info)),
                                    view=False)
        except (CalledProcessError, RuntimeError) as e:
            print('Saving of symbolic representation failed. Formula too large?')


def generate_numba_fn(sympy_fn, variables):
    """
    Generates a fast numeric expression from a szmpy expression using numba.
    :param sympy_fn: A sympy expression.
    :param variables: List of sympy variables used in the sympy expression.
    :return: A numba function
    """
    string_rep = eval(lambdastr(variables, sympy_fn, printer=Printer))
    numba_fn = jit(string_rep)
    return numba_fn


def calculate_complexity(kernels, biases, fns_list):
    """
    Routine that counts units with nonzero input * output weights (only non-identity units)
    :param kernels: list of numpy matrices
    :param biases: list of numpy arrays
    :param fns_list: list of lists containg (tf_fn, sp_fn, repeats, arg_num) tuples
    :return: complexity (number of active nodes) of network
    """
    complexities = [complexity_of_layer(fns=fns, in_biases=in_biases, in_weights=in_weights, out_weights=out_weights)
                    for fns, in_biases, in_weights, out_weights in zip(fns_list, biases[:-1], kernels[:-1], kernels[1:])]
    complexity = sum(complexities)
    return complexity


def complexity_of_layer(fns, in_biases, in_weights, out_weights):
    """
    Routine that returns the complexity (number of active nodes) of a given layer.
    :param fns: list of (tf_fn, sp_pn, repeats, arg_num) tuples, one for each fn block in layer
    :param in_biases: numpy array describing the biases added to inputs for this layer
    :param in_weights: numpy matrix describing the weights between the previous layer and this layer
    :param out_weights: numpy matrix describing the weights between this layer and the next layer
    :return: complexity (number of active nodes) of a given layer
    """
    in_weight_sum = np.sum(np.abs(in_weights), axis=0) + in_biases  # adding up all abs weights contributing to input
    out_weight_sum = np.sum(np.abs(out_weights), axis=1)  # adding up all abs weights that use specific output
    output_fns, _, repeats, arg_nums = zip(*fns)
    input_iterator = generate_arguments(all_args=in_weight_sum, repeats=repeats, arg_nums=arg_nums)
    fn_iterator = yield_with_repeats(output_fns, repeats)
    count = sum([complexity_of_node(out_weight, in_weights, fn)
                 for out_weight, in_weights, fn in zip(out_weight_sum, input_iterator, fn_iterator)])
    return count


def complexity_of_node(out_weight, in_weights, fn):
    """
    Routine that returns the complexity of a node.
    :param out_weight: float output weight of node
    :param in_weights: tuple of input weights for node
    :param fn: tensorflow function used in this node
    :return: 1 if node is active and 0 if inactive
    """
    if fn == tf.identity:
        return 0
    all_weights = [out_weight, *in_weights]
    if all(weight != 0 for weight in all_weights):
        return 1
    else:
        return 0
