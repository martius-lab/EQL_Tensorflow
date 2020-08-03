from collections import namedtuple, OrderedDict

import tensorflow as tf
from .EQL_Layer_tf import EQL_Layer as EQL_Layer
from .utils import get_div_thresh_fn,  \
                  tensorboard_summarize
from .evaluation import generate_symbolic_expression
from .timeout import time_limit

class EQLDivGraph(object):
    """ Class that defines a graph for EQL. """

    def __init__(self, mode, metadata, layer_width, num_h_layers, output_bound,
                 weight_init_param, batch_size, test_div_threshold,
                 train_val_split, L0_beta=None, network_init_seed=None, **_):
        self.train_data_size = int(train_val_split * metadata['train_val_examples'])
        self.width = layer_width
        self.num_h_layers = num_h_layers
        self.seed = network_init_seed
        self.output_bound = output_bound or metadata['extracted_output_bound']
        self.batch_size = batch_size
        self.is_training = mode == tf.estimator.ModeKeys.TRAIN
        div_thresh_fn = get_div_thresh_fn(self.is_training, self.batch_size,
                                          test_div_threshold,
                                          train_examples=self.train_data_size)
        reg_div = namedtuple('reg_div', ['repeats', 'div_thresh_fn'])
        hidden_layer_fns = OrderedDict(sin=self.width, cos=self.width,
                                       multiply=self.width, id=self.width)
        self.eql_layers = [EQL_Layer(op_dict=hidden_layer_fns,
                                         weight_init_scale=weight_init_param,
                                         num_inputs=metadata['num_inputs'],
                                         bias_init_value=0.,
                                         L0_beta=L0_beta,
                                         is_training=tf.constant(self.is_training),
                                         seed=self.seed)
                           for _ in range(self.num_h_layers)]
        final_layer_fns = OrderedDict(reg_div=reg_div(repeats=metadata['num_outputs'],
                                      div_thresh_fn=div_thresh_fn))
        self.eql_layers.append(EQL_Layer(op_dict=final_layer_fns,
                                             weight_init_scale=weight_init_param,
                                             num_inputs=metadata['num_inputs'],
                                             bias_init_value=1.,
                                             L0_beta=L0_beta,
                                             is_training=tf.constant(self.is_training),
                                             seed=self.seed))


    def __call__(self, inputs, reg_scale):
        output = inputs
        for layer in self.eql_layers:
            output = layer(output, weight_reg=reg_scale)
        P_bound = (tf.abs(output) - self.output_bound) * \
                   tf.cast((tf.abs(output) > self.output_bound),
                           dtype=tf.float32)
        tf.add_to_collection('Bound_penalties', P_bound)
        return output


class ModelFn(object):
    """ Сlass for creating an model_fn to Estimator as well as storing current model graph. """

    def __init__(self,  config, evaluation_hook):
        self.model_graph = None
        self.evaluation_hook = evaluation_hook
        self.config = config
        self.reg_scale = None
        self.metadata = None
        self.weights = None
        self.fns_list = None
        self.numba_expr = None
        self.sympy_expr = None


    def __call__(self, features, labels, mode, params):
        """ The model_fn argument for creating an Estimator. """
        self.model_graph = EQLDivGraph(mode=mode, metadata=self.metadata,**params)
        self.evaluation_hook.init_network_structure(self.reg_scale, self.model_graph, params)
        global_step = tf.train.get_or_create_global_step()
        if mode == tf.estimator.ModeKeys.TRAIN:
            predictions = self.model_graph(features, self.reg_scale)[:-1]  # last input element in each batch is extrapolation element, no label given
            labels = labels[:-1]
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss = tf.reduce_sum([tf.reduce_mean(reg_loss) for reg_loss in reg_losses], name='reg_loss')
            bound_penalty = tf.reduce_sum(tf.get_collection('Bound_penalties'))
            P_theta = tf.reduce_sum(tf.get_collection('Threshold_penalties'))
            mse_loss = tf.identity(tf.losses.mean_squared_error(labels, predictions), name='mse_loss')
            loss = tf.losses.get_total_loss() + P_theta + bound_penalty
            loss = tf.identity(loss, name='total_loss')
            train_accuracy = tf.identity(
                tf.metrics.percentage_below(values=tf.abs(labels - predictions), threshold=0.02)[1], name='train_accuracy')
            tensorboard_summarize([loss, mse_loss, reg_loss, bound_penalty, P_theta], family='losses')
            tensorboard_summarize(train_accuracy, family='accuracies')
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss,
                train_op=tf.train.AdamOptimizer(params['learning_rate'], beta1=params['beta1']).minimize(loss, global_step))
        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = self.model_graph(features, self.reg_scale)
            loss = tf.sqrt(tf.losses.mean_squared_error(labels, predictions))
            eval_acc_metric = tf.metrics.percentage_below(values=tf.abs(labels - predictions), threshold=0.02)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                              eval_metric_ops={'eval_accuracy': eval_acc_metric})
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = self.model_graph(features, self.reg_scale)
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    def set_reg_scale(self, reg_scale):
        """
        Regularization hyperparameter for currect stage of training
        """
        self.reg_scale = reg_scale


    def set_metadata(self, metadata):
        """ The metadata about current training data. Used to constract proper division schedule
            and infer parameters of the graph dimentions
        """
        self.metadata = metadata


    def get_weights(self):
        """
        Get the current network weights
        """
        self.weights = self.evaluation_hook.weights


    def get_fns_list(self):
        """
        Get the current list of functions in model graph
        """
        self.fns_list = [layer.get_fns() for layer in self.model_graph.eql_layers]


    @time_limit(40)
    def generate_symbolic_expression(self, round_decimals):
        """
        Generates sympy and a fast numba expression
        :param round_decimals: integer specifying to which decimal the expression is rounded
        """
        tf.logging.info('Generating symbolic expression.')
        self.get_weights()
        self.get_fns_list()
        kernels, biases = zip(*self.weights)
        self.sympy_expr, self.numba_expr = generate_symbolic_expression(kernels, biases, self.fns_list, 3)
