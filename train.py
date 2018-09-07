""" Neural Network Estimator for EQL - Equation Learner """
import math
import sys
from collections import namedtuple

import tensorflow as tf

import EQL_Layer_tf as eql
from data_utils import get_input_fns, extract_metadata
from evaluation import set_evaluation_hook
from utils import step_to_epochs, get_run_config, save_results, update_runtime_params, \
    get_div_thresh_fn, get_max_episode

# more network parameters are loaded from utils.py
default_params = {'model_base_dir': 'results',
                  'id': 1,  # job_id to identify jobs in result metrics file, separate model_dir for each id
                  'train_val_file': 'data/F1data_train_val',  # Datafile containing training, validation data
                  'test_file': 'data/F1data_test',  # Datafile containing test data, if set to None no test data is used
                  'epoch_factor': 1000,  # max_epochs = epoch_factor * num_h_layers
                  'num_h_layers': 1,  # number of hidden layers used in network
                  'generate_symbolic_expr': True,  # saves final network as a latex png and symbolic graph
                  'kill_summaries': False,  # reduces data generation, recommended when creating many jobs
                  }


class Model(object):
    """ Class that defines a graph for EQL. """

    def __init__(self, mode, layer_width, num_h_layers, reg_sched, output_bound, weight_init_param, epoch_factor,
                 batch_size, test_div_threshold, reg_scale, l0_threshold, train_val_split, network_init_seed=None, **_):
        self.train_data_size = int(train_val_split * metadata['train_val_examples'])
        self.width = layer_width
        self.num_h_layers = num_h_layers
        self.weight_init_scale = weight_init_param / math.sqrt(metadata['num_inputs'] + num_h_layers)
        self.seed = network_init_seed
        self.reg_start = math.floor(num_h_layers * epoch_factor * reg_sched[0])
        self.reg_end = math.floor(num_h_layers * epoch_factor * reg_sched[1])
        self.output_bound = output_bound or metadata['extracted_output_bound']
        self.reg_scale = reg_scale
        self.batch_size = batch_size
        self.l0_threshold = l0_threshold
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        div_thresh_fn = get_div_thresh_fn(self.is_training, self.batch_size, test_div_threshold,
                                          train_examples=self.train_data_size)
        reg_div = namedtuple('reg_div', ['repeats', 'div_thresh_fn'])
        self.eql_layers = [eql.EQL_Layer(sin=self.width, cos=self.width, multiply=self.width, id=self.width,
                                         weight_init_scale=self.weight_init_scale, seed=self.seed)
                           for _ in range(self.num_h_layers)]
        self.eql_layers.append(
            eql.EQL_Layer(reg_div=reg_div(repeats=metadata['num_outputs'], div_thresh_fn=div_thresh_fn),
                          weight_init_scale=self.weight_init_scale, seed=self.seed))

    def __call__(self, inputs):
        global_step = tf.train.get_or_create_global_step()
        num_epochs = step_to_epochs(global_step, self.batch_size, self.train_data_size)
        l1_reg_sched = tf.multiply(tf.cast(tf.less(num_epochs, self.reg_end), tf.float32),
                                   tf.cast(tf.greater(num_epochs, self.reg_start), tf.float32)) * self.reg_scale
        l0_threshold = tf.cond(tf.less(num_epochs, self.reg_end), lambda: tf.zeros(1), lambda: self.l0_threshold)

        output = inputs
        for layer in self.eql_layers:
            output = layer(output, l1_reg_sched=l1_reg_sched, l0_threshold=l0_threshold)

        P_bound = (tf.abs(output) - self.output_bound) * tf.cast((tf.abs(output) > self.output_bound), dtype=tf.float32)
        tf.add_to_collection('Bound_penalties', P_bound)
        return output


def model_fn(features, labels, mode, params):
    """ The model_fn argument for creating an Estimator. """
    model = Model(mode=mode, **params)
    evaluation_hook.init_network_structure(model, params)
    global_step = tf.train.get_or_create_global_step()
    input_data = features
    predictions = model(input_data)
    if mode == tf.estimator.ModeKeys.TRAIN:
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum([tf.reduce_mean(reg_loss) for reg_loss in reg_losses], name='reg_loss_mean_sum')
        bound_penalty = tf.reduce_sum(tf.get_collection('Bound_penalties'))
        P_theta = tf.reduce_sum(tf.get_collection('Threshold_penalties'))
        penalty_loss = P_theta + bound_penalty
        mse_loss = tf.losses.mean_squared_error(labels, predictions)
        normal_loss = tf.losses.get_total_loss() + P_theta
        loss = penalty_loss if penalty_flag else normal_loss
        train_accuracy = tf.identity(
            tf.metrics.percentage_below(values=tf.abs(labels - predictions), threshold=0.02)[1], name='train_accuracy')
        tf.summary.scalar('total_loss', loss, family='losses')
        tf.summary.scalar('MSE_loss', mse_loss, family='losses')  # inaccurate for penalty epochs (ignore)
        tf.summary.scalar('Penalty_Loss', penalty_loss, family='losses')
        tf.summary.scalar("Regularization_loss", reg_loss, family='losses')
        tf.summary.scalar('train_acc', train_accuracy, family='accuracies')  # inaccurate for penalty epochs (ignore)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=tf.train.AdamOptimizer(params['learning_rate'], beta1=params['beta1']).minimize(loss, global_step))
    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.sqrt(tf.losses.mean_squared_error(labels, predictions))
        eval_acc_metric = tf.metrics.percentage_below(values=tf.abs(labels - predictions), threshold=0.02)
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL, loss=loss,
                                          eval_metric_ops={'eval_accuracy': eval_acc_metric})


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    runtime_params = update_runtime_params(sys.argv, default_params)
    metadata = extract_metadata(runtime_params['train_val_file'], runtime_params['test_file'])
    run_config = get_run_config(runtime_params['kill_summaries'])
    eqlearner = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=runtime_params['model_dir'],
                                       params=runtime_params)
    logging_hook = tf.train.LoggingTensorHook(tensors={'train_accuracy': 'train_accuracy'}, every_n_iter=1000)
    evaluation_hook = set_evaluation_hook(**runtime_params)
    max_episode = get_max_episode(**runtime_params)

    train_input, penalty_train_input, val_input, test_input = get_input_fns(**runtime_params, **metadata)
    print('One train episode equals %d normal epochs and 1 penalty epoch.' % runtime_params['penalty_every'])
    for train_episode in range(1, max_episode + 1):
        print('Train episode: %d out of %d.' % (train_episode, max_episode))
        penalty_flag = True
        eqlearner.train(input_fn=penalty_train_input)
        penalty_flag = False
        eqlearner.train(input_fn=train_input, hooks=[logging_hook])
    print('Training complete. Evaluating...')
    val_results = eqlearner.evaluate(input_fn=val_input, name='validation', hooks=[evaluation_hook])
    results = dict(val_error=val_results['loss'], complexity=evaluation_hook.get_complexity())
    if test_input is not None:  # test_input function is only provided if extrapolation data is given
        extr_results = eqlearner.evaluate(input_fn=test_input, name='extrapolation')
        results['extr_error'] = extr_results['loss']
    save_results(results, runtime_params)
    print('Model evaluated. Results:\n', results)
