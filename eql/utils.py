""" Useful Routines used in EQL. """
import csv
import inspect
from itertools import accumulate
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

try:
   from cluster import save_metrics_params, update_params_from_cmdline
except ImportError:
   print('Cluster package missing, importing local copy of setting code.')
   from .settings import save_metrics_params, update_params_from_cmdline


# The following parameters should not be changed in most cases.
network_parameters = {
    'train_val_split': .9,  # how data in train_val_file is split, .9 means 90% train 10% validation
    'layer_width': 10,  # number of identical nodes per hidden layer
    'learning_rate': 5e-2,
    'beta1': .4,  # training parameter for Adam optimizer
    'L0_beta': 2 / 3,  # parameter for L0 regularization
    'evaluate_every': 5,  # feed in penalty data for training and evaluate after every n epochs
    'reg_scale': (1e-10, 1e-3),  # schedule of regularization constants
    # The following parameters usually don't need to be changed or optimized
    'batch_size': 20,  # size of data batches used for training
    'weight_init_param': 1.,  # initialization constant for weights, should be in [-1, 1] range
    'test_div_threshold': 1e-4,  # threshold for denominator in division layer used when testing
    'network_init_seed': None,  # seed for initializing weights in network
    'val_acc_thresh': .98,  # accuracy threshold at which, if reached, unregularized training stops
    'keys': None,  # (('observations', 'actions'), ('next_observations',)) used for data consisting of dictionaries
    'output_bound': None,
    # output boundary for penalty epochs, if None it is automatically calculated from test or train/val data
    'penalty_bounds': None,
    # domain boundaries for generating penalty data, if None it is automatically calculated from test or train/val data
    }


def update_runtime_params(argv, params):
    """Routine to update the default parameters with network_parameters and parameters from commandline."""
    params.update(network_parameters)
    params['use_cluster'] = len(argv) > 1
    if params['use_cluster']:
        params = update_params_from_cmdline(default_params=params)
    return params


def step_to_epochs(global_step, batch_size, train_examples, **_):
    if train_examples > batch_size:
        epoch = tf.div(global_step, int(train_examples / batch_size)) + 1
    else:
        #TODO: add warning that the data set size is smaller than the batch size
        epoch = global_step
    return epoch


def to_float32(list_of_arrays):
    return tuple([arr.astype(np.float32) for arr in list_of_arrays])


def number_of_positional_arguments(fn):
    params = [value.default for key, value in inspect.signature(fn).parameters.items()]
    return sum(1 for item in params if item == inspect.Parameter.empty)


def get_run_config(kill_summaries):
    """
    Creates run config for Estimator.
    :param kill_summaries: Boolean flag, if set to True run_config prevents creating too many checkpoint files.
    """
    if kill_summaries:
        checkpoint_args = dict(save_summary_steps=None, save_checkpoints_secs=None, save_checkpoints_steps=1e8)
    else:
        checkpoint_args = dict(save_summary_steps=1000)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1)
    run_config = tf.estimator.RunConfig().replace(log_step_count_steps=1000, keep_checkpoint_max=1,
                                                  session_config=session_conf, **checkpoint_args)
    return run_config


def save_results(results, params):
    """
    Routine that saves the results as a csv file. Also generates a reg_constant-complexity-graph.
    :param results: dictionary containing evaluation results
    :param params: runtime parameters
    """
    use_extrapolation = True # fix it
    best = dict(best_val_error=min(results['val_error']))
    if use_extrapolation:
        if isinstance(params["test_file"], tuple) or isinstance(params["test_file"], list):
            for i in range(len(params["test_file"])):
                best['best_extr_error_{}'.format(i)] = min(results['extr_error_{}'.format(i)])
        else:
            best['best_extr_error'] = min(results['extr_error'])
    results.update(best)

    if params['use_cluster']:
        save_metrics_params(metrics=best, params=params, save_dir=params['model_dir'])

    results['id'] = params['id']
    results_file = path.join(params['model_dir'], 'results.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file)

    grouped_results = results_df.groupby(by='complexity').min()
    complexity_graph = plt.figure()
    plt.ylabel('error')
    grouped_results.val_error.plot()
    # if use_extrapolation:
    #     grouped_results.extr_error.plot()
    plt.legend()
    complexity_graph.savefig(path.join(params['model_dir'], 'complexity_graph.png'))


def save_dict_as_csv(dict_to_save, file_path):
    with open(file_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=dict_to_save.keys())
        writer.writeheader()
        writer.writerow(dict_to_save)


def yield_with_repeats(iterable, repeats):
    """ Yield the ith item in iterable repeats[i] times. """
    it = iter(iterable)
    for num in repeats:
        new_val = next(it)
        for i in range(num):
            yield new_val


def yield_equal_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def iter_by_chunks(lst, chunk_lens):
    """ Split list into groups of given size and return an iterator of the groups.
    Example iter_by_chunks([1, 2, 3, 4], [1, 0, 0, 2, 1]) = ([1], [], [], [2, 3], [4]).
    :param lst: a list
    :param chunk_lens: a list specifying lengths of individual chunks
    :return: a generator object yielding one chunk at a time
    """
    splits = [0] + list(accumulate(chunk_lens))
    for beg, end in zip(splits[:-1], splits[1:]):
        yield lst[beg:end]


def generate_arguments(all_args, repeats, arg_nums):
    """
    Split all args into chunks for functions. Example:
    generate_arguments([0,1,2,3,4,5,6,7,8,9,10], [1, 3, 1], [2, 2, 3]) -> [(0,1), (2,5), (3,6), (4,7), (8,9,10)]
    :param all_args: list of all arguments
    :param repeats: list of number of repeats for each function group
    :param arg_nums: list of number of inputs for each function group
    :return a generator object yielding one chunk at a time
    """
    lengths = (a * b for a, b in zip(repeats, arg_nums))
    all_chunks = iter_by_chunks(all_args, lengths)
    for big_chunk, repeat in zip(all_chunks, repeats):
        yield from zip(*yield_equal_chunks(big_chunk, repeat))


def get_div_thresh_fn(is_training, batch_size, test_div_threshold, train_examples, **_):
    """
    Returns function to calculate the division threshold from a given step.
    :param is_training: Boolean to decide if training threshold or test threshold is used.
    """
    if is_training:
        def get_div_thresh(step):
            epoch = step_to_epochs(global_step=step, batch_size=batch_size, train_examples=train_examples)
            return 1. / tf.sqrt(tf.cast(epoch, dtype=tf.float32))
    else:
        def get_div_thresh(step):
            return test_div_threshold
    return get_div_thresh


def tensorboard_summarize(tensor, collections=None, family=None):
    """
    Save scalar summary for given tensor/s
    :param tensor: a tf tensor or potentially nested list of tensors
    """
    if isinstance(tensor, list):
        for ten in tensor:
            tensorboard_summarize(ten, collections=collections, family=family)
    else:
        assert isinstance(tensor, tf.Tensor)
        tensor_name = tensor.name.replace(':', '_')
        tf.summary.scalar(tensor_name, tensor, collections=collections, family=family)


def evaluate_learner(learner, res, eval_hook, val_input, test_input, reg_scale):
    """
    Routine to evaluate a learner and extract results.
    :param learner: An estimator.
    :param res: A defaultdict containing the results.
    :param eval_hook: A hook to evaluate learner, conatining a function 'get_complexity()'
    :return: updated results
    """
    val_results = learner.evaluate(input_fn=val_input, name='validation', hooks=[eval_hook])
    res['val_error'].append(val_results['loss'])
    res['complexity'].append(eval_hook.get_complexity())
    res['reg_scale'].append(reg_scale)
    if isinstance(test_input, list):
        for i, test in enumerate(test_input):
            extr_results = learner.evaluate(input_fn=test, name='extrapolation_{}'.format(i))
            res['extr_error_{}'.format(i)].append(extr_results['loss'])
    elif test_input is not None:  # test_input function is only provided if extrapolation data is given
        extr_results = learner.evaluate(input_fn=test_input, name='extrapolation')
        res['extr_error'].append(extr_results['loss'])
    return res
