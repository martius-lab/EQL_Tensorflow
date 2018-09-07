""" Useful Routines used in EQL. """
import csv
import inspect
import json
from ast import literal_eval
from itertools import accumulate
from os import path

import numpy as np
import tensorflow as tf

# The following parameters should not be changed in most cases.
network_parameters = {'train_val_split': .9,  # how data in train_val_file is split, .9 means 90% train 10% validation
                      'layer_width': 10,  # number of identical nodes per hidden layer
                      'batch_size': 20,  # size of data batches used for training
                      'learning_rate': 5e-4,
                      'beta1': .4,
                      'l0_threshold': .05,  # threshold for regularization, see paper: chapter 2.3 Reg Phases
                      'reg_scale': 1e-5,
                      'reg_sched': (.25, .95),  # (reg_start, reg_end)
                      'output_bound': None,  # output boundary for penalty epochs, if set to None it is  calculated
                      # from training/validation data
                      'weight_init_param': 1.,
                      'test_div_threshold': 1e-4,  # threshold for denominator in division layer used when testing
                      'complexity_threshold': 0.01,  # determines how small a weight has to be to be considered inactive
                      'penalty_every': 50,  # feed in penalty data for training and evaluate after every n epochs
                      'penalty_bounds': None,  # domain boundaries for generating penalty data, if None it is calculated
                      # from extrapolation_data (if provided) or training/validation data
                      'network_init_seed': None,  # seed for initializing weights in network
                      }


def update_runtime_params(argv, params):
    """Routine to update the default parameters with network_parameters and parameters from commandline."""
    params.update(network_parameters)
    if len(argv) > 1:
        params.update(literal_eval(argv[1]))
    params['model_dir'] = path.join(params['model_base_dir'], str(params['id']))
    return params


def get_max_episode(num_h_layers, epoch_factor, penalty_every, **_):
    """Routine to calculate the total number of training episodes
    (1 episode = 1 penalty epoch + *penalty_every* normal epochs"""
    max_episode = (num_h_layers * epoch_factor) // penalty_every
    if max_episode == 0:
        raise ValueError('Penalty_every has to be smaller than the total number of epochs.')
    return max_episode


def step_to_epochs(global_step, batch_size, train_examples, **_):
    epoch = tf.div(global_step, int(train_examples / batch_size)) + 1
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


def weight_name_for_i(i, weight_type):
    if i == 0:
        return 'dense/{}:0'.format(weight_type)
    return 'dense_{}/{}:0'.format(i, weight_type)


def save_results(results, params):
    """
    Routine that saves the results as a csv file.
    :param results: dictionary containing evaluation results
    :param params: dict of runtime parameters
    """
    results['id'] = params['id']
    results_file = path.join(params['model_dir'], 'results.csv')
    with open(path.join(params['model_dir'], 'parameters.json'), 'w') as f:
        json.dump(params, f, sort_keys=True, indent=4)
    save_dict_as_csv(results, results_file)


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
