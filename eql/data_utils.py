"""
Handling of data related tasks, e.g. reading of input and generating data files.
    - *data_from file* and *data_from_pickle_dict* extract input and output data from data file.
    - *get_penalty_data generates new input data (using penalty boundaries) for penalty epochs. The output fed into
    the Estimator for these epochs is set to zero because in penalty epochs we compute gradients based only on the
    output calculated by the EQL, not the expected output (no MSE or similar is calculated).
    - *get_input_fns* generates train-, validation- and test-input for EQL from given data
    - *extract_metadata* extracts important information from the given data-files
    - *files_from_fn* generates a data files containing training/validation and extrapolation-data for a fn
    passed through the input in parameter dictionary. The python function has to be defined in data_utils.py.
   *files_from_fn* is also called when *data_utils.py* is run from command line with the parameter dictionary passed as
   a string.
"""
import gzip
import os.path
import pickle
from ast import literal_eval
from sys import argv, stderr

import numpy as np
import tensorflow as tf

from .utils import to_float32, number_of_positional_arguments

"""Equation 1-4 from the paper. Equation 5 describes the cart pendulum from the paper."""


def F1(x1, x2, x3, x4):
    """Requires 1 hidden layer."""
    y0 = (np.sin(np.pi * x1) + np.sin(2 * np.pi * x2 + np.pi / 8.0) + x2 - x3 * x4) / 3.0
    return y0,


def F2(x1, x2, x3, x4):
    """Requires 2 hidden layers."""
    y0 = (np.sin(np.pi * x1) + x2 * np.cos(2 * np.pi * x1 + np.pi / 4.0) + x3 - x4 * x4) / 3.0
    return y0,


def F3(x1, x2, x3, x4):
    """Requires 2 hidden layers."""
    y0 = ((1.0 + x2) * np.sin(np.pi * x1) + x2 * x3 * x4) / 3.0
    return y0,


def F4(x1, x2, x3, x4):
    """Requires 3 hidden layers."""
    y0 = 0.5 * (np.sin(np.pi * x1) + np.cos(2.0 * x2 * np.sin(np.pi * x1)) + x2 * x3 * x4)
    return y0,


def F5(x1, x2, x3, x4):
    """Equation for cart pendulum. Requires 4 hidden layers."""
    y1 = x3
    y2 = x4
    y3 = (-x1 - 0.01 * x3 + x4 ** 2 * np.sin(x2) + 0.1 * x4 * np.cos(x2) + 9.81 * np.sin(x2) * np.cos(x2)) \
         / (np.sin(x2) ** 2 + 1)
    y4 = -0.2 * x4 - 19.62 * np.sin(x2) + x1 * np.cos(x2) + 0.01 * x3 * np.cos(x2) - x4 ** 2 * np.sin(x2) * np.cos(x2) \
         / (np.sin(x2) ** 2 + 1)
    return y1, y2, y3, y4,


data_gen_params = {'file_name': 'F1data',  # file name for the generated data file, will be created in data/file_name
                   'fn_to_learn': 'F1',  # python function to learn, should be defined in data_utils
                   'train_val_examples': 10000,  # total number of examples for training and validation
                   'train_val_bounds': (-1.0, 1.0),  # domain boundaries for validation and training normal epochs
                   'test_examples': 5000,  # number of test examples, if set to None no test_data file is created
                   'test_bounds': (-2.0, 2.0),  # domain boundaries for test data
                   'noise': 0.01,
                   'seed': None
                   }


def generate_data(fn, num_examples, bounds, noise, seed=None):
    np.random.seed(seed)
    lower, upper = bounds
    input_dim = number_of_positional_arguments(fn)
    xs = np.random.uniform(lower, upper, (num_examples, input_dim)).astype(np.float32)
    xs_as_list = np.split(xs, input_dim, axis=1)
    ys = fn(*xs_as_list)
    ys = np.concatenate(ys, axis=1)
    ys += np.random.uniform(-noise, noise, ys.shape)
    return xs, ys


def data_from_file(filename, split=None):
    """
    Routine extracting and splitting data from given file.
    :param filename: path to the file data should be extracted from, data should look like (input_array, label_array)
    :param split: float or None
    :return: chunk(s) of data
    """
    data = to_float32(pickle.load(gzip.open(filename, "rb"), encoding='latin1'))
    if split is not None:
        data = split_data(data, split)
    return data


def data_from_pickle_dict(filename, keys, split=None):
    """
    Routine extracting and splitting data from given file.
    :param filename: path to the file data should be extracted from, data should be a dict of features
    :param keys: tuple of tuples, first tuple are keys used for input, second tuple are keys used for labels
    :param split: float or None
    :return: chunk(s) of data
    """
    with open(filename, 'rb') as file:
        raw_data = pickle.load(file)
    input_data = np.concatenate(tuple(to_float32(raw_data[key]) for key in keys[0]), axis=1)
    output_data = np.concatenate(tuple(to_float32(raw_data[key]) for key in keys[1]), axis=1)
    data = (input_data, output_data)
    if split is not None:
        data = split_data(data, split)
    return data


def split_data(data, split):
    """
    Routine splitting given data in two chunks while maintaining the internal input/label structure.
    :param data: data to split
    :param split: float. if split is not None, the data is split into two chunks, one of size split*num_examples and one
                  of size (1-split)*num_examples. If it is None, all data is returned as one chunk
    :return: data split into two chunks
    """
    split_point = int(len(data[0]) * split)
    data = [np.split(dat, [split_point]) for dat in data]
    return zip(*data)


def ds_from_generator(data):

    def generator():
        for sample in zip(*data):
            yield sample

    return tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32),
                                          output_shapes=(tf.TensorShape((data[0].shape[-1],)),
                                                         tf.TensorShape((data[1].shape[-1],))))


def input_from_data(data, batch_size, repeats, data2=None):
    """
    Function turning data into input for the network. Provides enough data for *repeats* epochs.
    :param data: numpy array of data
    :param batch_size: size of batch returned, only relevant for training regime
    :param repeats: integer factor determining how many times (epochs) data is reused
    :param data2: more potential data to use in same batch as data
    :return: *repeats* times data split into inputs and labels in batches
    """
    if data2 is not None:
        eff_batch_size = batch_size - 1
    else:
        eff_batch_size = batch_size

    ds1 = ds_from_generator(data).shuffle(buffer_size=1000).batch(eff_batch_size).repeat(repeats)

    if data2 is not None:
        ds2 = ds_from_generator(data2).batch(1).repeat(repeats)
        ds = tf.data.Dataset.zip((ds1, ds2))
        (x1, y1), (x2, y2) = ds.make_one_shot_iterator().get_next()
        return tf.concat([x1, x2], axis=0), tf.concat([y1, y2], axis=0)
    else:
        return ds1.make_one_shot_iterator().get_next()


def get_penalty_data(num_examples, penalty_bounds, num_inputs, num_outputs):
    """
    Function returning penalty data. In penalty epoch labels are irrelevant, therefore labels are set to zero.
    Only provides enough data to train for one epoch.
    :param num_examples: Total number of examples to be trained in penalty epoch.
    :param penalty_bounds: tuple of Boundaries to be used to generate penalty data
    """
    if isinstance(penalty_bounds, tuple):
        lower, upper = penalty_bounds
    else:
        lower, upper = zip(*penalty_bounds)
    xs = np.random.uniform(lower, upper, (num_examples, num_inputs)).astype(np.float32)
    ys = np.zeros((num_examples, num_outputs), dtype=np.float32)
    return xs, ys


def get_train_input_fns_from_ndarrays(num_epochs, inputs, outputs, train_val_split, batch_size, **_):
    data = (inputs, outputs)
    train_data, val_data = split_data(data, train_val_split)
    # add aditional penalty data (as data2) to the batch of training data.
    # used to force EQL solutions be bounded with penalty_bounds in extrapolation region
    train_input = lambda: input_from_data(data=train_data, batch_size=batch_size, repeats=num_epochs)
    val_input = lambda: input_from_data(data=val_data, batch_size=batch_size, repeats=1)
    return train_input, val_input


def get_input_fns(num_epochs, train_val_split, batch_size, train_val_file, test_file, num_inputs, num_outputs,
                  train_val_examples, penalty_bounds, extracted_penalty_bounds, keys=None, **_):
    """
    Routine to determine which input function to use for training(normal or penalty epoch) / validation / testing.
    :param num_epochs: how many epoch should data be provided for
    :param train_val_split: float specifying the data split, .8 means 80% of data is used for training, 20% for val
    :param batch_size: Size of batches used for training (both in normal and penalty epochs).
    :param train_val_file: Path to file containing training and validation data.
    :param test_file: Path to file containing test data. If several test data, list of pathes.
    :param num_inputs: number of input arguments
    :param num_outputs: number of outputs
    :param train_val_examples: number of examples to use for training and validation
    :param penalty_bounds: default domain boundaries used to generate penalty epoch training data.
    :param extracted_penalty_bounds: domain boundaries for penalty data generation extracted from data files
    :param keys: tuple of tuples containing input/label keys if input file contains a dict of data, otherwise None
    :return: functions returning train-, penalty_train-, validation- and (if provided in datafile) test-input
             if no extrapolation test data is provided test_input is None
    """
    penalty_bounds = penalty_bounds or extracted_penalty_bounds
    if keys is None:
        train_data, val_data = data_from_file(filename=train_val_file, split=train_val_split)
    else:
        train_data, val_data = data_from_pickle_dict(filename=train_val_file, keys=keys, split=train_val_split)
    penalty_data = get_penalty_data(num_examples=int(train_val_examples * train_val_split),
                                    penalty_bounds=penalty_bounds, num_inputs=num_inputs, num_outputs=num_outputs)

    # add aditional penalty data (as data2) to the batch of training data.
    # used to force EQL solutions be bounded with penalty_bounds in extrapolation region
    train_input = lambda: input_from_data(data=train_data, batch_size=batch_size, repeats=num_epochs,
                                          data2=penalty_data)
    val_input = lambda: input_from_data(data=val_data, batch_size=batch_size, repeats=1)
    if isinstance(test_file, str):
        test_data = data_from_file(test_file)
        test_input = lambda: input_from_data(data=test_data, batch_size=batch_size, repeats=1)
    elif isinstance(test_file, tuple) or isinstance(test_file, list):
        test_input = []
        for test in test_file:
            test_data = data_from_file(test)
            test_input.append(lambda: input_from_data(data=test_data, batch_size=batch_size, repeats=1))
    else:
        test_input = None
    return train_input, val_input, test_input


def extract_metadata_from_array(train_val_data, test_data, domain_bound_factor=2, res_bound_factor=20, keys=None, **_):
    """
    Routine to extract additional information about data from data file.
    :param train_val_file: Path to training/validation data file
    :param test_file: Path to extrapolation data file
    :param domain_bound_factor: factor to scale the domain boundary of train/val data to get penalty data boundary
    :param res_bound_factor: factor to scale the maximum output of train/val data to get penalty data result boundary
    :param keys: tuple of tuples containing input/label keys if input file contains a dict of data, otherwise None
    :return: metadata dict
    """
    train_val_examples = train_val_data[0].shape[0]
    num_inputs = train_val_data[0].shape[1]
    num_outputs = train_val_data[1].shape[1]
    if test_data is not None:
        extracted_penalty_bounds = list(zip(np.min(test_data[0], axis=0), np.max(test_data[0], axis=0)))
        extracted_output_bound = np.max(np.abs(test_data[1])) * res_bound_factor
    else:
        extracted_penalty_bounds = list(zip(np.min(train_val_data[0], axis=0) * domain_bound_factor,
                                            np.max(train_val_data[0], axis=0) * domain_bound_factor))
        extracted_output_bound = np.max(np.abs(train_val_data[1])) * res_bound_factor
    metadata = dict(train_val_examples=train_val_examples, num_inputs=num_inputs, num_outputs=num_outputs,
                    extracted_output_bound=extracted_output_bound, extracted_penalty_bounds=extracted_penalty_bounds)
    return metadata

def extract_metadata(train_val_file, test_file, domain_bound_factor=2, res_bound_factor=20, keys=None, **_):
    """
    Routine to extract additional information about data from data file.
    :param train_val_file: Path to training/validation data file
    :param test_file: Path to extrapolation data file
    :param domain_bound_factor: factor to scale the domain boundary of train/val data to get penalty data boundary
    :param res_bound_factor: factor to scale the maximum output of train/val data to get penalty data result boundary
    :param keys: tuple of tuples containing input/label keys if input file contains a dict of data, otherwise None
    :return: metadata dict
    """
    if keys is None:
        train_val_data = data_from_file(filename=train_val_file)
    else:
        train_val_data = data_from_pickle_dict(filename=train_val_file, keys=keys)
    train_val_examples = train_val_data[0].shape[0]
    num_inputs = train_val_data[0].shape[1]
    num_outputs = train_val_data[1].shape[1]
    if isinstance(test_file, tuple) or isinstance(test_file, list):
        test_file = test_file[0]
    if test_file is not None:
        if keys is None:
            test_data = data_from_file(filename=test_file)
        else:
            test_data = data_from_pickle_dict(filename=test_file, keys=keys)
        extracted_penalty_bounds = list(zip(np.min(test_data[0], axis=0), np.max(test_data[0], axis=0)))
        extracted_output_bound = np.max(np.abs(test_data[1])) * res_bound_factor
    else:
        extracted_penalty_bounds = list(zip(np.min(train_val_data[0], axis=0) * domain_bound_factor,
                                            np.max(train_val_data[0], axis=0) * domain_bound_factor))
        extracted_output_bound = np.max(np.abs(train_val_data[1])) * res_bound_factor
    metadata = dict(train_val_examples=train_val_examples, num_inputs=num_inputs, num_outputs=num_outputs,
                    extracted_output_bound=extracted_output_bound, extracted_penalty_bounds=extracted_penalty_bounds)
    return metadata


def files_from_fn(file_name, fn_to_learn, train_val_examples, test_examples, train_val_bounds,
                  test_bounds, noise, seed=None):
    """
    Routine generating .gz file with train-, validation, test and meta-data from function.
    It is worth noting that that the function is saved as a string in metadata.
    :param file_name: Name of the data file to be created. It is being saved in the directory 'data'.
    :param fn_to_learn: string name of python function used to generate data. Should be defined in data_utils.py.
    :param train_val_examples: Total number of examples used for training and validation.
    :param train_val_bounds: Boundaries used to generate training and validation data.
    :param test_examples: Total number of examples used for testing.
    :param test_bounds: Boundaries used to generate test data.
    """
    fn_to_learn = globals()[fn_to_learn]
    if not os.path.exists('data'):
        os.mkdir('data')
    train_val_set = generate_data(fn=fn_to_learn, num_examples=train_val_examples, bounds=train_val_bounds, noise=noise,
                                  seed=seed)
    train_val_data_file = os.path.join('data', file_name + '_train_val')
    pickle.dump(train_val_set, gzip.open(train_val_data_file, "wb"))
    print('Successfully created train/val data file in %s.' % train_val_data_file)

    if test_examples is not None:
        test_set = generate_data(fn=fn_to_learn, num_examples=test_examples, bounds=test_bounds, noise=noise, seed=seed)
        test_data_file = os.path.join('data', file_name + '_test')
        pickle.dump(test_set, gzip.open(test_data_file, "wb"))
        print('Successfully created test data file in %s.' % test_data_file)


if __name__ == '__main__':
    if len(argv) > 1:
        print('Updating default parameters.')
        data_gen_params.update(literal_eval(argv[1]))
    else:
        print('Using default parameters.')
    files_from_fn(**data_gen_params)
