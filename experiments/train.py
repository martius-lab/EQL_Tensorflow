""" Neural Network Estimator for EQL - Equation Learner """
import sys
import os, glob
from collections import defaultdict

import tensorflow as tf


from eql.data_utils import get_input_fns, extract_metadata
from eql.model import ModelFn
from eql.evaluation import EvaluationHook
from eql.utils import get_run_config, save_results, update_runtime_params, \
                  evaluate_learner

# more network parameters are loaded from utils.py
default_params = {'model_dir': 'results/2',
                  'id': 1,  # job_id to identify jobs in result metrics file, separate model_dir for each id
                  'train_val_file': '../data/F1data_train_val',  # Datafile containing training, validation data
                  'test_file': '../data/F1data_test',  # Datafile containing test data, if set to None no test data is used
                  'epochs_first_reg': 100,
                  'epochs_per_reg': 100,
                  'num_h_layers': 1,  # number of hidden layers used in network
                  'generate_symbolic_expr': True,  # saves final network as a latex png and symbolic graph
                  'kill_summaries': False,  # reduces data generation, deleting all the graphs, recommended when creating many jobs
                  }

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    runtime_params = update_runtime_params(sys.argv, default_params)
    run_config = get_run_config(runtime_params['kill_summaries'])
    evaluation_hook = EvaluationHook(store_path=runtime_params['model_dir'])
    model_fn = ModelFn(config=run_config, evaluation_hook=evaluation_hook)
    metadata = extract_metadata(**runtime_params)
    model_fn.set_metadata(metadata=metadata)
    eqlearner = tf.estimator.Estimator(model_fn=model_fn, config=model_fn.config, model_dir=runtime_params['model_dir'],
                                       params=runtime_params)
    logging_hook = tf.train.LoggingTensorHook(tensors={'train_accuracy': 'train_accuracy'}, every_n_iter=1000)
    results = defaultdict(list)
    train_input, val_input, test_input = get_input_fns(num_epochs=runtime_params['evaluate_every'], **runtime_params, **model_fn.metadata)
    print('One train episode equals %d epochs.' % runtime_params['evaluate_every'])
    for i, reg_scale in  enumerate(runtime_params['reg_scale']):
        print('Regularizing with scale %s' % str(reg_scale))
        model_fn.set_reg_scale(reg_scale)
        if i == 0 :
            max_episode = runtime_params['epochs_first_reg'] // runtime_params['evaluate_every']
        else :
            max_episode = runtime_params['epochs_per_reg'] // runtime_params['evaluate_every']
        if max_episode == 0:
            raise ValueError('evaluate_every has to be smaller than the total number of epochs.')
        for train_episode in range(1, max_episode + 1):
            print('Regularized train episode with scale %s: %d out of %d.' % (str(reg_scale), train_episode, max_episode))
            eqlearner.train(input_fn=train_input, hooks=[logging_hook])
            val_results = eqlearner.evaluate(input_fn=val_input, name='validation')
            if (i == 0) and (val_results['eval_accuracy'] > runtime_params['val_acc_thresh']):
                print('Reached accuracy of %d, starting regularization.' % val_results['eval_accuracy'])
                break
        results = evaluate_learner(learner=eqlearner, res=results,
                                   eval_hook=model_fn.evaluation_hook, val_input=val_input,
                               test_input=test_input, reg_scale=reg_scale)
    save_results(results, runtime_params)
    print('Model evaluated. Results:\n', results)
    if runtime_params['kill_summaries']:
        all_events_files = glob.glob(os.path.join(runtime_params['model_dir'], 'events.out.tfevents*'))
        for file in all_events_files:
            os.remove(file)
