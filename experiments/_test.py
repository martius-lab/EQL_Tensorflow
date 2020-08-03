# %%

import sys
import os, glob
from collections import defaultdict

import tensorflow as tf


from eql.data_utils import get_input_fns, extract_metadata
from eql.model import ModelFn
from eql.evaluation import EvaluationHook
from eql.utils import get_run_config, save_results, update_runtime_params, \
                  evaluate_learner


# %%
# more network parameters are loaded from utils.py
import json
#Read JSON data into the datastore variable
filename = "./settings/test_F1.json"
with open(filename, 'r') as f:
        runtime_params  = json.load(f)
runtime_params
default_params['reg_scale'] = [10**(-10), 10**(-3)]
tf.logging.set_verbosity(tf.logging.INFO)

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

# %%
from eql.evaluation import generate_symbolic_expression

model_fn.get_weights()
model_fn.get_fns_list()
kernels, biases = zip(*model_fn.weights)


res, numba_res = generate_symbolic_expression(kernels, biases, model_fn.fns_list, 3)
