import os
import shutil
from pathlib2 import Path

from cluster import cluster_run, execute_submission, init_plotting
from cluster.distributions import *
from cluster.report import produce_basic_report
from cluster.utils import mkdtemp
import numpy as np

home = str(Path.home())

init_plotting()

submission_name = 'EQL0_passive_cartpole_gridsearch'
project_path = mkdtemp(suffix=submission_name + '-' + 'project')
results_path = os.path.join(home, 'experiments/results')
jobs_path = mkdtemp(suffix=submission_name + '-' + 'jobs')

git_params = dict(url='git@gitlab.tuebingen.mpg.de:abhattacharjee/EQL_Tensorflow.git',
                  local_path=project_path,
                  branch='EQL0',
                  commit=None,
                  remove_local_copy=True,
                  )

paths_and_files = dict(script_to_run=os.path.join(project_path, 'train.py'),
                       result_dir=os.path.join(results_path, submission_name),
                       jobs_dir=jobs_path,
                       custom_pythonpaths=['/home/azadaianchuk/eql_cluster/lib/python3.6/site-packages/'])

submission_requirements = dict(request_cpus=1,
                               request_gpus=0,
                               cuda_requirement=None,  # 'x.0' or None
                               memory_in_mb=4000,
                               bid=5)

other_params = {'train_val_file': '/home/azadaianchuk/eql_data,
                'test_file': '/home/azadaianchuk/eql_data,
                'epochs_first_reg': 600,
                'epochs_per_reg': 600,
                # 'num_h_layers': 2,
                'generate_symbolic_expr': False,
                'kill_summaries': True,
                'train_val_split': .9,
                'layer_width': 10,
                # 'learning_rate': 5e-3,
                'beta1': .4,
                'L0_beta': 2 / 3,
                'evaluate_every': 100,
                # 'reg_scale': (1e-4, 3e-4, 8e-4, 1e-3, 3e-3, 8e-3, 1e-2, 3e-2, 8e-2, .1, .5),
                'batch_size': 20,
                'weight_init_param': 1.,
                'test_div_threshold': 1e-4,
                'val_acc_thresh': .98,
                # None-parameters cannot be passed without error right now, doesn't matter because they are set to None
                # in original code
                # 'network_init_seed': None,
                # 'keys': None,
                # 'output_bound': None,
                # 'penalty_bounds': None,
                }

reg_scales = [ (1e-10, reg_scale) for reg_scale in np.logspace(-6, -3, 20)]
hyperparam_dict = {'learning_rate': [5e-4, 1e-3, 5e-3],
                   'reg_scale': reg_scales,
                   'num_h_layers': [3,4]}

submit = True

all_args = dict(submission_name=submission_name,
                paths=paths_and_files,
                submission_requirements=submission_requirements,
                hyperparam_dict=hyperparam_dict,
                other_params=other_params,
                samples=None,
                restarts_per_setting=5,
                smart_naming=True,
                git_params=git_params
               )

if __name__ == '__main__':
  submission = cluster_run(**all_args)
  if submit:
    df, all_params, metrics, submission_hook_stats = execute_submission(submission, paths_and_files['result_dir'])
    df.to_csv(os.path.join(paths_and_files['result_dir'], 'results_raw.csv'))

    relevant_params = list(hyperparam_dict.keys())
    output_pdf = os.path.join(paths_and_files['result_dir'], '{}_report.pdf'.format(submission_name))
    produce_basic_report(df, relevant_params, metrics, submission_hook_stats=submission_hook_stats,
                         procedure_name=submission_name, output_file=output_pdf)

  # copy this script to the result dir
  my_path = os.path.realpath(__file__)
  shutil.copy(my_path, paths_and_files['result_dir'])
