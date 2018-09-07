"""
Generation of job files for model selection.
    - *generate_jobs* creates shell script files for multiple jobs. Varying the number of hidden layers and other
    parameters allows for effective model selection. Also creates a submission file to submit all jobs.
"""
import os
from ast import literal_eval
from sys import argv

import numpy as np

job_dir = 'jobs'
result_dir = os.path.join("results", "model_selection")
submitfile = os.path.join('jobs', 'submit.sh')


def generate_jobs(train_val_file, test_file):
    if not os.path.exists(job_dir):
        os.mkdir(job_dir)
    with open(submitfile, 'w') as submit:
        pwd = os.getcwd()
        id = 0
        l1_reg_range = 10 ** (-np.linspace(35, 60, num=10) / 10.0)
        for l1_reg_scale in l1_reg_range:
            for num_h_layers in [1, 2, 3, 4]:
                params = dict(model_base_dir=result_dir, train_val_file=train_val_file, test_file=test_file, id=id,
                              num_h_layers=num_h_layers, reg_scale=l1_reg_scale, kill_summaries=True,
                              generate_symbolic_expr=False)
                dict_str = str(params)
                cmd = '{} "{}"'.format('python3 ' + os.path.join(pwd, 'train.py '), dict_str)
                script_fname = os.path.join(job_dir, str(id) + ".sh")
                submit.write(str(id) + ".sh" + "\n")
                with open(script_fname, 'w') as f:
                    f.write(cmd)
                os.chmod(script_fname, 0o755)  # makes script executable
                id += 1
    os.chmod(submitfile, 0o755)  # makes script executable
    print('Jobs succesfully generated.')


if __name__ == '__main__':
    if len(argv) > 1:
        passed_dict = literal_eval(argv[1])
        train_val_file = passed_dict['train_val_file']
        test_file = passed_dict['test_file']
    else:
        raise ValueError('No dict given. Please pass a dict {"train_val_file": ..., "test_file": ...}.')
    print('Using %s and %s as paths to data files.' % (train_val_file, test_file))
    generate_jobs(train_val_file=train_val_file, test_file=test_file)
