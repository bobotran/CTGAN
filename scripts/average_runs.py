import shutil
import os
import pickle

exp_name = 'baseline_likelihood'
num_trials = 5

accuracy, f1, syn_likelihood, test_likelihood = [], [], [], []

consolidated_dir = os.path.join('logs/', exp_name)
os.makedirs(consolidated_dir)
for i in range(1, num_trials + 1):
    name = exp_name + str(i)
    path = os.path.join('logs/', name)

    try:
        fp = os.path.join(path, 'accuracy.pickle')
        with open(fp, 'rb') as f:        
            accuracy += pickle.load(f)
    except FileNotFoundError:
        print('No accuracy scores found. Skipping.')

    try:
        fp = os.path.join(path, 'f1.pickle')
        with open(fp, 'rb') as f:        
            f1 += pickle.load(f)
    except FileNotFoundError:
        print('No F1 scores found. Skipping.')

    try:
        fp = os.path.join(path, 'syn_likelihood.pickle')
        with open(fp, 'rb') as f:        
            syn_likelihood += pickle.load(f)
    except FileNotFoundError:
        print('No syn_likelihood scores found. Skipping.')

    try:
        fp = os.path.join(path, 'test_likelihood.pickle')
        with open(fp, 'rb') as f:        
            test_likelihood += pickle.load(f)
    except FileNotFoundError:
        print('No test_likelihood scores found. Skipping.')

    shutil.move(path, consolidated_dir)

if len(accuracy) > 0:
    with open(os.path.join(consolidated_dir, 'accuracy.pickle'), 'wb') as f:
        pickle.dump(accuracy, f)
if len(f1) > 0:
    with open(os.path.join(consolidated_dir, 'f1.pickle'), 'wb') as f:
        pickle.dump(f1, f)
if len(syn_likelihood) > 0:
    with open(os.path.join(consolidated_dir, 'syn_likelihood.pickle'), 'wb') as f:
        pickle.dump(syn_likelihood, f)
if len(test_likelihood) > 0:
    with open(os.path.join(consolidated_dir, 'test_likelihood.pickle'), 'wb') as f:
        pickle.dump(test_likelihood, f)
