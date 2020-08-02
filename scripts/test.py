from sdgym import benchmark
from ctgan import CTGlowSynthesizer
from ctgan import argparser
import json
import os
from sdgym.synthesizers import IdentitySynthesizer
import pickle

synthetic = ['grid', 'gridr', 'ring', 'asia', 'alarm', 'child', 'insurance']
real = ['adult', 'census', 'covtype', 'credit', 'intrusion', 'mnist12', 'mnist28', 'news']

args = argparser.parse_args()
ctgan = CTGlowSynthesizer(args)

accuracy, f1, syn_likelihood, test_likelihood = [], [], [], []
for _ in range(args.iterations):
    scores = benchmark(synthesizers={args.name: ctgan.fit_sample}, datasets=args.datasets, iterations=1)
    print(scores)
    for dataset in args.datasets:
        if dataset in synthetic:
            syn_likelihood.append(scores['{}/syn_likelihood'.format(dataset)][args.name])
            test_likelihood.append(scores['{}/test_likelihood'.format(dataset)][args.name])
        elif dataset in real:
            accuracy.append(scores['{}/accuracy'.format(dataset)][args.name])
            f1.append(scores['{}/f1'.format(dataset)][args.name])
        else:
            raise AssertionError('Selected dataset not one of {} or {}'.format(synthetic, real))
if len(accuracy) > 0:
    with open(os.path.join(args.output_dir, 'accuracy.pickle'), 'wb') as f:
        pickle.dump(accuracy, f)
if len(f1) > 0:
    with open(os.path.join(args.output_dir, 'f1.pickle'), 'wb') as f:
        pickle.dump(f1, f)
if len(syn_likelihood) > 0:
    with open(os.path.join(args.output_dir, 'syn_likelihood.pickle'), 'wb') as f:
        pickle.dump(syn_likelihood, f)
if len(test_likelihood) > 0:
    with open(os.path.join(args.output_dir, 'test_likelihood.pickle'), 'wb') as f:
        pickle.dump(test_likelihood, f)
