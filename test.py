from sdgym import benchmark
from ctgan import CTGlowSynthesizer
from ctgan import argparser
import json
import os

args = argparser.parse_args()
ctgan = CTGlowSynthesizer(args)

scores = benchmark(synthesizers={args.name: ctgan.fit_sample}, datasets=args.datasets, iterations=1)
print(scores)
