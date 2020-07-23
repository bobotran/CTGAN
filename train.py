from ctgan import load_demo
from ctgan import CTGlowSynthesizer
from ctgan import argparser

args = argparser.parse_args()

data = load_demo()

discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

ctgan = CTGlowSynthesizer(args)

ctgan.fit(data, discrete_columns)

samples = ctgan.sample(args.num_samples)
print(samples)
