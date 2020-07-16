from ctgan import load_demo
from ctgan import CTGANSynthesizer

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

ctgan = CTGANSynthesizer()

ctgan.fit(data, discrete_columns, epochs=5)

samples = ctgan.sample(1000)
print(samples)
