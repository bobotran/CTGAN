from ctgan import load_demo
from ctgan import CTGANSynthesizer
import argparse

def str2bool(s):
    return s.lower().startswith('t')

parser = argparse.ArgumentParser(description='Glow on tabular data')

parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--hidden_layers', '-C', default=[512], type=int, nargs='+', help='Number of channels in hidden layers')
parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
parser.add_argument('--num_samples', default=1000, type=int, help='Number of samples at test time')
parser.add_argument('--l2scale', '--weight_decay', default=1e-6, type=float, help='L2 regularization on flow weights')
parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

args = parser.parse_args()

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

ctgan.fit(data, discrete_columns, epochs=args.num_epochs)

samples = ctgan.sample(args.num_samples)
print(samples)
