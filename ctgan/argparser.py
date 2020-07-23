import argparse
import os

def str2bool(s):
    return s.lower().startswith('t')

def parse_args():

    parser = argparse.ArgumentParser(description='Glow on tabular data')

    parser.add_argument('--batch_size', default=512, type=int, help='Batch size per GPU')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--hidden_layers', '-C', default=[512], type=int, nargs='+', help='Space-separated list of hidden layer dimensions for scale-shift networks')
    parser.add_argument('--num_levels', '-L', default=1, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=10, type=int, help='Number of samples at test time')
    parser.add_argument('--l2scale', '--weight_decay', default=1e-6, type=float, help='L2 regularization on flow weights')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--datasets', default=['adult'], type=str, nargs='*', help='Space-separated list of test dataset names: GM: [grid, gridr, ring], BN: [asia, alarm, child, insurance], Real World: [adult, census, covtype, credit, intrusion, mnist12, mnist28, news]')
    parser.add_argument('--name', type=str, default='CTGlow', help='Experiment Name')

    parser.add_argument('--log_dir', type=str, default='logs', help='Logs directory name')

    args = parser.parse_args()

    args.output_dir = os.path.join(args.log_dir, args.name)
    
    return args
