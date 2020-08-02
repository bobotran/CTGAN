import argparse
import os
import json

def str2bool(s):
    return s.lower().startswith('t')

def parse_args():

    parser = argparse.ArgumentParser(description='Glow on tabular data')

    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size per GPU')
    parser.add_argument('--lr', default=5e-5, type=float, help='Learning rate')
    parser.add_argument('--hidden_layers', '-C', default=[512], type=int, nargs='+', help='Space-separated list of hidden layer dimensions for scale-shift networks')
    parser.add_argument('--num_levels', '-L', default=1, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train')
    parser.add_argument('--l2scale', '--weight_decay', default=1e-6, type=float, help='L2 regularization on flow weights')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--datasets', default=['adult', 'asia'], type=str, nargs='*', help='Space-separated list of test dataset names: GM: [grid, gridr, ring], BN: [asia, alarm, child, insurance], Real World: [adult, census, covtype, credit, intrusion, mnist12, mnist28, news]')
    parser.add_argument('--name', type=str, default='CTGlow', help='Experiment Name')

    parser.add_argument('--log_dir', type=str, default='logs', help='Logs directory name')

    parser.add_argument('--iterations', type=int, default=1, help='Number of evaluation runs over which results are averaged')

    parser.add_argument('--smote', type=str, default='', help='[augment | latent | alpha_beta]')

    parser.add_argument('--lr_decay', default=0.98, type=float, help='Multiplicative factor applied to learning rate after each epoch')

    parser.add_argument('--output_latent', default=False, action='store_true', help='Dump data and latent representations')

    parser.add_argument('--config_path', type=str, default=None, help='Loads arguments from specified json. Any CLI arguments provided override the json values')
    args = parser.parse_args()

    if args.config_path:
        cli_args = args
        saved_args = argparse.Namespace()
        with open(cli_args.config_path) as f:
            saved_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=saved_args)

    args.output_dir = os.path.join(args.log_dir, args.name)
    
    return args
