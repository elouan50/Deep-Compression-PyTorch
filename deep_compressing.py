import argparse
import subprocess

parser = argparse.ArgumentParser(description='Apply all deep compression steps to a model')
parser.add_argument('--stats', action='store_true', default=False,
                    help='print stats of use')
parser.add_argument('--model', type=str, default='LeNet100',
                    help='model to be used for training')
parser.add_argument('--dataset', type=str, default='MNIST',
                    help='dataset to be used for training')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
parser.add_argument('--output', default='saves/', type=str,
                    help='path to folder containing model outputs')
args = parser.parse_args()


# Prune
if args.no_cuda:
    subprocess.run(['python3', 'pruning.py',
                    f'--stats=False',
                    f'--batch-size={args.batch_size}',
                    f'--test-batch-size={args.test_batch_size}',
                    f'--epochs={args.epochs}',
                    f'--lr={args.lr}',
                    f'--no-cuda={args.no_cuda}',
                    f'--seed={args.seed}',
                    f'--log-interval={args.log_interval}',
                    f'--log={args.log}',
                    f'--sensitivity={args.sensitivity}',
                    f'--output={args.output}'
                    ])
else:
    
    subprocess.run(['python3', 'pruning.py',
                    f'--stats=False',
                    f'--batch-size={args.batch_size}',
                    f'--test-batch-size={args.test_batch_size}',
                    f'--epochs={args.epochs}',
                    f'--lr={args.lr}',
                    f'--seed={args.seed}',
                    f'--log-interval={args.log_interval}',
                    f'--log={args.log}',
                    f'--sensitivity={args.sensitivity}',
                    f'--output={args.output}'
                    ])

# Quantize
if args.no_cuda:
    subprocess.run(['python3', 'weight_share.py',
                    f'{args.output}model_after_retraining.ptmodel',
                    f'--stats=False',
                    f'--no-cuda={args.no_cuda}',
                    f'--output={args.output}model_after_weight_sharing.ptmodel'
                    ])
else:
    subprocess.run(['python3', 'weight_share.py',
                    f'{args.output}model_after_retraining.ptmodel',
                    f'--stats=False',
                    f'--output={args.output}model_after_weight_sharing.ptmodel'
                    ])

# Huffman encode
if args.no_cuda:
    subprocess.run(['python3', 'huffman_encode.py',
                    f'{args.output}model_after_weight_sharing.ptmodel',
                    f'--stats=False',
                    f'--no-cuda={args.no_cuda}'
                    ])
else:
    subprocess.run(['python3', 'huffman_encode.py',
                    f'{args.output}model_after_weight_sharing.ptmodel',
                    f'--stats=False'
                    ])

# Print stats
if args.no_cuda:
    if args.stats:
        subprocess.run(['python3', 'compare_stages.py',
                        f'--no-cuda={args.no_cuda}',
                        f'--path={args.output}'
                        ])
else:
    if args.stats:
        subprocess.run(['python3', 'compare_stages.py',
                        f'--path={args.output}'
                        ])
