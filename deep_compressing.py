import argparse
import subprocess

parser = argparse.ArgumentParser(description='Apply all deep compression steps to a model')
parser.add_argument('--stats', action='store_true', default=False,
                    help='print stats of use')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--output', default='saves/', type=str,
                    help='path prefix to model output')
args = parser.parse_args()

# print(f"Model: {args.model}")
print(args)


subprocess.run(['python3', 'pruning.py', f'--epochs={args.epochs}', f'--output={args.output}'])

subprocess.run(['python3', 'weight_share.py', f'{args.output}model_after_retraining.ptmodel', f'--output={args.output}model_after_weight_sharing.ptmodel'])

subprocess.run(['python3', 'huffman_encode.py', f'{args.output}model_after_weight_sharing.ptmodel'])

if args.stats:
    subprocess.run(['python3', 'compare_stages.py', f'--path={args.output}'])
