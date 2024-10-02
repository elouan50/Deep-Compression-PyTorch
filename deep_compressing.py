import argparse

import pruning
import weight_share
import huffman_encode
import compare_stages

parser = argparse.ArgumentParser(description='Apply all deep compression steps to a model')
parser.add_argument('model', type=str,
                    help='saved quantized model')
parser.add_argument('--stats', action='store_true', default=False,
                    help='print stats of use')
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
parser.add_argument('--output', default='saves/model_after_weight_sharing.ptmodel', type=str,
                    help='path to model output')
args = parser.parse_args()
print(f"Model: {args.model}")

pruning()
args.model = "saves/model_after_retraining.ptmodel"
weight_share()
args.model = "saves/model_after_weight_sharing"
huffman_encode()
compare_stages()