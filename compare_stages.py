import argparse
import os

import torch
import util
from net.huffmancoding import huffman_encode_model

parser = argparse.ArgumentParser(description='This program compares all stages of the deep compression procedure.')
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

use_cuda = not args.no_cuda and torch.cuda.is_available()

print(' ')
print('-'*100)
print(' '*33+'*'*33+' '*34)
print(' '*33+'** Deep compression statistics **')
print(' '*33+'*'*33+' '*34)
print('-'*100)
print(' ')

model = torch.load("saves/initial_model.ptmodel")
print('Initial model:', type(model).__name__)
util.print_model_parameters(model)

print(' ')
print('-'*100)
print(' ')
print(' '*10+'***After initial training***')
print(' ')
util.test(model, use_cuda)


print(' ')
print('-'*100)
print(' ')
print(' '*10+'***After pruning***')
print(' ')
model = torch.load("saves/model_after_retraining.ptmodel")
util.print_nonzeros(model)


print(' ')
print('-'*100)
print(' ')
print(' '*10+'***After retraining***')
print(' ')
util.test(model, use_cuda)


print(' ')
print('-'*100)
print(' ')
print(' '*10+'***After quantizing***')
print(' ')
model = torch.load("saves/model_after_weight_sharing.ptmodel")
util.test(model, use_cuda)


print(' ')
print('-'*100)
print(' ')
print(' '*10+'***After Huffman encoding***')
print(' ')
huffman_encode_model(model, stats=True)
