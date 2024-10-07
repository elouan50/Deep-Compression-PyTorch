import argparse
import os

import torch
import util
from net.huffmancoding import huffman_encode_model

parser = argparse.ArgumentParser(description='This program compares all stages of the deep compression procedure.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--path', default='saves/', type=str,
                    help='path to models output')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()

print(' ')
print('-'*100)
print(' '*33+'*'*33+' '*34)
print(' '*33+'** Deep compression statistics **')
print(' '*33+'*'*33+' '*34)
print('-'*100)
print(' ')

model = torch.load(f"{args.path}/initial_model.ptmodel")
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
model = torch.load(f"{args.path}/model_after_retraining.ptmodel")
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
model = torch.load(f"{args.path}/model_after_weight_sharing.ptmodel")
util.test(model, use_cuda)


print(' ')
print('-'*100)
print(' ')
print(' '*10+'***After Huffman encoding***')
print(' ')
huffman_encode_model(model, stats=True)
