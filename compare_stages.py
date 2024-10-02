import argparse
import os

import torch
import util
from net.huffmancoding import huffman_encode_model

parser = argparse.ArgumentParser(description='This program compares all stages of this deep compression preocedure.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()

model = torch.load("saves/initial_model.ptmodel")
print('Initial model:', type(model).__name__)
util.print_model_parameters(model)

print('-'*100)
print('***After initial training***')
util.test(model, use_cuda)


print('-'*100)
print('***After pruning***')
model = torch.load("saves/model_after_retraining.ptmodel")
util.print_nonzeros(model)


print('-'*100)
print('***After retraining***')
util.test(model, use_cuda)


print('-'*100)
print('***After quantizing***')
model = torch.load("saves/model_after_weight_sharing.ptmodel")
util.test(model, use_cuda)


print('-'*100)
print('***After Huffman encoding***')
huffman_encode_model(model, stats=True)
