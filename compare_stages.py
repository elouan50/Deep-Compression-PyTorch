import argparse
import os

import torch
import util
from net.models import LeNet
from net.huffmancoding import huffman_encode_model, huffman_decode_model

parser = argparse.ArgumentParser(description='This program compares all stages of the deep compression procedure.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--path', default='saves/', type=str,
                    help='path to models output')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

print(' ')
print('-'*100)
print(f"{'*'*33:^100}")
print(f"{'** Deep compression statistics **':^100}")
print(f"{'*'*33:^100}")
print('-'*100)
print(' ')

model = torch.load(f"{args.path}/initial_model.ptmodel")
print('Initial model:', type(model).__name__)
util.print_model_parameters(model)

print(' ')
print('-'*100)
print(' ')
print(f'{"***After initial training***":^60}')
print(' ')
util.test(model, use_cuda)
print(' ')
print("Saving to binary, statistics:")
util.dump_raw(model, 'encodings/temp/')

print(' ')
print('-'*100)
print(' ')
print(f'{"***After pruning***":^60}')
print(' ')
model = torch.load(f"{args.path}/model_after_retraining.ptmodel")
util.print_nonzeros(model)


print(' ')
print('-'*100)
print(' ')
print(f'{"***After retraining***":^60}')
print(' ')
util.test(model, use_cuda)


print(' ')
print('-'*100)
print(' ')
print(f'{"***After quantizing***":^60}')
print(' ')
model = torch.load(f"{args.path}/model_after_weight_sharing.ptmodel")
util.print_nonzeros(model)
print(' ')
util.test(model, use_cuda)


print(' ')
print('-'*100)
print(' ')
print(f'{"***After Huffman encoding***":^60}')
print(' ')
huffman_encode_model(model)
print(' ')
util.print_nonzeros(model)
print(' ')
util.test(model, use_cuda)


print(' ')
print('-'*100)
print(' ')
print(f'{"***FYI: after Huffman decoding***":^60}')
print(' ')
model = LeNet(mask=True).to(device)
huffman_decode_model(model)
util.print_nonzeros(model)
print(' ')
util.test(model, use_cuda)
print(' ')
