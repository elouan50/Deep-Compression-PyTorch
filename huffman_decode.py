import argparse
import os

import torch

from net.huffmancoding import huffman_decode_model
from net.models import LeNet
import util

parser = argparse.ArgumentParser(description='This program decodes a Huffman encoded model.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--directory', default='encodings/', type=str,
                    help='path to model encodings input')
parser.add_argument('--output', default='saves/model_after_decoding.ptmodel', type=str,
                    help='path to model output')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

# Define which model to use
model = LeNet(mask=True).to(device)
huffman_decode_model(model, args.directory)

util.print_nonzeros(model)

# Save the new model
os.makedirs('saves', exist_ok=True)
torch.save(model, args.output)