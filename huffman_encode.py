import argparse

import torch

from net.huffmancoding import huffman_encode_model
import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
    parser.add_argument('model', type=str,
                        help='saved quantized model')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='print stats of use')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')

    model = torch.load(args.model)

    if not(args.stats):
        print("Huffman encoding running...")
    huffman_encode_model(model, stats=args.stats)

    # Add --stats to see stats during execution
    if not(args.stats):
        print("Done!")