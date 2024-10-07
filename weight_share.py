import argparse
import os

import torch

from net.models import LeNet
from net.quantization import apply_weight_sharing
import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program quantizes weight by using weight sharing')
    parser.add_argument('model', type=str,
                        help='saved quantized model')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='print stats of use')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--output', default='saves/model_after_weight_sharing.ptmodel', type=str,
                        help='path to model output')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()


    # Define the model
    model = torch.load(args.model)
    if args.stats:
        print('accuracy before weight sharing')
        util.test(model, use_cuda)
    else:
        print("Weight sharing running...")

    # Weight sharing
    apply_weight_sharing(model)
    if args.stats:
        print('accuracy after weight sharing')
        util.test(model, use_cuda)
    else:
        print("Done!")

    # Save the new model
    os.makedirs('saves', exist_ok=True)
    torch.save(model, args.output)
