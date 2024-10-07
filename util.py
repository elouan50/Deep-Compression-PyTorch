import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
from pathlib import Path
from scipy.sparse import csr_matrix, csc_matrix
import struct
import torch
import math
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')


def test(model, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=False, **kwargs)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


def dump_raw(model, directory='encodings/'):
    os.makedirs(directory, exist_ok=True)
    original_total = 0
    dumped_total = 0
    print(f"{'Layer':<15} | {'.ptmodel file':>13} {'dumped':>6} {'improvement':>10} {'percent':>8}")
    print('-'*70)
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        if 'weight' in name:
            weight = param.data.cpu().numpy()
            shape = weight.shape
            form = 'csr' if shape[0] < shape[1] else 'csc'
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

            # Encode
            dumped = encode(mat.data, name+f'_{form}_data', directory)

            # Print statistics
            original = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes

            print(f"{name:<15} | {original:13} {dumped:6} {original / dumped:>9.2f}x {100 * dumped / original:>7.2f}%")
        else: # bias
            bias = param.data.cpu().numpy()
            bias.dump(f'{directory}/{name}', True)

            # Print statistics
            original = bias.nbytes
            dumped = original

            print(f"{name:<15} | {original:13} {dumped:6} {original / dumped:>9.2f}x {100 * dumped / original:>7.2f}%")
        original_total += original
        dumped_total += dumped


def encode(arr, prefix, save_dir='./', stats=False):
    """
    Encodes numpy array 'arr' and saves to `save_dir`
    The names of binary files are prefixed with `prefix`
    returns the number of bytes for the tree and the data after the compression
    """
    # Infer dtype
    dtype = str(arr.dtype)

    # Path to save location
    directory = Path(save_dir)

    # Dump data
    data_encoding = ''.join(float2bitstr(value) for value in np.nditer(arr))
    datasize = dump(data_encoding, directory/f'{prefix}.bin', stats)

    return datasize

def float2bitstr(f):
    four_bytes = struct.pack('>f', f) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's


# My own dump / load logics
def dump(code_str, filename, stats=False):
    """
    code_str : string of either '0' and '1' characters
    this function dumps to a file
    returns how many bytes are written
    """
    # Make header (1 byte) and add padding to the end
    # Files need to be byte aligned.
    # Therefore we add 1 byte as a header which indicates how many bits are padded to the end
    # This introduces minimum of 8 bits, maximum of 15 bits overhead
    num_of_padding = -len(code_str) % 8
    header = f"{num_of_padding:08b}"
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = bytearray(int(code_str[i:i+8], 2) for i in range(0, len(code_str), 8))

    # Dump to a file
    if not(stats):
        with open(filename, 'wb') as f:
            f.write(byte_arr)
    return len(byte_arr)

