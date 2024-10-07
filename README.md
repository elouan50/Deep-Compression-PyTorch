# Deep-Compression-PyTorch
PyTorch implementation of 'Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding'  by Song Han, Huizi Mao, William J. Dally

This implementation implements three core methods in the paper - Deep Compression
- Pruning
- Weight sharing
- Huffman Encoding

## Requirements
Following packages are required for this project
- Python `>=3.6`
- tqdm
- numpy
- pytorch, torchvision
- scipy
- scikit-learn

or just use docker
``` bash
$ docker pull tonyapplekim/deepcompressionpytorch
```
## Structure

This project contains files executing the main stages of the Deep Compression procedure described in the above paper.
1. Pruning
2. Weight sharing (Quantization)
3. Huffman encoding

For ease of use, were added:
- Comparison of performance after each stage with a single command (see below)
- Decoding of the Huffman coded files (seee below as well)

## Usage
Here you find usage recommandations for all three stages of the deep compression described in the paper. Please follow them in the given order.

### All in One
``` bash
$ python deep_compressing.py
```

This command
- Executes all three stages of deep compression
- Prints statistics at the end, of how much each stage compressed

### Pruning
``` bash
$ python pruning.py
```
This command
- trains LeNet-300-100 model with MNIST dataset
- prunes weight values that has low absolute value
- retrains the model with MNIST dataset
- prints out non-zero statistics for each weights in the layer

You can control other values such as
- random seed
- epochs
- sensitivity
- batch size
- learning rate
- and others

For more, type `python pruning.py --help`

### Weight sharing
``` bash
$ python weight_share.py saves/model_after_retraining.ptmodel
```
This command
* Applies K-means clustering algorithm for the data portion of CSC or CSR matrix representation for each weight
* Then, every non-zero weight is now clustered into (2**bits) groups.
(Default is 32 groups - using 5 bits)
- This modified model is saved to
`saves/model_after_weight_sharing.ptmodel`

For more, type `python weight_share.py --help`

### Huffman coding
``` bash
$ python huffman_encode.py saves/model_after_weight_sharing.ptmodel
```
This command
- Applies Huffman coding algorithm for each of the weights in the network
- Saves each weight to `encodings/` folder
- Prints statistics for improvement

For more, type `python huffman_encode.py --help`

### Comparison
``` bash
$ python compare_stages.py
```

This command
- Runs through all saved models (which thus need to be already compiled)
- Prints all relevant results for the different stages of the deep compression

For more, type `python compare_stages.py --help`

### Huffman Decoding
``` bash
$ python huffman_decode.py
```

This command
- Retrieves a `.ptmodel` file from the binary encodings contained in the `encodings/` directory

For more, type `python huffman_decode.py --help`

## Note
Note that no pruning nor weight sharing nor Huffman coding were applied for bias values. Maybe it’s better if those are applied to the biases as well, it wasn't tried out yet.

Note that this work was done when the original developer (github.com/mightydeveloper) was employed at http://nota.ai

