# Deep Metric Learning and Image Classification with Nearest Neighbour Gaussian Kernels
Pytorch implementation of "Deep metric learning and image classification with nearest neighbour gaussian kernels"  - [paper](https://arxiv.org/pdf/1705.09780.pdf "paper").

Citation:
```
@inproceedings{meyer2018deep,
  title={Deep metric learning and image classification with nearest neighbour gaussian kernels},
  author={Meyer, Benjamin J and Harwood, Ben and Drummond, Tom},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  pages={151--155},
  year={2018},
}
```

Note that this is not the implementation used to produce the results in the original paper. Differences include:
- Normalisation of the input with standard deviation (resulting in the use of a smaller Gaussian sigma, default 10).
- Shorter updated interval (5 epochs default).
- Default input shape (256x256) and default data augmentation (changing this may improve results).
- No approximate nearest neighbour search implmeneted (brute force approach used).
- Default network has no added dropout and contains final ReLU layer (changing this may improve results).

Tested with Python 3.5.2, Pytorch 1.1 and Ubuntu 16.04.6.

# Usage
## Datasets
Parent directory of dataset should contain class-specific sub-directories e.g. class_000/, class_001/ etc.
Leading zeros are important if you want the class labels to be sorted in the correct order.

## Training
Training script is set up to train a ResNet18 model (512-dimensional feature space). This can be changed by altering the train.py file.

Basic usage (replace g with GPU ID (or leave out for CPU) and n with the number of training classes):
```bash
python3 train.py --data_dir /path/to/dataset --save_dir /path/to/save/directory --gpu_id g  --num_classes n
```
Other training settings (e.g. sigma, maximum training time, batch size etc.) can be seen by running:
```bash
python3 train.py --help
```
or by looking in train.py.

Note the training script currently contains no testing code.
