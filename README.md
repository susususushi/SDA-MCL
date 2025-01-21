# Enhancing Feature Learning with Hard Samples in Mutual Learning for Online Class Incremental Learning
Official implementation of the paper Enhancing Feature Learning with Hard Samples in Mutual Learning for Online Class Incremental Learning.

## 1. Requirements

The experiments are conducted using the following hardware and software:

- Hardware: NVIDIA GeForce RTX 3090 GPUs
- Software: Please refer to `requirements.txt` for the detailed package versions. Conda is highly recommended.

## 2. Datasets

### CIFAR-100
The CIFAR-100 dataset will be automatically download during the first run and stored in `./dataset/cifar100`.

### TinyImageNet
The codebase should be able to handle TinyImageNet dataset automatically and save it in the `dataset` folder. If the automatic download fails, please download from http://cs231n.stanford.edu/tiny-imagenet-200.zip , and unzip it into `./dataset/tiny-imagenet-200`.

### ImageNet-100
Download the ImageNet dataset from [this link](http://www.image-net.org/) and follow [this](https://github.com/danielchyeh/ImageNet-100-Pytorch) for ImageNet-100 dataset generation. Put the dataset in the `./dataset/imagenet100_data` folder.

## 3. Training
### Training with a configuration file
Training can be done by specifying the dataset path and parameters in a configuration file. The detailed commands for different datasets are as follows:

```
cifar100：
python main.py --data-root-dir ./dataset/cifar100 --config ./config/2025/cifar100/SDAMCL,c100,m1000.yaml
python main.py --data-root-dir ./dataset/cifar100 --config ./config/2025/cifar100/SDAMCL,c100,m2000.yaml
python main.py --data-root-dir ./dataset/cifar100 --config ./config/2025/cifar100/SDAMCL,c100,m5000.yaml

tiny：
python main.py --data-root-dir ./dataset/tiny-imagenet-200 --config ./config/2025/tiny/SDAMCL,tiny,m2000.yaml
python main.py --data-root-dir ./dataset/tiny-imagenet-200 --config ./config/2025/tiny/SDAMCL,tiny,m5000.yaml
python main.py --data-root-dir ./dataset/tiny-imagenet-200 --config ./config/2025/tiny/SDAMCL,tiny,m10000.yaml

im100
python main.py --data-root-dir ./dataset/imagenet100_data --config ./config/2025/in100/SDAMCL,in100,m2000.yaml
python main.py --data-root-dir ./dataset/imagenet100_data --config ./config/2025/in100/SDAMCL,in100,m5000.yaml
python main.py --data-root-dir ./dataset/imagenet100_data --config ./config/2025/in100/SDAMCL,in100,m10000.yaml
```

## 4. Preliminary Experiment in Method Section

The preliminary experiment can be  conducted using the following command:

``````
python preliminary_first.py --data-root-dir ../dataset/cifar100 --config ./config/CVPR24/cifar100/PRE,c100,m1000.yaml --n-runs 10 --augmentation randaug1 --mem-iters 1
python preliminary_second.py --data-root-dir ../dataset/cifar100 --config ./config/CVPR24/cifar100/PRE,c100,m1000.yaml --n-runs 10 --augmentation randaug1 --mem-iters 5
``````

The scores will be saved in the  `./scores` directory.

## Acknowledgement

This implementation is based on the CCL-DC framework. Special thanks to [maorong-wang](https://github.com/maorong-wang) for his contribution to the framework and the implementation of recent state-of-the-art methods. 
