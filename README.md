# White Paper Assistamce: A Step Forward Beyond the Shortcut Learning

Pytorch implementation of White Paper Assistance in CIFAR.

## Main Requirements
- torch == 1.0.1
- torch == 0.2.0
- Python 3
- Pillow

## Training Examples
- CIFAR-100 ResNet-110(bottleneck) with $ P=1 $ and $ \lambda = 1 $.

`python train.py -d cifar100 --depth 110 --block_name bottleneck --trigger 1. --lambda_para 1. --gpu-id 0`

## Shortcut-CIFAR100 and CIFAR99
- White Paper Assistance

`python train_wp_shortcutcifar100.py --trigger 1. --lambda_para 1. --gpu-id 0`
  
- Spectral Decoupling

`python train_sd_shortcutcifar100.py`
  
- LfF

`python train_lff_shortcutcifar100.py`
