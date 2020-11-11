# BRIMs
_Bidirectional Recurrent Independent Mechanisms_

Implementation of the paper [_Learning to Combine Top-Down and Bottom-Up Signals in Recurrent Neural Networks with Attention over Modules_](https://arxiv.org/abs/2006.16981)

```
@article{mittal2020learning,
  title={Learning to combine top-down and bottom-up signals in recurrent neural networks with attention over modules},
  author={Mittal, Sarthak and Lamb, Alex and Goyal, Anirudh and Voleti, Vikram and Shanahan, Murray and Lajoie, Guillaume and Mozer, Michael and Bengio, Yoshua},
  journal={arXiv preprint arXiv:2006.16981},
  year={2020}
}
```

### MNIST Experiments

To run MNIST Experiments, please use the following command

`python train_mnist.py --emsize 300 --nlayers 2 --cuda --cudnn --algo blocks --num_blocks 6 3 --topk 4 2 --nhid 300 300 --use_inactive`

### CIFAR10 Experiments

To run CIFAR10 Experiments, please use the following command

`python train_cifar.py --emsize 300 --nlayers 2 --cuda --cudnn --algo blocks --num_blocks 6 6 --topk 4 4 --nhid 300 300 --use_inactive`