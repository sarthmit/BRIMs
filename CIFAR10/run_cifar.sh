#!/bin/bash
echo Running on $HOSTNAME
source /home/anirudh/.bashrc
conda activate torch1
lr=.0007
dim1=$1
em=$1
block1=$2
topk1=$3
drop=0.5
log=500
name="/home/anirudh/hierarchical_rims/blocks/sparse_relational/CIFAR10_v4/CIFAR_blocks/Blocks_"$dim1"_"$em"_"$block1"_"$topk1"_FALSE_"$drop"_"$lr"_"$log
name="${name//./}"

#name="CIFAR/LSTM_"$dim1"_"$em"_"$drop"_"$lr"_"$log
#name="${name//./}"

echo Running version $name

python /home/anirudh/hierarchical_rims/blocks/sparse_relational/CIFAR10_v4/train_cifar.py --cuda --cudnn --algo blocks --name $name --lr $lr --drop $drop --nhid $dim1 --num_blocks $block1 --topk $topk1 --use_inactive --nlayers 1 --emsize $em --log-interval $log #--noise

#python train_cifar.py --cuda --cudnn --algo lstm --name $name --lr $lr --drop $drop --nhid $dim1 --emsize $em --log-interval $log
