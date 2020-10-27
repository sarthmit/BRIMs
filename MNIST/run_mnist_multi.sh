#!/bin/bash
echo Running on $HOSTNAME
source /home/anirudh/.bashrc
conda activate torch1
run=1
lr=.0007
dim1=$1
dim2=$2
em=$1
block1=$3
block2=$4
topk1=$5
topk2=$6
drop=0.5
log=500
name="/home/anirudh/hierarchical_rims/blocks/sparse_relational/MNIST_v4/Blocks_2/Blocks_"$dim1"_"$dim2"_"$em"_"$block1"_"$block2"_"$topk1"_"$topk2"_FALSE_"$drop"_"$lr"_"$log
name="${name//./}"
echo Running version $name
python /home/anirudh/hierarchical_rims/blocks/sparse_relational/MNIST_v4/train_mnist.py --cuda --cudnn --algo blocks --name $name --lr $lr --drop $drop --nhid $dim1 $dim2 --num_blocks $block1 $block2 --topk $topk1 $topk2 --use_inactive --nlayers 2 --emsize $em --log-interval $log
