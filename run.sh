#!/bin/bash

dataset=cifar100
model=resnet50

for i in -1 0 1 2 3 4
do
    python -u experiment.py $model $i $dataset ${dataset}_${model}_${i} --batch-size 64 --epochs 0 &> ${dataset}_${model}_${i}
    mv ${dataset}_${model}_${i} results/${dataset}_${model}_${i}/output.txt
done;

exp=${dataset}_${model}_-1,${dataset}_${model}_0,${dataset}_${model}_1,${dataset}_${model}_2,${dataset}_${model}_3,${dataset}_${model}_4
python plot_graph.py  $exp "scratch,pretrained: 0,pretrained: 1,pretrained: 2,pretrained: 3,pretrained: 4" "$model on $dataset dataset"
