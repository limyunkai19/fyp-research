#!/bin/bash

dataset=cifar100
model=resnet50

python -u experiment.py $model -1 $dataset ${dataset}_${model}_sc --batch-size 64 --epochs 100 &> ${dataset}_${model}_sc
mv ${dataset}_${model}_sc results/${dataset}_${model}_sc/output.txt

python -u experiment.py $model 0 $dataset ${dataset}_${model}_0 --batch-size 64 --epochs 100 &> ${dataset}_${model}_0
mv ${dataset}_${model}_0 results/${dataset}_${model}_0/output.txt

python -u experiment.py $model 1 $dataset ${dataset}_${model}_1 --batch-size 64 --epochs 100 &> ${dataset}_${model}_1
mv ${dataset}_${model}_1 results/${dataset}_${model}_1/output.txt

python -u experiment.py $model 2 $dataset ${dataset}_${model}_2 --batch-size 64 --epochs 100 &> ${dataset}_${model}_2
mv ${dataset}_${model}_2 results/${dataset}_${model}_2/output.txt

python -u experiment.py $model 3 $dataset ${dataset}_${model}_3 --batch-size 64 --epochs 100 &> ${dataset}_${model}_3
mv ${dataset}_${model}_3 results/${dataset}_${model}_3/output.txt

python -u experiment.py $model 4 $dataset ${dataset}_${model}_4 --batch-size 64 --epochs 100 &> ${dataset}_${model}_4
mv ${dataset}_${model}_4 results/${dataset}_${model}_4/output.txt

python plot_graph.py ${dataset}_${model}_sc,${dataset}_${model}_0,${dataset}_${model}_1,${dataset}_${model}_2,${dataset}_${model}_3,${dataset}_${model}_4 "scratch,pretrained: 0,pretrained: 1,pretrained: 2,pretrained: 3,pretrained: 4" "resnet50 on cifar100 dataset"
