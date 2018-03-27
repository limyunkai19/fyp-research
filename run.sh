#!/bin/bash

dataset=cifar100
model=resnet50
sample=10

for i in -1 0 1 2 3 4
do
    python -u experiment.py $model $i $dataset ${dataset}_${sample}_${model}_${i}_dataAug --sample-per-class ${sample} --batch-size 32 --epochs 40 --data-augmentation &> ${dataset}_${sample}_${model}_${i}_dataAug
    mv ${dataset}_${sample}_${model}_${i}_dataAug results/${dataset}_${sample}_${model}_${i}_dataAug/output.txt
done;

exp=${dataset}_${sample}_${model}_-1_dataAug,${dataset}_${sample}_${model}_0_dataAug,${dataset}_${sample}_${model}_1_dataAug,${dataset}_${sample}_${model}_2_dataAug,${dataset}_${sample}_${model}_3_dataAug,${dataset}_${sample}_${model}_4_dataAug
python plot_graph.py $exp "scratch,pretrained: 0,pretrained: 1,pretrained: 2,pretrained: 3,pretrained: 4" "$model on $dataset (${sample}) dataset with data augmentation"
