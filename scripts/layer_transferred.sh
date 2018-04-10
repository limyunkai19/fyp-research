#!/bin/bash

dataset=stl10
#model=alexnet,vgg16_bn,densenet121,resnet50

for model in alexnet vgg16_bn densenet121 resnet50; do
    for i in -1 0 1 2 3 4
    do
        python -u experiment.py $model $i $dataset layer_transferred_${dataset}_${model}_${i} --batch-size 32 --epochs 20 --data-augmentation &> layer_transferred_${dataset}_${model}_${i}
        mv layer_transferred_${dataset}_${model}_${i} results/layer_transferred_${dataset}_${model}_${i}/output.txt
    done;
    exp=layer_transferred_${dataset}_${model}_-1,layer_transferred_${dataset}_${model}_1,layer_transferred_${dataset}_${model}_2,layer_transferred_${dataset}_${model}_3,layer_transferred_${dataset}_${model}_4
    python plot_graph.py $exp "scratch,pretrained: 0,pretrained: 1,pretrained: 2,pretrained: 3,pretrained: 4" "$model on $dataset dataset with data augmentation"
    mv ${model}_on_${dataset}_dataset_with_data_augmentation.png results/graphs/layer_transferred_${dataset}_${model}.png
done;

