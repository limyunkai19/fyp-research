#!/bin/bash

model=resnet18

for size in 5 20 50; do
    for dataset in stl10 cifar10 mnist; do
        for i in -1 2
        do
            python -u experiment.py $model $i $dataset dataset_size_${dataset}_${size}_${model}_${i} --sample-per-class $size --batch-size 64 --epochs 40 --data-augmentation &> dataset_size_${dataset}_${size}_${model}_${i}
            mv dataset_size_${dataset}_${size}_${model}_${i} results/dataset_size_${dataset}_${size}_${model}_${i}/output.txt
        done;
    done;
    exp=dataset_size_stl10_${size}_${model}_-1,dataset_size_stl10_${size}_${model}_2,dataset_size_cifar10_${size}_${model}_-1,dataset_size_cifar10_${size}_${model}_2,dataset_size_mnist_${size}_${model}_-1,dataset_size_mnist_${size}_${model}_2
    python plot_graph2.py $exp "stl10 - sctrach,stl10 - pretrained 2,cifar10 - sctrach,cifar10 - pretrained 2,mnist - sctrach,mnist - pretrained 2" "$model on several dataset with $size sample per class"
    mv ${model}_on_several_dataset_with_${size}_sample_per_class.png results/graphs/dataset_size_${model}_on_several_dataset_with_${size}_sample_per_class.png
done;

