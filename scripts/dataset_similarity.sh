#!/bin/bash

dataset=(stl10 cifar10 emnist)
datasetSize=(80 80 17)
model=resnet18

for d in 0 1 2; do
    for i in -1 0 2 4
    do
        python -u experiment.py $model $i ${dataset[$d]} dataset_similarity_${dataset[$d]}_${model}_${i} --sample-per-class ${datasetSize[$d]} --batch-size 64 --epochs 40 --data-augmentation &> dataset_similarity_${dataset[$d]}_${model}_${i}
        mv dataset_similarity_${dataset[$d]}_${model}_${i} results/dataset_similarity_${dataset[$d]}_${model}_${i}/output.txt
    done;
    exp=dataset_similarity_${dataset[$d]}_${model}_-1,dataset_similarity_${dataset[$d]}_${model}_0,dataset_similarity_${dataset[$d]}_${model}_2,dataset_similarity_${dataset[$d]}_${model}_4
    python plot_graph.py $exp "sctrach,pretrained: 0,pretrained: 2,pretrained: 4" "$model on ${dataset[$d]} dataset with data augmentation"
    mv ${model}_on_${dataset[$d]}_dataset_with_data_augmentation.png results/graphs/dataset_similarity_${dataset[$d]}_${model}.png
done;

