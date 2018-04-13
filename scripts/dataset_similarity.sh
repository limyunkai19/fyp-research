#!/bin/bash

# dataset=stl10 cifar10 emnist
datasetSize=(80 80 17)
model=resnet18

for dataset in stl10 cifar10 emnist; do
    for i in 0 2 4
    do
        python -u experiment.py $model $i $dataset dataset_similarity_${dataset}_${model}_${i} --sample-per-class ${datasetSize[$i]} --batch-size 64 --epochs 40 --data-augmentation &> dataset_similarity_${dataset}_${model}_${i}
        mv dataset_similarity_${dataset}_${model}_${i} results/dataset_similarity_${dataset}_${model}_${i}/output.txt
    done;
    exp=dataset_similarity_${dataset}_${model}_-1,dataset_similarity_${dataset}_${model}_1,dataset_similarity_${dataset}_${model}_2,dataset_similarity_${dataset}_${model}_3,dataset_similarity_${dataset}_${model}_4
    python plot_graph.py $exp "scratch,pretrained: 0,pretrained: 1,pretrained: 2,pretrained: 3,pretrained: 4" "$model on $dataset dataset with data augmentation"
    mv ${model}_on_${dataset}_dataset_with_data_augmentation.png results/graphs/dataset_similarity_${dataset}_${model}.png
done;

