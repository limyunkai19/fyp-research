#!/bin/bash

dataset=datasets/IrisDataset

for model in resnet18 resnet50 resnet152 densenet121 densenet201 densenet161; do
    for i in 0 2 4
    do
        python -u experiment.py $model $i $dataset explainability_iris_${model}_${i} --batch-size 4 --epochs 40 --data-augmentation --save-best --save-state &> explainability_iris_${model}_${i}
        mv explainability_iris_${model}_${i} results/explainability_iris_${model}_${i}/output.txt
        mv results/explainability_iris_${model}_${i}/best_model.pth results/explainability_iris_${model}_${i}/model.pth
        python gen_gradcam.py --numpy explainability_iris_${model}_${i} datasets/IrisDataset/all
    done;
    exp=explainability_iris_${model}_0,explainability_iris_${model}_2,explainability_iris_${model}_4
    python plot_graph.py $exp "pretrained: 0,pretrained: 2,pretrained: 4" "$model on Iris flower dataset with data augmentation"
    mv ${model}_on_Iris_flower_dataset_with_data_augmentation.png results/graphs/explainability_iris_${model}.png
done;

