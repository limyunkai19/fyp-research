from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from matplotlib import pyplot as PLT

from pytorch_cnn_visualization import Visualize
import pytorch_cnn_visualization.utils as vis_utils

import models, utils

import os, argparse

# Visualizing some training result with saliency map and gradcam

# Command line argument
parser = argparse.ArgumentParser(description='Visualize prediction')
parser.add_argument('model', help='directory of the trained model saves')
parser.add_argument('image', help='image for generating the visualization')
args = parser.parse_args()

# parameter
model_parameter = args.model

cnn = utils.model_load(model_parameter)

input_size = vis_utils.get_input_size(cnn.meta['base_model'])
target_layer = vis_utils.get_conv_layer(cnn.meta['base_model'])
preprocess = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
])
visualizer = Visualize(cnn, preprocess, target_layer, num_classes=cnn.meta['num_classes'], retainModel=False)

img_pil = Image.open(args.image)
img_pil = img_pil.resize((input_size,input_size))

visualizer.input_image(img_pil)
x = visualizer.get_prediction_output()
x = F.softmax(Variable(x)).data
score = x.cpu().numpy()[0]
idx = score.argmax()

class_name = ["setosa", "versicolor", "virginica"]

print(idx, score[idx], class_name[idx])

img = [
    visualizer.get_gradcam_heatmap(idx)[0],
    visualizer.get_guided_backprop_gradient(idx)[0],
    visualizer.get_vanilla_backprop_gradient(idx)[0],
    visualizer.get_guided_gramcam_saliency(idx)[0]
]
title = ["Grad-CAM", "Guided Backpropagation", "Backpropagation", "Guided Grad-CAM"]
fig = PLT.figure(class_name[idx].split(",")[0])

for i in range(4):
    ax = fig.add_subplot(221+i)
    ax.axis('off')
    ax.imshow(img[i])
    ax.set_title(title[i])

PLT.suptitle(class_name[idx]+" Score: "+str(x[0][idx])[:5], fontsize=18)
PLT.show()
