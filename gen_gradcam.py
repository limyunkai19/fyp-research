from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np
import os, argparse

from pytorch_cnn_visualization import Visualize
import pytorch_cnn_visualization.utils as vis_utils

import models, utils

def gen_gradcam_image(model_parameter, data_dir, result_dir):
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

    result_dir = os.sep.join(["results", result_dir])
    if os.path.isdir(result_dir):
        if len(os.listdir(result_dir)) != 0:
            print("Result path exist and not empty, not generating")
            return
    else:
        os.mkdir(result_dir)

    from torchvision.datasets.folder import find_classes, make_dataset
    classes, class_to_idx = find_classes(data_dir)
    dataset = make_dataset(data_dir, class_to_idx)

    for class_name in os.listdir(data_dir):
        os.mkdir(os.sep.join([result_dir, class_name]))

    for img_path, idx in dataset:
        target_img_path = img_path.replace(data_dir+os.sep, "")
        print("Processing: ", target_img_path)

        img_pil = Image.open(img_path)
        img_pil = img_pil.resize((input_size,input_size))

        visualizer.input_image(img_pil)
        x = visualizer.get_prediction_output()
        x = F.softmax(Variable(x)).data

        for pred_c, idx in class_to_idx.items():
            gradcam = visualizer.get_gradcam_heatmap(idx)[0]
            target_img = target_img_path.replace('.jpg', '_{}_{:.4f}.jpg'.format(pred_c, x[0][idx]))
            gradcam.save(os.sep.join([result_dir, target_img]))

def gen_gradcam_numpy(model_parameter, data_dir):
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

    result = np.empty(0)

    from torchvision.datasets.folder import find_classes, make_dataset
    classes, class_to_idx = find_classes(data_dir)
    dataset = make_dataset(data_dir, class_to_idx)

    for img_path, idx in dataset:
        print("Processing: ", img_path.replace(data_dir+os.sep, ""))

        img_pil = Image.open(img_path)
        img_pil = img_pil.resize((input_size,input_size))

        visualizer.input_image(img_pil)

        gradcam = visualizer.get_gradcam_intensity(idx)
        gradcam = vis_utils.normalize(gradcam)
        result = np.append(result, gradcam)
        print(result.shape)

    np.save(os.sep.join(['results', model_parameter, 'gradcam.npy']), result)

# Command line argument
parser = argparse.ArgumentParser(description='Generate gradcam result')
parser.add_argument('model', help='directory of the trained model saves')
parser.add_argument('dataset', help='dataset for generating the gradcam')
parser.add_argument('--result-dir', help='directory to store the gradcam image result')
parser.add_argument('--image', action='store_true', default=False,
                                    help='generate gradcam image heatmap')
parser.add_argument('--numpy', action='store_true', default=False,
                                    help='generate gradcam numpy intensity')
args = parser.parse_args()

if not args.image and not args.numpy:
    print("Please specific --image or --numpy")
    exit()

# parameter
model_parameter = args.model
data_dir = args.dataset

if args.image:
    if not args.result_dir:
        print("Please specific --result-dir")
        exit()
    result_dir = args.result_dir
    gen_gradcam_image(model_parameter, data_dir, result_dir)
if args.numpy:
    gen_gradcam_numpy(model_parameter, data_dir)
