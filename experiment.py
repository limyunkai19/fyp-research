import argparse

import torch, torchvision
import torch.nn as nn
import torch.optim as optim

import models, datasets
from utils import model_fit, model_save, get_trainable_param

# Command line argument
parser = argparse.ArgumentParser(description='Transfer Learning Experiment')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training and testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('model', help='name of model used for training')
parser.add_argument('pretrained', type=int, help='level knowledge of transferred from pretrained')
parser.add_argument('dataset', help='dataset for experiemnt')
parser.add_argument('--data-augmentation', action='store_true', default=False,
                    help='apply data augmentation on dataset (default: False)')
parser.add_argument('--sample-per-class', default="-1",  metavar='n',
                    help='use only n sample per class to train (default: use all)')
parser.add_argument('saves', help='the name to saves the experiment result after training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu-id', type=int, default=0,
                    help='use gpu of specific id for training, --no-cuda will override this (default: 0)')
parser.add_argument('--save-state', action='store_true', default=False,
                    help='save the state of model after training')
parser.add_argument('--save-best', action='store_true', default=False,
                    help='save the model with best validation accuracy')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed for replicable result (default: 1)')
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_id)

sample = [int(i) for i in args.sample_per_class.split(',')]*2

if args.dataset not in datasets.available_datasets:
    # custom folder dataset
    import os
    train_data = os.path.join(args.dataset, 'train')
    test_data = os.path.join(args.dataset, 'val')

    from torchvision.datasets.folder import find_classes
    num_classes = len(find_classes(train_data)[0])
    train_loader = datasets.folder(train_data, batch_size=args.batch_size,
                                data_augmentation=args.data_augmentation, sample_per_class=sample[0])
    test_loader = datasets.folder(test_data, batch_size=args.batch_size,
                                data_augmentation=args.data_augmentation, sample_per_class=sample[1])
else:
    num_classes = datasets.num_classes[args.dataset]
    dataset = datasets.__dict__[args.dataset]
    train_loader, test_loader = dataset(batch_size=args.batch_size, data_augmentation=args.data_augmentation,
                                            download=False, sample_per_class=tuple(sample))

cnn = models.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained)
if args.cuda:
    cnn.cuda()

optimizer = optim.Adam(get_trainable_param(cnn), lr=args.lr)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    criterion.cuda()

if args.save_best:
    save_best_name=args.saves
else:
    save_best_name=None
history = model_fit(cnn, train_loader, criterion, optimizer,
                        epochs=args.epochs, validation=test_loader, cuda=args.cuda, save_best_name=save_best_name)

model_save(cnn, history, args.saves, save_state=args.save_state)
