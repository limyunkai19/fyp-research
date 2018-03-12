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
parser.add_argument('saves', help='the name to saves the experiment result after training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--save-state', action='store_true', default=False,
                    help='save the state of model after training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed for replicable result (default: 1)')
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

num_classes = datasets.num_classes[args.dataset]
cnn = models.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained)
dataset = datasets.__dict__[args.dataset]
train_loader, test_loader = dataset(batch_size=args.batch_size, download=False)

if args.cuda:
    cnn.cuda()

optimizer = optim.Adam(get_trainable_param(cnn), lr=args.lr)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    criterion.cuda()

history = model_fit(cnn, train_loader, criterion, optimizer,
                        epochs=args.epochs, validation=test_loader, cuda=args.cuda)

model_save(cnn, history, args.saves, save_state=args.save_state)
