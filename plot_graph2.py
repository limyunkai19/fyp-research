import argparse
from utils import History

# Command line argument
parser = argparse.ArgumentParser(description='Plot History Graph')
parser.add_argument('name', metavar='experiment-name',
                    help='name of the experiment, comma seperate for multiple experiment')
parser.add_argument('label', help='label for the axis, comma seperate for multiple experiment')
parser.add_argument('title', help='title for the figure plotted')
parser.add_argument('--base-path', default='results', metavar="PATH",
                    help='the path to find the experiment saves file (default: results)')
parser.add_argument('--save-path', metavar="PATH",
                    help='the path to save the plotted figure (default: ./title.png)')
args = parser.parse_args()

import os, json
import seaborn as sns
import matplotlib.pyplot as plt


experiments = args.name.split(',')
labels = args.label.split(',')

# construct matplotlib fig
fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(18,7))

# plot axis
for i, (exp, label) in enumerate(zip(experiments, labels)):
    # open file
    f = open(os.path.join(args.base_path, exp, 'history.json'), 'r')
    hist = json.load(f)
    f.close()

    history = History().from_dict(hist).epoch_history

    # plot accuracy axis
    if i%2 == 0:
        ax_acc.plot(history['val_acc'], label=label, color=('C'+str(i//2)), linestyle=':')
    else:
        ax_acc.plot(history['val_acc'], label=label, color=('C'+str(i//2)), linestyle='-')

    # plot loss axis
    if i%2 == 0:
        ax_loss.plot(history['val_loss'], label=label, color=('C'+str(i//2)), linestyle=':')
    else:
        ax_loss.plot(history['val_loss'], label=label, color=('C'+str(i//2)), linestyle='-')

# add axis label
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('accuracy')
ax_acc.legend(loc='lower right')
ax_acc.set_title('Accuracy')

ax_loss.set_xlabel('epoch')
ax_loss.set_ylabel('loss')
ax_loss.legend(loc='upper right')
ax_loss.set_title('Loss')

sns.despine(fig)
fig.suptitle(args.title, fontsize=18)
fig.subplots_adjust(top=0.8)

if args.save_path is not None:
    fig.savefig(args.save_path, dpi='figure', format='png')
else:
    fig.savefig(args.title.replace(' ', '_')+'.png', dpi='figure', format='png')
