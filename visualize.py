from dataloader.dataloader import *
from params import argument_parser, MODEL_CFGS, MODEL_CFGS_V3, CLASSIFIER_CFGS, CLASSIFIER_CFGS_VGG19, CLASSIFIER_CFGS_ResNet50
from models.models import *
import torch
import torch.nn as nn
from dataloader.transforms import *
import numpy as np
import pdb

import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import seaborn as sns


VIZ_MODE = 'T-SNE'

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

if __name__ == '__main__':
    args = argument_parser()

    if VIZ_MODE == 'T-SNE':
        ## Call Model for visualization
        inv_feat_list = []
        label_list = []
        if args.tsne_model == 'DnQ':
            model = DnQNet(args, MODEL_CFGS_V3['F'], CLASSIFIER_CFGS['B'])
            model.load_state_dict(torch.load(args.tsne_state_dict_path))
            print("Loaded | ", args.tsne_state_dict_path, " | for visualization model")
            model.to(args.device)

        elif args.tsne_model == 'VGG19':
            model = VGG19(CLASSIFIER_CFGS_VGG19['A'])
            model.load_state_dict(torch.load(args.tsne_state_dict_path))
            print("Loaded | ", args.tsne_state_dict_path, " | for visualization model")
            model.to(args.device)


        ## Call dataset for visualization
        if args.tsne_dataset == 'MNIST':
            dataloader = MNISTDataloader(args, 'test', T_MNIST_ROT)

        elif args.tsne_dataset == 'CIFAR10':
            dataloader = CIFAR10Dataloader(args, 'test', T_CIFAR10_ROT)


        for i, datum in enumerate(dataloader, 0):
            inputs, labels = datum
            inputs = inputs.to(args.device)

            # Save invariant feature
            outputs, equi_feat, inv_feat = model(inputs)
            inv_feat = torch.squeeze(inv_feat).detach().cpu().numpy()
            inv_feat_list.append(inv_feat)

            # Save corresponding label
            label_list.append(torch.squeeze(labels).cpu().numpy())


            if i % 10 == 0:
                print('[ ', i, ' / ', len(dataloader), ']')

        y = np.concatenate(label_list, axis=0)
        X = np.concatenate(inv_feat_list, axis=0)

        sns.set_style('darkgrid')
        sns.set_palette('muted')
        sns.set_context("notebook", font_scale=1.5,
                        rc={"lines.linewidth": 2.5})

        feat_proj = TSNE(random_state=20150101).fit_transform(X)

        scatter(feat_proj, y)
        filename = 'viz_examples/' + args.tsne_model + '_' + args.tsne_dataset + '_tsne-generated.png'
        plt.savefig(filename, dpi=120)




