import torch
import pdb
from dataloader.dataloader import *

from params import *
from models.models import *
import torchvision
import torchvision.transforms.functional as TF
import pdb
import matplotlib.pyplot as plt

class Rotation:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

class expandChannel(object):
    """
    Expand grayscale image to 3 channel format
    """

    def __call__(self, x):
        x = x.repeat(3, 1, 1) #(b, c, h, w)
        return x

    def __repr__(self):
        return self.__class__.__name__+'()'

def transform(angle):
    T_MNIST_SINGLE_ROT = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Resize((32, 32)),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                  Rotation(angle),
                                                  expandChannel()])
    return T_MNIST_SINGLE_ROT

def show_img(img):
    img = torch.squeeze(img).cpu().detach().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    args = argument_parser()
    T_0 = transform(0)
    T_45 = transform(45)
    dataloader0 = MNISTDataloader(args, 'test', T_0)
    dataloader45 = MNISTDataloader(args, 'test', T_45)

    model = DnQNet(args, MODEL_CFGS_V3['F'], CLASSIFIER_CFGS['B'], mask=True)
    model.load_state_dict(torch.load('./data/saved_models/dnq_cfg_fb_mnist.tar'))
    model = model.to(args.device)
    # mask = model.feature_mask
    # plt.imshow(torch.squeeze(mask).cpu().detach().numpy(), cmap='gray')
    # plt.show()

    for i, datum0 in enumerate(dataloader0, 0):
        for j, datum45 in enumerate(dataloader45, 0):
            if i == 0 and j == 0:
                input0, label0 = datum0
                input45, label45 = datum45
                input0 = input0.to(args.device)
                input45 = input45.to(args.device)
                # show_img(input0[:, 0, :, :])

                _, equi0, inv0 = model(input0)
                _, equi45, inv45 = model(input45)
                # show_img(equi45[0, 1, :, :])
                pdb.set_trace()

