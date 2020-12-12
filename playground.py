import torch
import pdb
from dataloader.dataloader import *
from dataloader.transforms import *

from params import *
from models.models import *
import torchvision
import torchvision.transforms.functional as TF
import pdb
import matplotlib.pyplot as plt


def transform(angle):
    T_MNIST_SINGLE_ROT = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Resize((32, 32)),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                  Rotation(angle),
                                                  torchvision.transforms.CenterCrop(32),
                                                  CircularMask(),
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
    model.load_state_dict(torch.load('./data/saved_models/checkpoint.pth.tar'))
    # model = DnQNet2(args, MODEL_CFGS_V3['F'], CLASSIFIER_CFGS['B'], mask=True)
    # model.load_state_dict(torch.load('./data/saved_models/dnq_cfg_fb_mnist.tar'))
    model = model.to(args.device)
    model = model.eval()
    # mask = model.feature_mask
    # plt.imshow(torch.squeeze(mask).cpu().detach().numpy(), cmap='gray')
    # plt.show()

    # for i, datum0 in enumerate(dataloader0, 0):
    #     for j, datum45 in enumerate(dataloader45, 0):
            # if i == 0 and j == 0:
            # if i == j:
    for i in range(len(dataloader0.dataset)):
        if True:
            if True:
                # input0, label0 = datum0
                # input45, label45 = datum45
                input0, label0 = dataloader0.dataset[i]
                input45, label45 = dataloader45.dataset[i]
                input0 = input0.to(args.device)
                input45 = input45.to(args.device)
                # show_img(input0[0, :, :])
                # show_img(input45[0, :, :])

                out0, equi0, inv0 = model(input0)
                out45, equi45, inv45 = model(input45)
                # pdb.set_trace()
                # show_img(equi0[0, 0, :, :])
                # show_img(equi45[0, 0, :, :])
                print (i, "th")
                pdb.set_trace()
                if False in out0 == out45:
                    print("not same detected")
                    pdb.set_trace()

