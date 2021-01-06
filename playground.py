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
import e2cnn

import numpy as np


def transform(angle):
    T_MNIST_SINGLE_ROT = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Resize((32, 32)),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                  Rotation(angle, ),
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
    dataloader = MNISTDataloader(args, 'test', T_MNIST)
    # dataloader = CIFAR10Dataloader(args, 'test', T_CIFAR10)

    model = DnQNet(args, MODEL_CFGS_V3['F'], CLASSIFIER_CFGS['B'])

    # model.load_state_dict(torch.load('./data/saved_models/dnq_cfg_fb_mnist.tar'))
    model.load_state_dict(torch.load('./data/saved_models/dnq_cfg_fb_cifar10.tar'))
    model = model.to(args.device)
    # mask = model.feature_mask
    # plt.imshow(torch.squeeze(mask).cpu().detach().numpy(), cmap='gray')
    # plt.show()
    e2cnn_model = C8SteerableCNN(n_classes=10, num_rot=16)
    # e2cnn_model.load_state_dict(torch.load('./data/saved_models/c16_mnist.tar'))
    e2cnn_model.load_state_dict(torch.load('./data/saved_models/c16_cifar10.tar'))
    e2cnn_model = e2cnn_model.to(args.device)
    e2cnn_model.eval()
    cnn_model = ResNet50(CLASSIFIER_CFGS_ResNet50['A'])
    # cnn_model.load_state_dict(torch.load('./data/saved_models/resnet50_cfg_a_mnist.tar'))
    cnn_model.load_state_dict(torch.load('./data/saved_models/resnet50_cfg_a_cifar10.tar'))
    cnn_model.to(args.device)
    cnn_model.eval()


    input, label = dataloader.dataset[0]
    input = input.unsqueeze(0).to(args.device)
    log_dnq=[]
    logcos_dnq=[]
    log_e2cnn=[]
    logcos_e2cnn = []
    log_cnn=[]
    logcos_cnn=[]
    for i in range(359):
        rot_input = TF.rotate(input, (i+1), expand=True)
        rot_input = TF.center_crop(rot_input, 32)
        rot_input = rot_input.to(args.device)
        # show_img(input0[0, :, :])
        # show_img(input45[0, :, :])

        out, equi, inv = model(input)
        rot_out, rot_equi, rot_inv = model(rot_input)

        _, _, e2inv = e2cnn_model(input)
        _, _, rot_e2inv = e2cnn_model(rot_input)

        _, _, cnninv = cnn_model(input)
        _, _, rot_cnninv = cnn_model(rot_input)

        # Calculate RDI (Rotational Deviation of Invariance)
        log_dnq.append((torch.norm(inv-rot_inv)/torch.norm(inv)).detach().cpu().numpy())
        log_e2cnn.append((torch.norm(e2inv-rot_e2inv)/torch.norm(e2inv)).detach().cpu().numpy())
        log_cnn.append((torch.norm(cnninv - rot_cnninv) / torch.norm(cnninv)).detach().cpu().numpy())

        # Calculate RDC (Rotational Deviation of Cosine-distance)
        # logcos_dnq.append((torch.dot(inv.squeeze(), rot_inv.squeeze())/(torch.norm(inv) * torch.norm(rot_inv))).detach().cpu().numpy())
        # logcos_e2cnn.append((torch.dot(e2inv.squeeze(), rot_e2inv.squeeze())/(torch.norm(e2inv) * torch.norm(rot_e2inv))).detach().cpu().numpy())
        # logcos_cnn.append((torch.dot(cnninv.squeeze(), rot_cnninv.squeeze())/(torch.norm(cnninv) * torch.norm(rot_cnninv))).detach().cpu().numpy())

        # pdb.set_trace()
        # # show_img(equi0[0, 0, :, :])
        # # show_img(equi45[0, 0, :, :])
        print (i, "th")
    log1 = np.expand_dims(np.array(log_dnq), axis=0)
    log2 = np.expand_dims(np.array(log_e2cnn), axis=0)
    log3 = np.expand_dims(np.array(log_cnn), axis=0)

    # log1 = np.expand_dims(np.array(logcos_dnq), axis=0)
    # log2 = np.expand_dims(np.array(logcos_e2cnn), axis=0)
    # log3 = np.expand_dims(np.array(logcos_cnn), axis=0)

    out = np.vstack([log1, log2, log3])

    np.savetxt("invstat_rdi_mnist.csv", out, delimiter=",")
