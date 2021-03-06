import torch
import torchvision
from params import argument_parser
import torchvision.transforms.functional as TF


class posConcat(object):
    """
    Concatenate positional vectors to channels
    """
    def __call__(self, x):
        c, h, w = x.shape
        x_coor = (torch.mm(torch.ones(h, 1), torch.range(0, w - 1).expand(1, w)).unsqueeze(0)-int(w/2))/(w/2)
        y_coor = (torch.mm(torch.transpose(torch.range(0, h - 1).expand(1, h), 0, 1), torch.ones(1, w)).unsqueeze(0)-int(h/2))/(h/2)

        # Concatenate with input image
        point = torch.cat((x_coor, y_coor, x), 0)
        return point

    def __repr__(self):
        return self.__class__.__name__+'()'

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

args = argument_parser()

T_MNIST = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Resize((32, 32)),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                          expandChannel()])

T_MNIST_ROT = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Resize((32, 32)),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                              Rotation(args.single_rotation_angle),
                                              # torchvision.transforms.RandomRotation(degrees=(300, 330)),
                                              expandChannel()])



T_CIFAR10 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

T_CIFAR10_ROT = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    Rotation(args.single_rotation_angle),
    # torchvision.transforms.RandomRotation(degrees=(300, 330))
])

T_CIFAR100 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

T_CIFAR100_ROT = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    torchvision.transforms.RandomRotation(degrees=(45, 315))
])