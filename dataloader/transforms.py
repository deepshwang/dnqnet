import torch
import torchvision
from params import argument_parser
import torchvision.transforms.functional as TF
import PIL
import pdb


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

class CircularMask(object):
    """
    Apply circular mask to the input
    """
    def __call__(self, x):
        mask = self.make_mask(16)

        x = x * mask
        return x

    def __repr__(self):
        return self.__class__.__name__+'()'

    def make_mask(self, R):
        s = int(2 * R)
        mask = torch.zeros(1, s, s, dtype=torch.float32)
        c = (s-1) / 2
        for x in range (s):
            for y in range(s):
                r = (x - c) ** 2 + (y - c) ** 2
                if r > ((R-2) ** 2):
                    mask[..., x, y] = 0
                else:
                    mask[..., x, y] = 1
        active = torch.count_nonzero(mask)
        return mask


class Rotation:
    """Rotate by one of the given angles."""

    def __init__(self, angle, resample):
        self.angle = angle
        self.resample = resample

    def __call__(self, x):
        return TF.rotate(x, self.angle, expand=True, resample=self.resample)

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

# INTER_METHOD = PIL.Image.NEAREST
INTER_METHOD = PIL.Image.BILINEAR


T_MNIST_GDCAL32 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Resize((32, 32))])

T_MNIST_GDCAL28 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Resize((28, 28))])

T_MNIST = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Resize((32, 32)),
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                          CircularMask(),
                                          expandChannel()])

T_MNIST_ROT = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Resize((32, 32), interpolation=INTER_METHOD),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                              # torchvision.transforms.RandomRotation(degrees=(30, 60), expand=True, resample=INTER_METHOD),
                                              Rotation(135, resample=INTER_METHOD),
                                              torchvision.transforms.CenterCrop(32),
                                              CircularMask(),
                                              expandChannel()])

T_MNIST_SINGLE_ROT = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Resize((32, 32)),
                                                     torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                     Rotation(args.single_rotation_angle, resample=INTER_METHOD),
                                                     CircularMask(),
                                                     expandChannel()])



T_CIFAR10 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    CircularMask(),
])

T_CIFAR10_ROT = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    Rotation(args.single_rotation_angle, resample=INTER_METHOD),
    CircularMask(),
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