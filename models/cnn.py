import torch.nn as nn
import torchvision.models as models
import torch
import pdb


class CNN(nn.Module):
    def __init__(self, cfg=[64, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512]):
        super(CNN, self).__init__()
        self.features = self.make_convlayers(cfg)
        self.classifiers = nn.Linear(3072, 10)

    def forward(self, x):
        # Feature extraction via convolutional layer
        _ = 0
        feat = self.features(x)

        return x, feat, _

    def make_convlayers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        if cfg is not None:
            for l in cfg:
                if l == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, l, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(l), nn.ReLU()]
                    else:
                        layers += [conv2d, nn.ReLU()]
                    in_channels = l

        return nn.Sequential(*layers)


class VGG19(nn.Module):
    def __init__(self, cfg):
        super(VGG19, self).__init__()
        self.encoder = models.vgg19(pretrained=True).features
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.adapt_avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(cfg[0], cfg[1]),
                                        nn.Linear(cfg[1], cfg[2]))


if __name__=="__main__":
    model = VGG19()
    pdb.set_trace()