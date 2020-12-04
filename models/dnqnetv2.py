import torch
import torch.nn as nn
from params import argument_parser
import torchvision.models as models

class MNISTCNN(nn.Module):
    def __init__(self, cfg=None):
        super(MNISTCNN, self).__init__()
        if cfg is None:
            cfg = [8, 8, 16, 'M', 32, 32, 64]

        self.features = self._make_convlayers(cfg)

    def forward(self, x):
        # Feature extraction via convolutional layer
        x = self.features(x)
        return x

    def _make_convlayers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 1
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

class GeM(nn.Module):
    '''
	Geometric Mean Pooling
	'''

    def __init__(self, kernel_size=2, stride=2, p=3):
        super(GeM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.p = p
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.power(x, self.p)
        x = self.avgpool(x)
        x = self.power(x, 1 / self.p)
        return x

    def power(self, x, exp):
        x = torch.pow(x, exp)
        return x



def _load_backbone(args):
    if args.backbone == 'VGG16':
        model = models.vgg16(pretrained=True)
        layers = list(model.features.children())
        model = nn.Sequential(*layers)
    elif args.backbone == 'MNISTCNN':
        model = MNISTCNN()
    return model


class DnQClassifier(nn.Module):
    def __init__(self, args, cfg=None):
        super(DnQClassifier, self).__init__()
        if cfg is None:
            cfg = [32, 16, 'P', 16, 'P', 16, 10, 'P', 10]
        self.layer = self._make_layers(args, cfg, 64)

    def forward(self, x):
        x = self.layer(x)
        return x

    def _make_layers(self, args, cfg, in_channels):
        layers = []
        for v in cfg:
            if v == 'P':
                if args.pool_type == 'M':
                    layers += [nn.MaxPool2d(kernel_size=args.max_pool_size)]
                elif args.pool_type == 'GeM':
                    layers += [GeM(kernel_size=args.max_pool_size, stride=args.max_pool_size, p=args.p)]
                else:
                    raise NotImplementedError("Choose a pooling type : M or GeM")

            else:
                pointconv = nn.Conv2d(in_channels, v, kernel_size=1)
                layers += [pointconv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class DnQNetv2(nn.Module):
    def __init__(self, args):
        super(DnQNetv2, self).__init__()
        self.backbone = _load_backbone(args)
        self.classifier = DnQClassifier(args)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x