import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import torchvision.models as models


class GraphNetBlock(nn.Module):
    '''
    Graph Convolutional Network Module
    '''
    def __init__(self, args, pointconv_cfg, residual=False):
        super(GraphNetBlock, self).__init__()
        self.vertexfeat = self.make_layer(args, pointconv_cfg)
        self.aggregate = self.weighted_aggregation(pointconv_cfg[-1], pointconv_cfg[-1], 3)
        # self.aggregate = self.unweighted_aggregation(args)
        self.residual = residual
        self.downsample=None
        if residual and pointconv_cfg[0] != pointconv_cfg[-1]:
            self.downsample = nn.Conv2d(pointconv_cfg[0], pointconv_cfg[-1], kernel_size=1, bias=False)



    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.vertexfeat(x)
        x = self.aggregate(x)
        if self.residual:
            x = x + identity
        return x

    def weighted_aggregation(self, in_channel, out_channel, kernel_size):

        # Laplacian Operator for Sparse Graph Convolution
        # w = torch.tensor([[-1/9,  -1/9, -1/9],
        #                   [-1/9,  8/9, -1/9],
        #                   [-1/9,  -1/9, -1/9]])
        alpha = torch.nn.Parameter(torch.tensor(1/9), requires_grad=False).to('cuda')
        beta = torch.nn.Parameter(torch.tensor(1/9), requires_grad=True).to('cuda')

        w = torch.tensor([[alpha.clone(),  alpha.clone(), alpha.clone()],
                          [alpha.clone(),  beta, alpha.clone()],
                          [alpha.clone(),  alpha.clone(), alpha.clone()]])
        w = w.view(1, 1, kernel_size, kernel_size).repeat(out_channel, 1, 1, 1)
        w = w.to('cuda')
        layer = torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1, bias=False, groups=out_channel)
        layer.weight = torch.nn.Parameter(w, requires_grad=True)
        return layer

    def unweighted_aggregation(self, args):
        if args.graph_agg_type == 'M':
            return torch.nn.MaxPool2d(kernel_size=args.edge_neighbor, stride=1,
                                            padding=int((args.edge_neighbor - 1) / 2))
        elif args.graph_agg_type == 'A':
            return torch.nn.AvgPool2d(kernel_size=args.edge_neighbor, stride=1,
                                            padding=int((args.edge_neighbor - 1) / 2))


        else:
            print("graph_agg_type set wrong!!")
            pdb.set_trace()


    def make_layer(self, args, pointconv_cfg):
        layer = []
        for i, v in enumerate(pointconv_cfg):
            if i == 0:
                in_channels = v
            else:
                try:
                    layer += [torch.nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=1, bias=True), torch.nn.BatchNorm2d(v), torch.nn.ReLU()]
                    # layer += [torch.nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=1, bias=True), torch.nn.ReLU()]
                except (TypeError):
                    print("debug! come to GraphNetBlock ")
                    pdb.set_trace()

                in_channels = v
        return torch.nn.Sequential(*layer)


class CosFaceClassifier(nn.Module):
    def __init__(self, cfgs, normalize=True):
        super(CosFaceClassifier, self).__init__()
        self.fc1 = nn.Linear(cfgs[0], cfgs[1])
        self.relu = nn.ReLU()
        self.normalize=normalize
        if normalize:
            self.fc2 = nn.utils.weight_norm(nn.Linear(cfgs[1], cfgs[2]))
        else:
            self.fc2 = nn.Linear(cfgs[1], cfgs[2])


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        out = self.fc2(x)
        return out


class DnQNet(nn.Module):
    '''
    Graph Neural Network - based v3
    '''

    def __init__(self, args, cfgs, cfgs_cls, residual=False, mask=True):

        super(DnQNet, self).__init__()
        self.encoder = self.make_layer(args, cfgs, residual)
        self.feature_mask = None
        self.avg_ratio = None
        if mask:
            self.feature_mask, self.avg_ratio = self.make_mask(16, args)
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = CosFaceClassifier(cfgs_cls)


    def forward(self, x):
        if self.feature_mask is not None:
            x = x * self.feature_mask
        equi_feat = self.encoder(x)
        inv_feat = self.GAP(equi_feat) * self.avg_ratio
        x = inv_feat.view(inv_feat.shape[0], -1)
        x = self.classifier(x)
        return x, equi_feat, inv_feat

    def make_layer(self, args, cfgs, residual):
        layer=[]
        for v in cfgs:
            if isinstance(v, list):
                layer += [GraphNetBlock(args, v, residual)]
            elif v == 'A':
                if args.point_agg_type == 'M':
                    layer += [torch.nn.MaxPool2d(2)]
                elif args.point_agg_type == 'A':
                    layer += [torch.nn.AvgPool2d(2)]
        return torch.nn.Sequential(*layer)

    def make_mask(self, R, args):
        s = int(2 * R)
        mask = torch.zeros(1, 1, s, s, dtype=torch.float32)
        c = (s-1) / 2
        for x in range (s):
            for y in range(s):
                r = (x - c) ** 2 + (y - c) ** 2
                if r > ((R-2) ** 2):
                    mask[..., x, y] = 0
                else:
                    mask[..., x, y] = 1
        active = torch.count_nonzero(mask)
        ratio = (s ** 2) / active
        return mask.to(args.device), ratio.to(args.device)

class VGG19(nn.Module):
    def __init__(self, cfg):
        super(VGG19, self).__init__()
        self.encoder = models.vgg19(pretrained=True).features
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(cfg[0], cfg[1],),
                                        # nn.ReLU(),
                                        nn.Linear(cfg[1], cfg[2]))

    def forward(self, x):
        equi_feat = self.encoder(x)
        inv_feat = self.avgpool(equi_feat)
        out = inv_feat.view(inv_feat.shape[0], -1)
        out = self.classifier(out)

        return out, equi_feat, inv_feat

class ResNet50(nn.Module):
    def __init__(self, cfg):
        super(ResNet50, self).__init__()
        modules = list(models.resnet50(pretrained=True).children())[:-2]
        self.encoder = nn.Sequential(*modules)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(cfg[0], cfg[1]),
                                        # nn.ReLU(),
                                        nn.Linear(cfg[1], cfg[2]))

    def forward(self, x):
        equi_feat = self.encoder(x)
        inv_feat = self.avgpool(equi_feat)
        out = inv_feat.view(inv_feat.shape[0], -1)
        out = self.classifier(out)
        return out, equi_feat, inv_feat


if __name__ == '__main__':
    model = ResNet50([2048, 256, 10])



