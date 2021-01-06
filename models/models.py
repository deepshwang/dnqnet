import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import torchvision.models as models
from e2cnn import gspaces
import e2cnn


class GraphNetBlock(nn.Module):
    '''
    Implementation of WN-NGCN
    '''
    def __init__(self, args, pointconv_cfg, residual=False):
        super(GraphNetBlock, self).__init__()
        self.vertexfeat = self.make_layer(args, pointconv_cfg)
        # self.aggregate = self.weighted_aggregation(pointconv_cfg[-1], pointconv_cfg[-1], 3)
        self.aggregate = self.weighted_aggregation(pointconv_cfg[0], pointconv_cfg[0], 3)
        # self.aggregate = self.unweighted_aggregation(args)
        self.residual = residual
        self.downsample=None
        if residual and pointconv_cfg[0] != pointconv_cfg[-1]:
            self.downsample = nn.Conv2d(pointconv_cfg[0], pointconv_cfg[-1], kernel_size=1, bias=False)



    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.aggregate(x)
        x = self.vertexfeat(x)
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

    def __init__(self, args, cfgs, cfgs_cls, residual=False, cosface=True):

        super(DnQNet, self).__init__()
        self.encoder = self.make_layer(args, cfgs, residual)
        # self.feature_mask, self.avg_ratio = self.make_mask(16, args)
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        if cosface:
            self.classifier = CosFaceClassifier(cfgs_cls)
        else:
            self.classifier = nn.Sequential(nn.Linear(cfgs_cls[0], cfgs_cls[1]),
                                            nn.Linear(cfgs_cls[1], cfgs_cls[2]))


    def forward(self, x):
        equi_feat = self.encoder(x)
        inv_feat = self.GAP(equi_feat)
        x = inv_feat.view(inv_feat.shape[0], -1)
        x = self.classifier(x)
        return x, equi_feat, inv_feat

    def make_layer(self, args, cfgs, residual):
        layer=[]
        for v in cfgs:
            if isinstance(v, list):
                layer += [GraphNetBlock(args, v, residual)]
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
        self.classifier = CosFaceClassifier(cfg)

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
        self.classifier = CosFaceClassifier(cfg)

    def forward(self, x):
        equi_feat = self.encoder(x)
        inv_feat = self.avgpool(equi_feat)
        out = inv_feat.view(inv_feat.shape[0], -1)
        out = self.classifier(out)
        return out, equi_feat, inv_feat

'''
E(2)-CNN
https://github.com/QUVA-Lab/e2cnn/blob/master/examples/model.ipynb
'''

class C8SteerableCNN(torch.nn.Module):

    def __init__(self, n_classes=10, num_rot=16, cosface=True):
        super(C8SteerableCNN, self).__init__()

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=num_rot)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = e2cnn.nn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 24 * [self.r2_act.regular_repr])
        self.block1 = e2cnn.nn.SequentialModule(
            e2cnn.nn.MaskModule(in_type, 32, margin=1),
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block2 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = e2cnn.nn.SequentialModule(
            e2cnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block3 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block4 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = e2cnn.nn.SequentialModule(
            e2cnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block5 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
        self.block6 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = e2cnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        self.gpool = e2cnn.nn.GroupPooling(out_type)

        # number of output channels
        c = self.gpool.out_type.size

        # Fully Connected
        if cosface:
            self.fully_net = CosFaceClassifier([c, 64, n_classes])
        else:
            self.fully_net = nn.Sequential(nn.Linear(c, 64),
                                            nn.Linear(64, n_classes))
        self.fully_net = CosFaceClassifier([c, 64, n_classes])

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = e2cnn.nn.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)

        # pool over the spatial dimensions
        equi = self.pool3(x)

        # pool over the group
        inv = self.gpool(equi)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        out = inv.tensor

        # classify with the final fully connected layers)
        out = self.fully_net(out.reshape(out.shape[0], -1))

        return out, equi.tensor, inv.tensor



if __name__ == '__main__':
    model = 0