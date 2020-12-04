import torch
import pdb

nb_channels = 12
h, w = 3, 3
x = torch.randn(1, nb_channels, h, w)
weights = torch.tensor([[0., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 0.]])
weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
pdb.set_trace()
