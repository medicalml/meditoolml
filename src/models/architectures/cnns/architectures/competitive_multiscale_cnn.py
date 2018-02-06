import torch
import torch.nn as nn


class MaxOut(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        cur_max = inputs[0]
        for inp in inputs[1:]:
            cur_max = torch.max(cur_max, inp)
        return cur_max


class ConvCompetitiveLayer(nn.Module):

    def __init__(self, in_channels, nb_kernels, kernels_sizes):
        super(ConvCompetitiveLayer, self).__init__()

        self.convs = [nn.Conv2d(in_channels, nb_kernels, k_sz, padding=k_sz//2)
                      for k_sz in kernels_sizes]

        self.bnorms = [nn.BatchNorm2d(nb_kernels) for _ in kernels_sizes]
        self.maxout = MaxOut()

    def forward(self, x):
        outs = [bnorm(conv(x)) for conv, bnorm in zip(self.convs, self.bnorms)]
        return self.maxout(outs)


class CompetitiveMultiscaleCNN(nn.Module):

    def __init__(self, in_channels):

        super(CompetitiveMultiscaleCNN, self).__init__()

        self.block0_layer_0 = ConvCompetitiveLayer(in_channels, 192, [1, 3, 5, 7])
        self.block0_layer_1 = ConvCompetitiveLayer(192, 160, [1, 3])
        self.block0_layer_2 = ConvCompetitiveLayer(160, 96, [1, 3])

        self.block0_maxpool = nn.MaxPool2d(3)
        self.block0_dropout = nn.Dropout2d()

        self.block1_layer_0 = ConvCompetitiveLayer(96, 192, [1, 3, 5, 7])
        self.block1_layer_1 = ConvCompetitiveLayer(192, 192, [1, 3])
        self.block1_layer_2 = ConvCompetitiveLayer(192, 192, [1, 3])

        self.block1_maxpool = nn.MaxPool2d(3)
        self.block1_dropout = nn.Dropout2d()

        self.block2_layer_0 = ConvCompetitiveLayer(192, 192, [1, 3, 5, 7])
        self.block2_layer_1 = ConvCompetitiveLayer(192, 192, [1, 3])
        self.block2_layer_2 = ConvCompetitiveLayer(192, 10, [1, 3])

        self.block2_avgpool = nn.AvgPool2d(3)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, return_logsoftmax=True):

        bl0_forw = self.block0_layer_2(self.block0_layer_1(self.block0_layer_0(x)))
        bl0_out = self.block0_dropout(self.block0_maxpool(bl0_forw))

        bl1_forw = self.block1_layer_2(self.block1_layer_1(self.block1_layer_0(bl0_out)))
        bl1_out = self.block1_dropout(self.block1_maxpool(bl1_forw))

        bl2_forw = self.block2_layer_2(self.block2_layer_1(self.block2_layer_0(bl1_out)))
        bl2_out = self.block2_avgpool(bl2_forw)
        if return_logsoftmax:
            bl2_out = self.logsoftmax(bl2_out)
        return bl2_out.view(bl2_out.size()[0], -1)

