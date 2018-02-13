import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['DPN', 'dpn131', 'dpn98', 'dpn92']


def dpn92(input_shape, nb_classes, use_small_initial_convo=False):
    """
    Creates DPN-92 architecture according to https://arxiv.org/pdf/1707.01629.pdf.
    :param input_shape: Tuple in the form (input_channels, height, width).
    :param nb_classes: Number of output classes.
    :param use_small_initial_convo: Whether to use initial convo with kernel of size 3 or 7.
    :return: DPN-92 model.
    """
    return DPN(
        input_shape=input_shape,
        init_filters=64,
        nb_filters_in_blocks=[
            (96, 256), (192, 512), (384, 1024), (768, 2048)
        ],
        nb_repetitions=(3, 4, 20, 3),
        groups=32,
        width_factors=(16, 32, 24, 128),
        nb_classes=nb_classes,
        use_small_initial_convo=use_small_initial_convo
    )


def dpn98(input_shape, nb_classes, use_small_initial_convo=False):
    """
    Creates DPN-98 architecture according to https://arxiv.org/pdf/1707.01629.pdf.
    :param input_shape: Tuple in the form (input_channels, height, width).
    :param nb_classes: Number of output classes.
    :param use_small_initial_convo: Whether to use initial convo with kernel of size 3 or 7.
    :return: DPN-98 model.
    """
    return DPN(
        input_shape=input_shape,
        init_filters=96,
        nb_filters_in_blocks=[
            (160, 256), (320, 512), (640, 1024), (1280, 2048)
        ],
        nb_repetitions=(3, 6, 20, 3),
        groups=40,
        width_factors=(16, 32, 32, 128),
        nb_classes=nb_classes,
        use_small_initial_convo=use_small_initial_convo
    )


def dpn131(input_shape, nb_classes, use_small_initial_convo=False):
    """
    Creates DPN-131 architecture according to https://arxiv.org/pdf/1707.01629.pdf.
    :param input_shape: Tuple in the form (input_channels, height, width).
    :param nb_classes: Number of output classes.
    :param use_small_initial_convo: Whether to use initial convo with kernel of size 3 or 7.
    :return: DPN-131 model.
    """
    return DPN(
        input_shape=input_shape,
        init_filters=128,
        nb_filters_in_blocks=[
            (160, 256), (320, 512), (640, 1024), (1280, 2048)
        ],
        nb_repetitions=(4, 8, 28, 3),
        groups=40,
        width_factors=(16, 32, 32, 128),
        nb_classes=nb_classes,
        use_small_initial_convo=use_small_initial_convo
    )


def _get_padding(kernel_size):
    return (kernel_size - 1) // 2


def get_model_repr_and_params(model):
    output = [
        # str(model),
        'Weights: {}'.format(sum([np.product(p.size()) for p in model.parameters()]))
    ]
    return '\n'.join(output)


class _BnActConvo(nn.Module):
    def __init__(self, input_features, output_features, kernel_size, groups=1, stride=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_features, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(input_features, output_features, kernel_size, padding=_get_padding(kernel_size),
                      groups=groups, stride=stride, bias=False)
        )

    def forward(self, x):
        return self.layers(torch.cat(x, dim=1) if isinstance(x, tuple) else x)


class _BasicBlock(nn.Module):
    def __init__(self, input_features, output_features, interpolation_features, kernel_size, groups=1, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            _BnActConvo(input_features, output_features, 1),
            _BnActConvo(output_features, output_features, kernel_size, groups, stride),
            _BnActConvo(output_features, interpolation_features, 1),
        )

    def forward(self, x):
        return self.layers(torch.cat(x, dim=1) if isinstance(x, tuple) else x)


class _DpnBlock(nn.Module):
    def __init__(self,
                 input_features,
                 output_filters,
                 interpolation_filters,
                 k,
                 groups,
                 stride=1,
                 project=False):
        super().__init__()
        self.interpolation_filters = interpolation_filters
        self.down_sampling = None
        self.proj = None

        if stride > 1 or project:
            self.proj = nn.Sequential(
                _BnActConvo(input_features, interpolation_filters + 2 * k, 1, stride=stride)
            )
        self.conv = nn.Sequential(
            _BasicBlock(input_features, output_filters, interpolation_filters + k,
                        3, groups=groups, stride=stride)
        )
        self.project = project
        self.stride = stride

    def forward(self, x):
        inputs = x
        if self.project or self.stride > 1:
            inputs = self.proj(inputs)
            x = self.conv(x)
            res, den = inputs[:, :self.interpolation_filters], inputs[:, self.interpolation_filters:]
            residual = res + x[:, :self.interpolation_filters]
            dense = torch.cat([den, x[:, self.interpolation_filters:]], dim=1)
            return residual, dense
        else:
            res, den = inputs
            x = self.conv(inputs)

            residual = x[:, :self.interpolation_filters]
            dense = x[:, self.interpolation_filters:]
            return residual + res, torch.cat([den, dense], dim=1)


class DPN(nn.Module):
    def __init__(self,
                 input_shape,
                 init_filters,
                 nb_filters_in_blocks,
                 nb_repetitions,
                 groups,
                 width_factors,
                 nb_classes,
                 use_small_initial_convo,
                 ):
        """
        Creates Dual Path Network according to https://arxiv.org/pdf/1707.01629.pdf. Standard defined structures are
        DPN-92, DPN-98 and DPN-131 (this one is very heavy ~79.5 x 10^6 params).
        :param input_shape: Input shape of the input tensor.
        :param init_filters: Number of filter in the initial convo.
            Typically, it is:
                - 64 for DPN-92
                - 96 for DPN-98
                - 128 for DPN-131
        :param nb_filters_in_blocks: Number filters in each in block in each downsampled branch in form:
            [(filters of 3x3 convo, interpolated number of filters), ...].
            Number of filters in the 3x3 convo must be multiple of number of groups!
        :param nb_repetitions: Number of repetitions of dual path blocks in each downsampled branch.
            Typically, it is:
                - (3, 4, 20, 3) for DPN-92,
                - (3, 6, 20, 3) for DPN-98,
                - (4, 8, 28, 3) for DPN-131
        :param groups: Number of groups in the grouped convolutional layers.
        :param width_factors: Factors of dense branch incremental ('k' parameter) defined
            for each downsampled branch.
        :param nb_classes: Number of output classes.
        :param use_small_initial_convo: Whether to use initial convo with kernel of size 3 or 7.
        """
        super().__init__()
        assert len(nb_filters_in_blocks) == len(nb_repetitions) == len(width_factors)

        c, h, w = input_shape
        self.init_convo = nn.Sequential(
            nn.Conv2d(c, init_filters, 3 if use_small_initial_convo else 7,
                      padding=_get_padding(3 if use_small_initial_convo else 7), stride=2, bias=False),
            nn.BatchNorm2d(init_filters),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=_get_padding(3))
        )

        dpn_blocks = []
        last_filters = init_filters

        for nb_filters, reps, width_factor in zip(nb_filters_in_blocks, nb_repetitions, width_factors):
            inside_filters, output_filters = nb_filters
            for rep in range(reps):
                if last_filters == init_filters:
                    project = True
                    stride = 1
                elif rep == 0:
                    project = True
                    stride = 2
                else:
                    project = False
                    stride = 1
                dpn_blocks.append(_DpnBlock(
                    last_filters,
                    inside_filters,
                    output_filters,
                    width_factor,
                    groups,
                    stride=stride,
                    project=project
                ))
                if project or stride > 1:
                    last_filters = output_filters + 3 * width_factor
                else:
                    last_filters += width_factor
        self.dpn_block = nn.Sequential(*dpn_blocks)
        self.classifier = nn.Linear(last_filters, nb_classes)

        # Standard initialization implemented in pytorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, *x):
        x = x[0]
        x = self.init_convo(x)
        x = self.dpn_block(x)
        x = torch.cat(x, dim=1)
        if self.training:
            x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), stride=1)
        else:
            # according https://arxiv.org/pdf/1707.01629.pdf appendix
            x = 0.5 * F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), stride=1) \
                + 0.5 * F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), stride=1)
        x = x.view(-1, x.size(1))
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # Just for testing purposes
    from torch.autograd import Variable

    # model = dpn98((3, 224, 224), 1000)
    model = dpn98((3, 224, 224), 1000)
    print(get_model_repr_and_params(model))
    # print(model)
    model.forward(Variable(torch.from_numpy(np.random.random([1, 3, 224, 224]).astype(np.float32))))
