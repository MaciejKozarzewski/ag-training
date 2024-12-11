import torch.nn
import torch.nn as nn


def add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = torch.permute(x, (0, 2, 3, 1))  # to NHWC
    x = x + bias
    return torch.permute(x, (0, 3, 1, 2))  # back to NCHW


class ConvBatchNormAct(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, act: str = None, name: str = ''):
        super(ConvBatchNormAct, self).__init__()
        pad = (kernel_size - 1) // 2
        self._conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=(pad, pad), bias=False)
        self._batchnorm = torch.nn.BatchNorm2d(out_channels, affine=False)
        self._bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self._act = act
        nn.init.uniform_(self._bias, 0.0, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        x = self._batchnorm(x)
        x = add_bias(x, self._bias)
        if self._act == 'relu':
            x = torch.relu(x)
        return x


class DenseBatchNormAct(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = None):
        super(DenseBatchNormAct, self).__init__()

        self._dense = torch.nn.Linear(in_channels, out_channels, bias=False)
        self._batchnorm = torch.nn.BatchNorm1d(out_channels, affine=False)
        self._bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self._act = act
        nn.init.uniform_(self._bias, 0.0, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._dense(x)
        x = self._batchnorm(x)
        x = x + self._bias
        if self._act == 'relu':
            x = torch.relu(x)
        return x


class InputBlock(torch.nn.Module):
    def __init__(self, in_channels: int, embedding: int):
        super(InputBlock, self).__init__()
        self._conv_5x5 = ConvBatchNormAct(in_channels, embedding, 5, 'relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_5x5(x)


class BottleneckBlock(torch.nn.Module):
    def __init__(self, embedding: int):
        super(BottleneckBlock, self).__init__()
        self._conv_1 = ConvBatchNormAct(embedding, embedding // 2, 1, 'relu')
        self._conv_2 = ConvBatchNormAct(embedding // 2, embedding // 2, 3, 'relu')
        self._conv_3 = ConvBatchNormAct(embedding // 2, embedding, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._conv_1(x)
        y = self._conv_2(y)
        y = self._conv_3(y)
        return torch.nn.functional.relu(x + y)


class PolicyHead(torch.nn.Module):
    def __init__(self, embedding: int):
        super(PolicyHead, self).__init__()
        self._conv_3x3 = ConvBatchNormAct(embedding, embedding, 3, 'relu')
        self._conv_1x1 = torch.nn.Conv2d(embedding, 1, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_3x3(x)
        return self._conv_1x1(x)


class ValueHead(torch.nn.Module):
    def __init__(self, embedding: int):
        super(ValueHead, self).__init__()
        self._conv_1x1 = ConvBatchNormAct(embedding, embedding, 1, 'relu')
        self._dense_1 = DenseBatchNormAct(2 * embedding, 256, 'relu')
        self._dense_2 = DenseBatchNormAct(256, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_1x1(x)

        board_size = (x.shape[2], x.shape[3])
        avg = torch.nn.functional.avg_pool2d(x, board_size).squeeze((2, 3))
        max = torch.nn.functional.max_pool2d(x, board_size).squeeze((2, 3))

        y = torch.cat([avg, max], dim=-1)
        y = self._dense_1(y)
        y = self._dense_2(y)
        return y


class BottleneckPV(torch.nn.Module):
    def __init__(self, input_channels: int, embedding: int, blocks: int):
        super(BottleneckPV, self).__init__()
        self._input_block = InputBlock(input_channels, embedding)
        self._blocks = nn.ModuleList()
        for i in range(blocks):
            self._blocks.append(BottleneckBlock(embedding))

        self._policy_head = PolicyHead(embedding)
        self._value_head = ValueHead(embedding)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    nn.init.uniform_(m.bias.data, 0.0, 0.1)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self._input_block(x)
        for block in self._blocks:
            x = block(x)
        p = self._policy_head(x)
        v = self._value_head(x)
        return p, v
