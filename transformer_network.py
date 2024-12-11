import torch.nn
import torch.nn as nn
import math

from torch.nn import RMSNorm

from conv_network import InputBlock, PolicyHead, ValueHead


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.mean(x * x, dim=1, keepdim=True)
    return x / (tmp + 1.0e-6)


class MHA(torch.nn.Module):
    def __init__(self, num_heads: int, embedding: int, symmetric: bool = False):
        super(MHA, self).__init__()

        self._num_heads = num_heads
        self._symmetric = symmetric
        if symmetric:
            self._qkv = torch.nn.Linear(embedding, 2 * embedding, bias=False)
        else:
            self._qkv = torch.nn.Linear(embedding, 3 * embedding, bias=False)

        self._softmax = nn.Softmax(dim=-1)
        self._out = torch.nn.Linear(embedding, embedding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        head_dim = C // self._num_heads
        qkv = self._qkv(x)
        qkv = qkv.reshape(N, H * W, 3, self._num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))
        attn = self._softmax(attn / math.sqrt(head_dim))

        x = (attn @ v).transpose(1, 2).reshape(N, H, W, C)
        x = self._out(x).permute(0, 3, 1, 2)
        return x


class FFN(torch.nn.Module):
    def __init__(self, embedding: int):
        super(FFN, self).__init__()

        self._dense1 = torch.nn.Linear(embedding, embedding)
        self._dense2 = torch.nn.Linear(embedding, embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self._dense1(x)
        x = torch.nn.functional.gelu(x)
        x = self._dense2(x).permute(0, 3, 1, 2)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding: int):
        super(TransformerBlock, self).__init__()
        self._mha = MHA(embedding // 32, embedding)
        self._ffn = FFN(embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._mha(rmsnorm(x))
        x = x + self._ffn(rmsnorm(x))
        return x


class TransformerPV(torch.nn.Module):
    def __init__(self, input_channels: int, embedding: int, blocks: int):
        super(TransformerPV, self).__init__()
        self._input_block = InputBlock(input_channels, embedding)
        self._blocks = nn.ModuleList()
        for i in range(blocks):
            self._blocks.append(TransformerBlock(embedding))

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
