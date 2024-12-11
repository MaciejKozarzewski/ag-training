import torch
import time
from typing import Optional

from click.core import batch

from Dataset import AGDataset

import numpy as np

torch.ops.load_library("build/libdataset_utils.so")


def make_channels_first(x: torch.Tensor) -> torch.Tensor:
    return torch.permute(x, (0, 3, 1, 2)).contiguous()


def make_channels_last(x: torch.Tensor) -> torch.Tensor:
    return torch.permute(x, (0, 2, 3, 1)).contiguous()


dataset = AGDataset(200, 249, '/home/maciek/alphagomoku/new_runs/btl_pv_8x128s/train_buffer_v200/')
dataset.print_info()

from conv_network import BottleneckPV
from transformer_network import TransformerPV
from loss import Loss

# model = BottleneckPV(32, 128, 8)
model = TransformerPV(32, 128, 5)
model = model.cuda(0)

optimizer = torch.optim.RAdam(model.parameters(), lr=1.0e-3)
loss_fn = Loss(1.0, 1.0, 0.0)


def lr_scheduler(epoch):
    if epoch < 10:
        return (1 + epoch) * 0.1
    elif epoch >= 75:
        return 0.1
    else:
        return 1.0


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)

print()


def flatten_to_2D(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        return x.reshape((x.shape[0], 1))
    elif len(x.shape) == 2:
        return x
    elif len(x.shape) == 3:
        return x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
    elif len(x.shape) == 4:
        return x.reshape((x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    else:
        raise Exception('incorrect number of dimensions')


def sum_running_loss(prev: Optional[dict], next: dict) -> dict:
    if prev is None:
        return next
    else:
        result = {}
        for k in prev.keys():
            result[k] = prev[k] + next[k]
        return result


def sum_running_accuracy(prev: Optional[torch.Tensor], next: torch.Tensor) -> torch.Tensor:
    if prev is None:
        return next
    else:
        return prev + next


def get_accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out_idx = torch.argmax(flatten_to_2D(output), dim=1)
    tar_idx = torch.argmax(flatten_to_2D(target), dim=1)
    return (out_idx == tar_idx).sum()


def train_one_epoch(batch_size: int, steps: int):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    running_loss = None
    policy_accuracy = None
    value_accuracy = None
    last_loss = [0.0, 0.0]

    compute_time = 0.0
    copy_time = 0.0
    stats_time = 0.0
    start = time.time()

    for i in range(steps):
        with torch.no_grad():
            training_data = dataset.load_entire_batch(batch_size)

        t0 = time.time()
        for k in training_data.keys():
            training_data[k] = training_data[k].cuda(0)
        copy_time += time.time() - t0

        t0 = time.time()
        optimizer.zero_grad()

        input_tensor = make_channels_first(training_data['input'])
        out_policy, out_value = model(input_tensor)
        out_policy = make_channels_last(out_policy)

        loss_dict = loss_fn(out_policy, training_data['policy_target'], out_value, training_data['value_target'])
        loss = loss_dict['policy_loss'] + loss_dict['value_loss']
        loss.backward()
        optimizer.step()

        compute_time += time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            running_loss = sum_running_loss(running_loss, loss_dict)
            acc = get_accuracy(out_policy, training_data['policy_target'])
            policy_accuracy = sum_running_accuracy(policy_accuracy, acc)

            acc = get_accuracy(out_value, training_data['value_target'])
            value_accuracy = sum_running_accuracy(value_accuracy, acc)

        stats_time += time.time() - t0

    last_loss[0] = running_loss['policy_loss'].item() / steps
    last_loss[1] = running_loss['value_loss'].item() / steps
    policy_acc = policy_accuracy.item() / (steps * batch_size)
    value_acc = value_accuracy.item() / (steps * batch_size)
    print('policy loss: {}, value loss {}'.format(last_loss[0], last_loss[1]))
    print('policy accuracy {}%, value accuracy {}%'.format(100 * policy_acc, 100 * value_acc))
    print('compute {}, total {}'.format(compute_time, time.time() - start))
    print('shuffling {}, DLL {}, preprocessing {}, copy {}, stats {}'.format(dataset.shuffling_time,
                                                                             dataset.total_dll_time,
                                                                             dataset.prepare_target, copy_time,
                                                                             stats_time))
    print()

    dataset.total_dll_time = 0.0
    dataset.prepare_target = 0.0
    dataset.shuffling_time = 0.0
    scheduler.step()


for i in range(100):
    print('epoch', i)
    train_one_epoch(128, 1000)
    if i % 10 == 0:
        torch.save(model, './model.pth')
