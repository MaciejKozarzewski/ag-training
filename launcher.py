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


dataset = AGDataset(250, 250, '/home/maciek/alphagomoku/new_runs/btl_pv_8x128s/train_buffer/')
dataset.print_info()

from conv_network import BottleneckPV
from loss import Loss

model = BottleneckPV(32, 128, 8)
model = model.cuda(0)

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, fused=True)
loss_fn = Loss(1.0, 1.0, 0.0)

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


def train_one_epoch(batch_size: int):
    running_loss = None
    policy_accuracy = None
    value_accuracy = None
    samples_processed = 0
    last_loss = [0.0, 0.0]

    compute_time = 0.0
    copy_time = 0.0
    stats_time = 0.0
    start = time.time()
    for i in range(10000):
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
            samples_processed += batch_size

        stats_time += time.time() - t0
        if i % 1000 == 999:
            last_loss[0] = running_loss['policy_loss'].item() / 1000
            last_loss[1] = running_loss['value_loss'].item() / 1000
            policy_acc = policy_accuracy.item() / samples_processed
            value_acc = value_accuracy.item() / samples_processed
            print('batch {}, policy loss: {}, value loss {}'.format(i + 1, last_loss[0], last_loss[1]))
            print('batch {}, policy accuracy {}%, value accuracy {}%'.format(i + 1, 100 * policy_acc, 100 * value_acc))
            print('compute {}, total {}'.format(compute_time, time.time() - start))
            print('shuffling {}, DLL {}, preprocessing {}, copy {}, stats {}'.format(dataset.shuffling_time,
                                                                                     dataset.total_dll_time,
                                                                                     dataset.prepare_target, copy_time,
                                                                                     stats_time))
            print()

            dataset.total_dll_time = 0.0
            dataset.prepare_target = 0.0
            dataset.shuffling_time = 0.0

            running_loss = None
            policy_accuracy = None
            value_accuracy = None
            samples_processed = 0
            compute_time = 0.0
            copy_time = 0.0
            stats_time = 0.0
            start = time.time()

    return last_loss


train_one_epoch(128)
