import torch
from torch.utils.data import Dataset

import random
import numpy as np
import time


class AGDataset(Dataset):
    def __init__(self, first_fragment: int, last_fragment: int, path: str):
        for i in range(0, first_fragment):
            torch.ops.dataset_utils.unload_dataset_fragment(torch.empty(0), i)

        for i in range(first_fragment, last_fragment + 1):
            torch.ops.dataset_utils.load_dataset_fragment(torch.empty(0), i, path + 'buffer_' + str(i) + '.bin')

        self._sizes = torch.ops.dataset_utils.get_dataset_size(torch.empty(0)).detach().cpu().numpy()
        self.total_dll_time = 0.0
        self.shuffling_time = 0.0
        self.prepare_target = 0.0

        self._permutation = None
        self._counter = 0

        self._reset_ordering()

    @staticmethod
    def print_info() -> None:
        torch.ops.dataset_utils.print_dataset_info(torch.empty(0))

    def __len__(self) -> int:
        return len(self._sizes)

    def load_entire_batch(self, batch_size: int) -> dict:
        t0 = time.time()

        sample_ids = np.zeros((batch_size, 4), dtype=np.int32)

        for i in range(batch_size):
            idx = self._permutation[self._counter]
            sample_ids[i, 0] = self._sizes[idx, 0]
            sample_ids[i, 1] = self._sizes[idx, 1]
            sample_ids[i, 2] = random.randrange(0, self._sizes[idx, 2])
            sample_ids[i, 3] = random.randrange(0, self._sizes[idx, 3])

            self._counter += 1
            if self._counter >= len(self._permutation):
                self._reset_ordering()

        sample_ids = torch.IntTensor(sample_ids)
        self.shuffling_time += time.time() - t0

        t0 = time.time()
        input, policy_target, value_target, moves_left_target, action_values_target = torch.ops.dataset_utils.get_multiple_samples(
            sample_ids)
        self.total_dll_time += (time.time() - t0)

        return {'input': input.pin_memory(), 'policy_target': policy_target.pin_memory(), 'value_target': value_target.pin_memory()}

    def _reset_ordering(self) -> None:
        self._permutation = np.random.permutation(range(len(self._sizes)))
        self._counter = 0
