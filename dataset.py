import numpy as np
from torch.utils.data import Dataset
import torch


class PretrainDataset(Dataset):
    def __init__(
        self,
        data_path_lst,
        max_length=256,
        memmap=False,
        random_sampling=False,
        start_token=0,
        end_token=None,
        seed=42,
    ):
        super().__init__()
        self.max_length = max_length
        self.random_sampling = random_sampling
        self.rng = np.random.default_rng(seed)

        if memmap:
            with open(data_path_lst[0], 'rb') as f:
                f.seek(0, 2)
                flen = f.tell() // np.dtype('uint16').itemsize
            all_tokens = np.memmap(data_path_lst[0], dtype=np.dtype('uint16'), shape=(flen,))
        else:
            data_lst = []
            for data_path in data_path_lst:
                with open(data_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.uint16)
                    data_lst.append(data)
            all_tokens = np.concatenate(data_lst)

        total_tokens = len(all_tokens)
        if end_token is None or end_token > total_tokens:
            end_token = total_tokens
        if start_token < 0 or start_token >= end_token:
            raise ValueError("Invalid token range for dataset")

        self.data = all_tokens[start_token:end_token]
        self.num_tokens = len(self.data)
        if self.num_tokens <= self.max_length:
            raise ValueError("Not enough tokens for the requested max_length")

        # approximate epoch size by non-overlapping chunks
        self.n_samples = (self.num_tokens - 1) // self.max_length

        print(
            "memmap:{} random_sampling:{} token_range:[{}, {}) n_tokens:{} n_samples:{}".format(
                memmap,
                random_sampling,
                start_token,
                end_token,
                self.num_tokens,
                self.n_samples,
            )
        )
        print("downloading finished.....")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int):
        if self.random_sampling:
            start = int(self.rng.integers(0, self.num_tokens - self.max_length - 1))
        else:
            start = index * self.max_length
            max_start = self.num_tokens - self.max_length - 1
            if start > max_start:
                start = max_start

        sample = self.data[start:start + self.max_length]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y)


if __name__ == "__main__":
    pass
