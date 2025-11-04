import numpy as np
import torch
from torch.utils.data import Dataset
from tablut.utils.utils import getNNImage

class RandomSymDataset(Dataset):
    def __init__(self, examples, n, perms):
        self.n = n
        self.perms = perms
        self.items = []
        for b, pi, v, time, size in examples:
            B = b.astype(np.float32) if hasattr(b, "astype") else np.array(b, np.float32)
            self.items.append((
                np.ascontiguousarray(B, dtype=np.float32),   # 基底先存成连续
                np.asarray(pi, dtype=np.float32),
                float(v),
                time,
                size
            ))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        B, pi, v, time, size = self.items[i]
        s = np.random.randint(8)
        k, flip = divmod(s, 2)

        # 旋/翻棋盘（可能产生负stride视图）
        if B.ndim == 2:
            img = np.rot90(B, k)
            if flip: img = np.fliplr(img)
        else:
            img = np.rot90(B, k, axes=(-2, -1))
            if flip: img = np.flip(img, axis=-1)

        # 消除负stride保证dtype
        img = np.ascontiguousarray(img, dtype=np.float32)  # 或 img = img.copy().astype(np.float32, copy=False)

        # 置换表重排连续化
        pi_new = np.ascontiguousarray(pi[self.perms[s]], dtype=np.float32)

        img = getNNImage(img, size, time)

        return (
            torch.from_numpy(img),
            torch.from_numpy(pi_new),
            torch.tensor(v, dtype=torch.float32),
        )
