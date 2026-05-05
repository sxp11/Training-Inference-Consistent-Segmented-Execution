import torch
from typing import Optional


class MultiHeadExponentialBuffer:
    """
    GPU-only exponentially growing buffer for multi-head data.

    Stores a single contiguous tensor:
        buffer: [H, capacity, D]
    where all heads share the same logical size.

    - append: amortized O(1), expects x: [H, n, D]
    - data(): view only (no copy), returns [H, size, D]
    - reset(): logical reset or hard free
    """

    def __init__(
        self,
        num_heads: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        init_capacity: int = 8192,
    ):
        self.H = num_heads
        self.D = dim
        self.device = device
        self.dtype = dtype

        self.capacity = init_capacity
        self.init_capacity = init_capacity
        self.size = 0

        self.buffer = torch.empty(
            (self.H, self.capacity, self.D),
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def append(self, x: torch.Tensor):
        """
        x: [H, n, D] on GPU
        """
        assert x.device == self.device
        assert x.dim() == 3 and x.shape[0] == self.H and x.shape[2] == self.D
        n = x.shape[1]

        if self.size + n > self.capacity:
            new_capacity = self.capacity
            while new_capacity < self.size + n:
                new_capacity *= 2

            new_buffer = torch.empty(
                (self.H, new_capacity, self.D),
                device=self.device,
                dtype=self.dtype,
            )
            new_buffer[:, :self.size].copy_(self.buffer[:, :self.size])
            self.buffer = new_buffer
            self.capacity = new_capacity

        self.buffer[:, self.size:self.size + n].copy_(x)
        self.size += n

    def data(self) -> torch.Tensor:
        """
        Return view: [H, size, D]
        """
        return self.buffer[:, :self.size]

    def reset(self, max_capacity: int = 16384):
        if self.capacity > max_capacity:
            del self.buffer
            self.capacity = self.init_capacity
            self.size = 0
            self.buffer = torch.empty(
                (self.H, self.capacity, self.D),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.size = 0


class TorchKVMemory:
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        gpu_id: int = 0,
        init_capacity: int = 8192,
        max_capacity: int = 16384,
        max_return: int = 512,  # NEW: match search default / prefix cache
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.device = torch.device(f"cuda:{gpu_id}")
        self.dtype = torch.bfloat16

        self.init_capacity = init_capacity
        self.max_capacity = max_capacity

        # NEW: cache settings for search
        self.max_return = max_return  # search() will reuse cached prefix if equal

        # NEW: cache small tensors on GPU (avoid per-call allocations)
        self._offsets8 = torch.tensor(
            [-3, -2, -1, 0, 1, 2, 3, 4],
            device=self.device,
            dtype=torch.long
        )
        self._prefix_max_return = torch.arange(
            self.max_return,
            device=self.device,
            dtype=torch.long
        )

        # contiguous multi-head buffers
        self.key_buffer = MultiHeadExponentialBuffer(
            num_heads=self.num_heads,
            dim=self.dim,
            device=self.device,
            dtype=self.dtype,
            init_capacity=self.init_capacity,
        )
        self.value_buffer = MultiHeadExponentialBuffer(
            num_heads=self.num_heads,
            dim=self.dim,
            device=self.device,
            dtype=self.dtype,
            init_capacity=self.init_capacity,
        )

    @torch.no_grad()
    def add(self, key_vectors: torch.Tensor, value_vectors: torch.Tensor):
        """
        key/value: [1, H, Q, D] on GPU
        Appends as [H, Q, D]
        """
        assert key_vectors.dim() == 4 and value_vectors.dim() == 4
        assert key_vectors.shape == value_vectors.shape
        assert key_vectors.shape[0] == 1
        assert key_vectors.shape[1] == self.num_heads
        assert key_vectors.shape[3] == self.dim

        self.key_buffer.append(key_vectors.squeeze(0))    # [H, Q, D]
        self.value_buffer.append(value_vectors.squeeze(0))  # [H, Q, D]
    
    @staticmethod
    @torch.no_grad()
    def sliding_plus_tail_q(q: torch.Tensor, win: int = 8, stride: int = 4, tail: int = 4):
        """
        q: [1, H, L, D]
        return: [H, Nc, D]
        """
        B, H, L, D = q.shape
        assert B == 1

        chunks = []

        # sliding windows
        if L >= win:
            q2 = q.squeeze(0)                 # [H, L, D]
            x = q2.unfold(1, win, stride)     # [H, Nc_slide, D, win]
            slide = x.mean(dim=-1)             # [H, Nc_slide, D]  <-- mean over win
            chunks.append(slide)
        else:
            chunks.append(q.squeeze(0).mean(dim=1, keepdim=True))  # [H, 1, D]

        # tail window
        if L >= tail:
            tail_q = q[:, :, -tail:, :].mean(dim=2)     # [1, H, D]
            chunks.append(tail_q.squeeze(0).unsqueeze(1))  # [H, 1, D]

        out = torch.cat(chunks, dim=1)      # [H, Nc, D]
        return out           # [ H, Nc, D]


    @torch.no_grad()
    def search(self, query_tensor: torch.Tensor, top_k: int = 16, max_return: int = 512):
        """
        Assumptions:
          - M > 0
          - M >= max_return
        query_tensor: [1, H, Nc, D]
        return: k,v -> [1, H, max_return, D]
        """
        q = self.sliding_plus_tail_q(query_tensor)  # [H, Nc, D]

        memory_k_all = self.key_buffer.data()    # [H, M, D]
        memory_v_all = self.value_buffer.data()  # [H, M, D]
        M = memory_k_all.shape[1]

        scores_all = torch.bmm(q, memory_k_all.transpose(1, 2))  # [H, Nc, M]
        top = scores_all.topk(top_k, dim=-1)
        indices_all = top.indices  # [H, Nc, top_k]
        values_all  = top.values   # [H, Nc, top_k]

        # anchors upper bound so that window(8) ~ max_return
        A = max_return // 8

        # use cached tensors (no new allocations)
        offsets = self._offsets8
        if max_return == self.max_return:
            prefix = self._prefix_max_return
        else:
            prefix = torch.arange(max_return, device=self.device, dtype=torch.long)

        k_list, v_list = [None] * self.num_heads, [None] * self.num_heads

        for h in range(self.num_heads):
            base_idx = indices_all[h].reshape(-1)  # [N]
            base_sc  = values_all[h].reshape(-1)   # [N]

            # (1) dedup base_idx, keep max score per idx
            uniq_idx, inv = torch.unique(base_idx, return_inverse=True)  # uniq_idx: [U], inv: [N]
            uniq_sc = torch.full(
                (uniq_idx.numel(),),
                -float("inf"),
                device=self.device,
                dtype=base_sc.dtype
            )
            uniq_sc.scatter_reduce_(0, inv, base_sc, reduce="amax", include_self=True)

            # (2) select anchors by score
            if uniq_idx.numel() <= A:
                anchors = uniq_idx
            else:
                pick = uniq_sc.topk(A, largest=True).indices
                anchors = uniq_idx[pick]  # [A]

            # window expand
            win_idx = (anchors.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1).clamp_(0, M - 1)

            # dedup window
            win_idx, _ = torch.sort(win_idx)
            idx = torch.unique_consecutive(win_idx)

            # pad to max_return from prefix [0, max_return) avoiding duplicates
            if idx.numel() < max_return:
                need = max_return - idx.numel()
                extra = prefix[~torch.isin(prefix, idx)][:need]
                idx = torch.cat([idx, extra], dim=0)
                idx, _ = torch.sort(idx)

            # enforce dtype long for indexing
            idx = idx.to(torch.long)

            k_list[h] = memory_k_all[h, idx]  # [max_return, D]
            v_list[h] = memory_v_all[h, idx]

        k_tensor = torch.stack(k_list, dim=0).unsqueeze(0)  # [1, H, max_return, D]
        v_tensor = torch.stack(v_list, dim=0).unsqueeze(0)
        return k_tensor, v_tensor

    def reset(self):
        self.key_buffer.reset(self.max_capacity)
        self.value_buffer.reset(self.max_capacity)


