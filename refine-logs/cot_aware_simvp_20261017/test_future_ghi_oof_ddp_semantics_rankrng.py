"""CPU witnesses for the frozen two-GPU global-batch semantics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "ddp_train", HERE / "train_monthly_simvp_oof_ddp_20261020_rankrng.py")
ddp = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ddp)


def reference_batches(n: int, batch: int, seed: int):
    ds = TensorDataset(torch.arange(n, dtype=torch.int64))
    loader = DataLoader(ds, batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        num_workers=0, persistent_workers=False)
    return [x.tolist() for (x,) in loader]


def test_global_order_and_step_count():
    for n in (1, 2, 19, 20, 21, 39, 40, 41, 257, 27521):
        for seed in (42, 45, 420):
            expected = reference_batches(n, 20, seed)
            ranks = [list(ddp.RankBatchSampler(n, 20, seed, rank, 2))
                     for rank in range(2)]
            assert len(ranks[0]) == len(ranks[1]) == len(expected)
            merged = []
            for b0, b1, ref in zip(ranks[0], ranks[1], expected):
                # Singleton final batches are replicated so both ranks participate.
                merged_batch = b0 + b1
                if len(ref) == 1:
                    assert b0 == b1 == ref
                    merged_batch = b0
                else:
                    assert len(merged_batch) == len(ref)
                assert merged_batch == ref
                merged.extend(merged_batch)
            assert merged == [i for b in expected for i in b]
            assert len(ranks[0]) == (n + 19) // 20


def test_global_masked_loss_gradient_matches_single_batch():
    torch.manual_seed(7)
    x = torch.randn(20, 5)
    y = torch.randn(20, 3)
    valid = torch.rand(20, 3) > 0.23
    w0 = torch.randn(5, 3)
    b0 = torch.randn(3)

    w_single = w0.clone().requires_grad_()
    b_single = b0.clone().requires_grad_()
    pred = x @ w_single + b_single
    den = valid.sum().clamp_min(1)
    (((pred - y).square() * valid).sum() / den).backward()
    expected_w, expected_b = w_single.grad.clone(), b_single.grad.clone()

    global_den = valid.sum().to(torch.float32)
    grads_w, grads_b = [], []
    for rank in range(2):
        lo, hi = rank * 10, (rank + 1) * 10
        w = w0.clone().requires_grad_()
        b = b0.clone().requires_grad_()
        p = x[lo:hi] @ w + b
        local_num = ((p - y[lo:hi]).square() * valid[lo:hi]).sum()
        loss = local_num * 2 / global_den
        loss.backward()
        grads_w.append(w.grad)
        grads_b.append(b.grad)
    actual_w = (grads_w[0] + grads_w[1]) / 2
    actual_b = (grads_b[0] + grads_b[1]) / 2
    torch.testing.assert_close(actual_w, expected_w, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_b, expected_b, rtol=1e-6, atol=1e-6)


def test_rank_cuda_seed_contract():
    seeds = [ddp.rank_cuda_seed(45, rank) for rank in range(2)]
    assert seeds == [45, 1_000_048]
    g0 = torch.Generator().manual_seed(seeds[0])
    g1 = torch.Generator().manual_seed(seeds[1])
    m0 = torch.rand(4096, generator=g0) < 0.1
    m1 = torch.rand(4096, generator=g1) < 0.1
    assert not torch.equal(m0, m1)
    assert abs(float(m0.float().mean()) - 0.1) < 0.025
    assert abs(float(m1.float().mean()) - 0.1) < 0.025


if __name__ == "__main__":
    test_global_order_and_step_count()
    test_global_masked_loss_gradient_matches_single_batch()
    test_rank_cuda_seed_contract()
    print("FUTURE_GHI_DDP_GLOBAL_BATCH_SEMANTICS_PASS")
