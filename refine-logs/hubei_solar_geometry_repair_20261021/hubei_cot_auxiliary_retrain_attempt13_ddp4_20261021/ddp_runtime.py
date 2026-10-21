"""Global-batch-preserving DDP reductions for attempt13 checkpoint migration."""
import hashlib
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous()
            h.update(str((a.dtype, tuple(a.shape))).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(repr(k).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x: visit(v)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


class GlobalBatchShard(Sampler):
    """Shard each global batch of 16 without padding, dropping, or extra steps."""
    def __init__(self, n, rank, world, shuffle, seed=42):
        self.n, self.rank, self.world = n, rank, world
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(self.n / 16)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        ids = torch.randperm(self.n, generator=g).tolist() if self.shuffle else list(range(self.n))
        for start in range(0, self.n, 16):
            part = ids[start:start + 16][self.rank::self.world]
            if not part:
                raise RuntimeError("Global tail batch has an empty DDP rank")
            yield part


def global_objective(local_sse, local_n):
    """Value is global mean; DDP-averaged parameter gradient is global mean too."""
    sse = local_sse.detach().to(torch.float64).clone()
    count = local_n.detach().to(device=sse.device, dtype=torch.float64).clone()
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        dist.all_reduce(sse); dist.all_reduce(count)
    valid = count > 0
    if not bool(valid.any()):
        return local_sse.sum() * 0.0
    # Average only over globally supported leads, never per-rank supported leads.
    value = (sse[valid] / count[valid]).mean().to(local_sse.dtype)
    differentiable = world * (local_sse[valid] / count[valid].to(local_sse.dtype)).mean()
    return differentiable + (value - differentiable.detach())


def cot_terms(method, pred, aux, indices, split):
    st = method.__dict__["_attempt13_cot_state"]
    ids = indices.detach().cpu().numpy()
    target = torch.tensor(np.asarray(st["targets"][split][ids]), device=pred.device)
    mask = torch.tensor(np.asarray(st["masks"][split][ids]), device=pred.device, dtype=torch.bool)
    agri = pred.float() * st["agri_std"] + st["agri_mean"]
    geom = aux[:, 8:24].float() * st["geom_std"] + st["geom_mean"]
    x = torch.cat((agri[..., 103:119, 118:134], geom[..., 103:119, 118:134]), dim=2)
    x = (x - st["r_mean"]) / st["r_std"]
    z = st["r_model"](x.reshape(-1, 16, 16, 16)).float().reshape(-1, 16, 1, 16, 16)
    e = torch.where(mask, z - torch.log1p(target.clamp_min(0)), torch.zeros_like(z))
    count = mask.sum(dim=(0, 2, 3, 4))
    physical = torch.expm1(z.clamp(0, math.log1p(100)))
    ep = torch.where(mask, physical - target, torch.zeros_like(z))
    return e.square().sum(dim=(0, 2, 3, 4)), count, ep.double().square().sum(dim=(0, 2, 3, 4))


def patch_method(cls):
    def train_step(self, batch, batch_idx):
        x, y, aux, ids = batch
        p = self(x, y, batch_aux=aux)
        agri = global_objective((p-y).square().sum().reshape(1), p.new_tensor([p.numel()]))
        ss, nn, _ = cot_terms(self, p, aux, ids, "train")
        cot = global_objective(ss, nn)
        loss = agri + self.__dict__["_attempt13_cot_state"]["lambda"] * cot
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("train_agri_mse", agri, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log("train_cot_aux_mse", cot, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def val_start(self):
        self._migration_val = torch.zeros(34, device=self.device, dtype=torch.float64)

    def val_step(self, batch, batch_idx):
        x, y, aux, ids = batch
        p = self(x, y, batch_aux=aux)
        _, n, s = cot_terms(self, p, aux, ids, "val")
        self._migration_val[:16] += s
        self._migration_val[16:32] += n.double()
        self._migration_val[32] += (p-y).double().square().sum()
        self._migration_val[33] += p.numel()

    def val_end(self):
        v = self._migration_val
        if dist.is_initialized(): dist.all_reduce(v)
        valid = v[16:32] > 0
        if not bool(valid.any()): raise RuntimeError("No global validation CPP support")
        lead = torch.sqrt(v[:16][valid] / v[16:32][valid])
        score = lead.mean()
        self.__dict__["_attempt13_last_val_lead_rmse"] = lead.detach().cpu().tolist()
        self.log("val_cot_rmse", score.float(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("val_loss", (v[32]/v[33]).float(), on_step=False, on_epoch=True, sync_dist=False)
    cls.training_step = train_step
    cls.on_validation_epoch_start = val_start
    cls.validation_step = val_step
    cls.on_validation_epoch_end = val_end


def reduction_witness():
    """Unequal masks, rank with zero support, absent lead, and uneven tail batch."""
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", rank)
    reports = []
    for n in (16, 6):
        full = torch.arange(n*4, dtype=torch.float64).reshape(n, 4) / 13
        mask = (torch.arange(n*4).reshape(n,4) % 3 != 0)
        mask[:, 3] = False
        # Identical global reference on every rank, including a zero-support rank.
        mask[0::world, 2] = False
        local = full[rank::world].to(device).clone().requires_grad_(True)
        lm = mask[rank::world].to(device)
        ls = torch.where(lm, local, torch.zeros_like(local)).square().sum(0)
        loss = global_objective(ls, lm.sum(0)); loss.backward()
        ref = full.clone().requires_grad_(True)
        rs = torch.where(mask, ref, torch.zeros_like(ref)).square().sum(0)
        cnt = mask.sum(0); present = cnt > 0
        ref_loss = (rs[present]/cnt[present]).mean(); ref_loss.backward()
        assert torch.allclose(loss.cpu(), ref_loss.detach(), atol=1e-12, rtol=1e-12)
        assert torch.allclose(local.grad.cpu()/world, ref.grad[rank::world], atol=1e-12, rtol=1e-12)
        reports.append({"global_batch":n,"loss":float(loss.detach().cpu()),"gradient_match":True})
    return reports
