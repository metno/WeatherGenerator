"""Checks for weathergen.model.stochastic (CPU): zero start, noise structure, seeds,
meta-device build + re-initialisation, FSDP2 on 2 CPU ranks with 2 members."""
import copy
import os
import types

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from weathergen.model.model import reinit_parameters
from weathergen.model.stochastic import LatentNoise


class Cf(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

    def __getattr__(self, k):
        return self[k]


D = 64
# a compact patch of hl8 cells in nested order: 4 hl6 parents x 16 children
cells = np.arange(1024, 1024 + 64, dtype=np.int64)
N = len(cells)
nb = torch.stack([torch.arange(N)] + [torch.roll(torch.arange(N), s) for s in (1, -1, 2, -2, 3, -3, 4, -4)], 1)
mp_ = types.SimpleNamespace(hp_nbours=nb.int(), domain=types.SimpleNamespace(active_cells=cells))
params = types.SimpleNamespace(params_for=lambda lvl: mp_)


def cf_(**sc):
    base = dict(enabled=True, level=8, noise_levels=[6, 7], noise_dim=4, smooth_passes=0,
                num_blocks=1, att_dim=32, att_heads=4, members_train=2, members_val=3, seed=7)
    base.update(sc)
    return Cf(ae_global_dim_embed=D, ae_local_num_queries=1, num_register_tokens=0,
              num_class_tokens=0, norm_eps=1e-5, stochastic=base)


def perturb(m, scale=0.1):
    # make the zero-initialised parts non-zero (as after some training)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "embed_aux.2" in n or n.endswith("slot"):
                p.normal_(0, scale)


def checks():
    # 1) zero start: output bit-identical to input for every member
    torch.manual_seed(0)
    noise = LatentNoise(cf_(), 8)
    x = torch.randn(2, N, D)
    noise.train(); noise.begin_forward()
    for m in range(2):
        assert torch.equal(noise(x, params, m), x)
    print("1) zero start: output identical to input (bitwise)")

    # 2) noise structure: before smoothing, the hl6 part is shared by the 16 children of a parent,
    #    the hl7 part by 4 children
    z = noise.draw(1, cells, nb, 0, "cpu")[0]
    z6, z7 = z[:, :4], z[:, 4:]
    assert torch.equal(z6[0], z6[15]) and not torch.equal(z6[0], z6[16])
    assert torch.equal(z7[0], z7[3]) and not torch.equal(z7[0], z7[4])
    zs = LatentNoise(cf_(smooth_passes=1), 8).draw(1, cells, nb, 0, "cpu")[0]
    assert torch.allclose(zs.std(0), torch.ones(8), atol=1e-4)
    print("2) parent noise shared by children; smoothing keeps unit variance")

    # 3) seeds: members differ; training steps differ; every validation repeats the same noise
    a0 = noise.draw(1, cells, nb, 0, "cpu"); a1 = noise.draw(1, cells, nb, 1, "cpu")
    assert not torch.equal(a0, a1)
    noise.begin_forward(); b0 = noise.draw(1, cells, nb, 0, "cpu")
    assert not torch.equal(a0, b0)
    noise.eval(); noise.begin_forward(); v1 = noise.draw(1, cells, nb, 0, "cpu")
    noise.begin_forward(); v1b = noise.draw(1, cells, nb, 0, "cpu")
    noise.train(); noise.begin_forward()
    noise.eval(); noise.begin_forward(); v2 = noise.draw(1, cells, nb, 0, "cpu")
    assert torch.equal(v1, v2) and not torch.equal(v1, v1b)
    assert noise.num_members() == 3
    noise.train(); assert noise.num_members() == 2
    print("3) seeds: members and steps differ, validations repeat, members_train/members_val ok")

    # 4) after perturbing, members differ and gradients reach every parameter
    perturb(noise)
    noise.begin_forward()
    y0, y1 = noise(x, params, 0), noise(x, params, 1)
    assert not torch.allclose(y0, y1)
    (y0.pow(2).mean() + y1.pow(2).mean()).backward()
    missing = [n for n, p in noise.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, missing
    print("4) members differ after training; gradients reach all", sum(1 for _ in noise.parameters()),
          "parameter tensors")

    # 5) meta-device build, to_empty, reinit_parameters -> zero start restored
    with torch.device("meta"):
        meta = LatentNoise(cf_(), 8)
    meta.to_empty(device="cpu")
    reinit_parameters(meta)
    meta.train(); meta.begin_forward()
    assert torch.equal(meta(x, params, 0), x)
    assert all(torch.isfinite(p).all() for p in meta.parameters())
    print("5) meta build + reinit_parameters: zero start restored")


# 6) FSDP2, 2 CPU ranks, 2 members, compared with an unsharded copy
def run(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29551")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    torch.manual_seed(0)
    ref = LatentNoise(cf_(), 8)
    perturb(ref)
    net = copy.deepcopy(ref)
    for blk in net.blocks:
        fully_shard(blk, mesh=mesh)
    fully_shard(net, mesh=mesh)
    torch.manual_seed(10 + rank)
    xr = torch.randn(1, N, D)
    for mdl in (net, ref):
        mdl.train(); mdl.begin_forward()
    loss = sum(net(xr, params, m).pow(2).mean() for m in range(2))
    loss.backward()
    # reference: each rank's loss on the unsharded copy (same seed: same rank, same call)
    loss_ref = sum(ref(xr, params, m).pow(2).mean() for m in range(2))
    assert torch.allclose(loss, loss_ref, atol=1e-6), (loss, loss_ref)
    loss_ref.backward()
    for (n, p), (_, q) in zip(net.named_parameters(), ref.named_parameters()):
        g = p.grad.full_tensor()
        qg = q.grad.clone()
        dist.all_reduce(qg)
        assert torch.allclose(g, qg / world, atol=1e-5), (n, (g - qg / world).abs().max())
    if rank == 0:
        print("6) FSDP2 on 2 CPU ranks, 2 members: loss and gradients equal the unsharded copy")
    dist.destroy_process_group()


if __name__ == "__main__":
    checks()
    mp.spawn(run, args=(2,), nprocs=2, join=True)
    print("ALL STOCHASTIC TESTS PASSED")
