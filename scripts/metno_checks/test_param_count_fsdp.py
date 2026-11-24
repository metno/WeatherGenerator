"""Parameter count under FSDP2 (CPU, 2 processes, gloo):
get_num_parameters() (used by Model.print_num_parameters) gives the same count on a model
sharded with fully_shard as on the plain model, because sharded parameters are DTensors whose
size() is the global (unsharded) shape. Frozen parameters are not counted in both cases.
So the parameter printout is also valid in multi-GPU (FSDP) runs."""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import fully_shard

from weathergen.model.utils import get_num_parameters


def make_model():
    torch.manual_seed(0)
    m = torch.nn.Sequential(
        torch.nn.Linear(37, 64), torch.nn.GELU(), torch.nn.Linear(64, 11), torch.nn.LayerNorm(11)
    )
    m[3].weight.requires_grad = False  # a frozen parameter: must not be counted
    return m


def run(rank, world_size, ref, port):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    m = make_model()
    fully_shard(m[0])
    fully_shard(m)
    local = sum(p.to_local().numel() for p in m.parameters() if p.requires_grad)
    n = int(get_num_parameters(m))
    print(f"rank {rank}: sharded count {n:,} (local shard {local:,}), plain {ref:,}")
    assert n == ref, (n, ref)
    assert local < ref
    dist.destroy_process_group()


if __name__ == "__main__":
    ref = int(get_num_parameters(make_model()))
    assert ref == 37 * 64 + 64 + 64 * 11 + 11 + 11  # LayerNorm weight frozen, bias counted
    mp.spawn(run, args=(2, ref, 29517), nprocs=2, join=True)
    print("OK")
