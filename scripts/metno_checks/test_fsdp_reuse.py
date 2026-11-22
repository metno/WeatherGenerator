"""Does FSDP2 give the right gradients when a sharded module is called M times per forward?"""
import copy, os
import torch, torch.distributed as dist, torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

class Net(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = torch.nn.Linear(8, 16); s.dec = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.GELU(), torch.nn.Linear(16, 4))
    def forward(s, x, M):
        h = s.enc(x)
        outs = [s.dec(h + 0.1 * m) for m in range(M)]
        return torch.stack(outs).pow(2).mean()

def run(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29541")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    torch.manual_seed(0); ref = Net(); net = copy.deepcopy(ref)
    fully_shard(net.dec[0], mesh=mesh); fully_shard(net.dec[2], mesh=mesh); fully_shard(net.dec, mesh=mesh); fully_shard(net, mesh=mesh)
    torch.manual_seed(1 + rank); x = torch.randn(5, 8)
    xs = [torch.zeros_like(x) for _ in range(world)]; dist.all_gather(xs, x)
    for M in (1, 3):
        net.zero_grad(); ref.zero_grad()
        net(x, M).backward()
        # reference: mean of per-rank losses on the full model (FSDP averages grads)
        sum(ref(xi, M) for xi in xs).div(world).backward()
        for (n, p), (_, q) in zip(net.named_parameters(), ref.named_parameters()):
            g = p.grad.full_tensor()
            assert torch.allclose(g, q.grad, atol=1e-6), (M, n, (g - q.grad).abs().max())
        if rank == 0: print(f"M={M}: FSDP2 gradients equal the reference")
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(run, args=(2,), nprocs=2, join=True); print("REUSE TEST PASSED")
