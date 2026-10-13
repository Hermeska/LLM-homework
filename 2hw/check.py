from datetime import timedelta

import torch
import torch.distributed as dist

from collectives import ring_all_gather, ring_all_reduce, ring_reduce_scatter


torch.set_num_threads(1)
dist.init_process_group("gloo", timeout=timedelta(seconds=30))
try:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generator = torch.Generator().manual_seed(42 + rank)

    for chunk_size in (1, 7, 31):
        values = torch.randn(world_size * chunk_size, generator=generator, dtype=torch.float64)
        original = values.clone()
        expected_shard = torch.empty(chunk_size, dtype=values.dtype)
        dist.reduce_scatter_tensor(expected_shard, values.clone(), op=dist.ReduceOp.SUM)
        torch.testing.assert_close(ring_reduce_scatter(values), expected_shard)
        torch.testing.assert_close(values, original, rtol=0, atol=0)

        shard = torch.randn(chunk_size, generator=generator, dtype=torch.float64)
        original_shard = shard.clone()
        expected_parts = [torch.empty_like(shard) for _ in range(world_size)]
        dist.all_gather(expected_parts, shard)
        torch.testing.assert_close(ring_all_gather(shard), torch.cat(expected_parts))
        torch.testing.assert_close(shard, original_shard, rtol=0, atol=0)

        expected_sum = values.clone()
        dist.all_reduce(expected_sum, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(ring_all_reduce(values), expected_sum)
        torch.testing.assert_close(values, original, rtol=0, atol=0)

    if rank == 0:
        print(f"OK: все три операции прошли проверки на {world_size} процессах.")
finally:
    dist.destroy_process_group()
