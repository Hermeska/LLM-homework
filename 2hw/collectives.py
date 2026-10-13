import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size 
    for step in range(world_size - 1): 
        send_index = (rank - step - 1) % world_size
        rec_index = (rank - step - 2)% world_size

        rec_buf = torch.empty_like(chunks[rec_index])
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(
                    dist.isend,
                    chunks[send_index],
                    next_rank,
                ),
                dist.P2POp(
                    dist.irecv,
                    rec_buf,
                    prev_rank,
                ),
            ]
        )

        for request in requests:
            request.wait()

        chunks[rec_index].add_(rec_buf)
    return chunks[rank]

def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)
    next_rank = (rank + 1) % world_size
    previous_rank = (rank - 1) % world_size

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        receive_index = (rank - step - 1) % world_size

        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(
                    dist.isend,
                    chunks[send_index],
                    next_rank,
                ),
                dist.P2POp(
                    dist.irecv,
                    chunks[receive_index],
                    previous_rank,
                ),
            ]
        )

        for request in requests:
            request.wait()
    return torch.cat(chunks, dim = 0)

def ring_all_reduce(tensor):
    reduced_shard = ring_reduce_scatter(tensor)
    return ring_all_gather(reduced_shard)
