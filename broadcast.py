import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = "192.168.1.100"
os.environ['MASTER_PORT'] = "12355"
os.environ['NCCL_DEBUG'] = "INFO"


def run(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
    print('before broadcast',' Rank ', rank_id, ' has data ', tensor)
    dist.broadcast(tensor, src = 0)
    print('after broadcast',' Rank ', rank_id, ' has data ', tensor)




def init_process(rank_id, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank_id, world_size=size)
    fn(rank_id, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()