import argparse
import os
import sys
import tempfile
from time import sleep
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import datetime


from torch.nn.parallel import DistributedDataParallel as DDP


def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
    tensor = torch.arange(2, dtype=torch.int64).cuda(local_rank) + 1 + 2 * rank
    print('before reudce',' Rank ', rank, ' has data ', tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('after reudce',' Rank ', rank, ' has data ', tensor)
    
def run():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5))
    #dist.init_process_group(backend="nccl")
    train()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()