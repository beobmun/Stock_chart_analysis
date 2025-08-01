import torch
import torch.distributed as dist
import os

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "55642"
    
    dist.init_process_group(
        backend='nccl', init_method='env://',
        rank=rank, world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()
    
