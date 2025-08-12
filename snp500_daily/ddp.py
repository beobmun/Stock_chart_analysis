import torch
import torch.distributed as dist
import os
import datetime

def setup_ddp(rank, world_size, gpu_ids):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '55642'

    dist.init_process_group(
        backend='gloo', init_method='env://',
        rank=rank, world_size=world_size,
    )
    
    torch.cuda.set_device(gpu_ids[rank])

def cleanup_ddp():
    dist.destroy_process_group()
    
