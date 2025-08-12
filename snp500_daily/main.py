from model import Model
from cnn import CNN
from dqn_resnet50 import DQN_ResNet50
from dqn_internimage import DQN_InternImage
from dqn_efficientnet import DQN_EfficientNet
from ddp import setup_ddp, cleanup_ddp

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import matplotlib
from setproctitle import setproctitle
import multiprocessing

matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib

# print("Torch version:", torch.__version__)
# print("gpu count:", torch.cuda.device_count())
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# single GPU 사용할 경우, 현재 파일의 device와 model.py의 device를 동일하게 설정

def main(rank=0, world_size=1, gpu_ids=[0]):
    if world_size > 1:
        setup_ddp(rank, world_size, gpu_ids)
    else:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

    setproctitle(f"Stock_S&P500_rank {rank}")
    
    learning_rate = 1e-4
    epsilon_init = 1.0
    epsilon_min = 0.1
    epochs = 10000
    batch_size = 512
    transaction_penalty = 0.05
    gamma = 0.99
    
    num_actions = 3
    buffer_size = 50000
    
    train_start_date = '2015-01-01'
    train_end_date = '2021-12-31'
    val_start_date = '2022-01-01'
    val_end_date = '2024-12-31'
    
    info_path = '../get_data/data/info/snp500_info.csv'
    data_path = '../get_data/data/US/time_series'
    imgs_dir = '../get_data/data/US/chart_image'
    # save_dir = 'results_cnn'
    # save_dir = "results_res50_pretrained"
    # save_dir = "results_internimage_t_1k_224"
    save_dir = "results_efficientnet_b4_pretrained"

    
    ### Training the model
    
    # multigpu 사용
    '''
    local_gpu_id = gpu_ids[rank]
    # learn_model = CNN(3, num_actions).to(local_gpu_id)
    # learn_model = DQN_ResNet50(num_actions, pretrained=True).to(local_gpu_id)
    learn_model = DQN_EfficientNet(num_actions, pretrained=True).to(local_gpu_id)
    setproctitle(f"Stock_S&P500_rank {local_gpu_id} - learn_model")
    learn_model = DDP(learn_model, device_ids=[local_gpu_id], find_unused_parameters=True)

    # target_model = CNN(3, num_actions).to(local_gpu_id)
    # target_model = DQN_ResNet50(num_actions, pretrained=True).to(local_gpu_id)
    target_model = DQN_EfficientNet(num_actions, pretrained=True).to(local_gpu_id)
    setproctitle(f"Stock_S&P500_rank {local_gpu_id} - target_model")
    target_model = DDP(target_model, device_ids=[local_gpu_id], find_unused_parameters=True)

    model = (Model()
             .set_info_path(info_path)
             .set_data_path(data_path)
             .load_info()
             .set_df_cache()
             .set_model(learn_model)
             .set_target_model(target_model)
             .set_gpu_ids(rank, world_size, gpu_ids)
             .set_memory(buffer_size)
             .set_num_workers(multiprocessing.cpu_count()//4)  
             )
    
    model.multi_gpu_train(
        epsilon_init, epsilon_min, num_actions,
        epochs, batch_size, learning_rate, transaction_penalty, gamma,
        train_start_date, train_end_date, val_start_date, val_end_date,
        imgs_dir, save_dir
    )
    
    cleanup_ddp()
    # '''
    

    # 1개 gpu로 학습할 때
    # '''
    model = (Model()
            .set_info_path(info_path)
            .set_data_path(data_path)
            .load_info(listed_date=train_start_date)
            .set_df_cache()
            # .set_model(CNN(3, num_actions))  
            # .set_target_model(CNN(3, num_actions))
            # .set_model(DQN_ResNet50(num_actions, pretrained=True))
            # .set_target_model(DQN_ResNet50(num_actions, pretrained=True))
            # .set_model(DQN_InternImage(num_actions, model_path="./models/internimage_t_1k_224"))
            # .set_target_model(DQN_InternImage(num_actions, model_path="./models/internimage_t_1k_224"))
            .set_model(DQN_EfficientNet(num_actions, pretrained=True))
            .set_target_model(DQN_EfficientNet(num_actions, pretrained=True))
            .to(device)
            .set_memory(buffer_size)
            .set_num_workers(multiprocessing.cpu_count()//4)  # Set to 0 for CPU, or adjust based on your system
        )
    
    model.train(
        epsilon_init, epsilon_min, num_actions,
        epochs, batch_size, learning_rate, transaction_penalty, gamma,
        train_start_date, train_end_date, val_start_date, val_end_date, 
        imgs_dir, save_dir
        )
    # '''
    
    ### Testing the model
    '''
    model_name = ""
    test_model_state_path = f"results_sl/models/{model_name}.pth"
    
    model = (Model()
             .set_info_path(info_path)
             .set_data_path(data_path)
             .load_info()
             .set_df_cache()
             .set_test_model(CNN(3, num_actions), test_model_state_path)
             .to(device, test=True)
             .set_num_workers(multiprocessing.cpu_count()//2)  # Set to 0 for CPU, or adjust based on your system
             )
    
    '''
    
if __name__ == "__main__":
    # Multi GPU training setup
    '''
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your setup.")
        
        # Use all available GPUs
        # world_size = torch.cuda.device_count()
        # gpu_ids = list(range(world_size))
        # mp.spawn(main, args=(world_size, gpu_ids), nprocs=world_size, join=True)
        
        # Use specific GPUs
        gpu_ids = [0, 1]
        world_size = len(gpu_ids)
        mp.spawn(main, args=(world_size, gpu_ids), nprocs=world_size, join=True)

    except (KeyboardInterrupt, RuntimeError) as e:
        print("Training interrupted or failed.")
        print("Error message:", e)
        cleanup_ddp()

    # '''
    
    # Single GPU training setup
    main(rank=1)