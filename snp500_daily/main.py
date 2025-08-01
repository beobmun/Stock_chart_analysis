from model import Model
from cnn import CNN
from dqn_resnet50 import DQN_ResNet50
from ddp import setup_ddp, cleanup_ddp

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import matplotlib
from setproctitle import setproctitle
import multiprocessing

matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(rank, world_size):
    setup_ddp(rank, world_size)
    
    setproctitle(f"Stock_S&P500_rank {rank}")
    
    learning_rate = 1e-4
    epsilon_init = 1.0
    epsilon_min = 0.1
    epochs = 30000
    batch_size = 512
    transaction_penalty = 0.05
    gamma = 0.99
    
    num_actions = 3
    buffer_size = batch_size * 10
    
    train_start_date = '2015-01-01'
    train_end_date = '2021-12-31'
    val_start_date = '2022-01-01'
    val_end_date = '2024-12-31'
    
    info_path = '../get_data/data/info/snp500_info.csv'
    data_path = '../get_data/data/US/time_series'
    imgs_dir = '../get_data/data/US/chart_image'
    save_dir = "results_res50_pretrained_all"
    
    ### Training the model
    '''
    # multigpu 사용
    learn_model = DQN_ResNet50(num_actions, pretrained=True).to(rank)
    setproctitle(f"Stock_S&P500_rank {rank} - learn_model")
    learn_model = DDP(learn_model, device_ids=[rank], find_unused_parameters=False)

    target_model = DQN_ResNet50(num_actions, pretrained=True).to(rank)
    setproctitle(f"Stock_S&P500_rank {rank} - target_model")
    target_model = DDP(target_model, device_ids=[rank], find_unused_parameters=False)

    model = (Model()
             .set_info_path(info_path)
             .set_data_path(data_path)
             .load_info()
             .set_df_cache()
             .set_model(learn_model)
             .set_target_model(target_model)
             .set_rank_world_size(rank, world_size)
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
    
    # '''
    # 1개 gpu로 학습할 때
    model = (Model()
            .set_info_path(info_path)
            .set_data_path(data_path)
            .load_info()
            .set_df_cache()
            .set_model(DQN_ResNet50(num_actions, pretrained=True))
            .set_target_model(DQN_ResNet50(num_actions, pretrained=True))
            .to(device)
            .set_memory(buffer_size)
            .set_num_workers(multiprocessing.cpu_count()//4)  # Set to 0 for CPU, or adjust based on your system
        )
    model.train(epsilon_init, epsilon_min, epochs, batch_size, learning_rate, transaction_penalty, gamma, \
        train_start_date, train_end_date, val_start_date, val_end_date, imgs_dir, save_dir)
    # model.train2(epsilon_init, epsilon_min, epochs, batch_size, learning_rate, transaction_penalty, gamma, \
    #     train_start_date, train_end_date, val_start_date, val_end_date, imgs_dir, save_dir)
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
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    except KeyboardInterrupt:
        cleanup_ddp()
        print("Training interrupted. DDP cleaned up.")
    '''
    # Single GPU training setup
    main(rank=0, world_size=1)