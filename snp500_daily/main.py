from model import Model
from cnn import CNN
from dqn_resnet50 import DQN_ResNet50

import torch
import matplotlib
from setproctitle import setproctitle
import multiprocessing

matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib

info_path = '../get_data/data/info/snp500_info.csv'
data_path = '../get_data/data/US'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    setproctitle("Stock_S&P500")
    
    learning_rate = 1e-4
    epsilon_init = 1.0
    epsilon_min = 0.1
    epochs = 30000
    batch_size = 32
    transaction_penalty = 0.05
    gamma = 0.99
    
    num_actions = 3
    buffer_size = 1000
    
    train_start_date = '2015-01-01'
    train_end_date = '2021-12-31'
    val_start_date = '2022-01-01'
    val_end_date = '2024-12-31'
    # save_dir = "results_res50_pretrained"
    save_dir = "results_res50_pretrained_exp_neut"
    
    ### Training the model
    # model = (Model()
    #          .set_info_path(info_path)
    #          .set_data_path(data_path)
    #          .load_info()
    #          .set_df_cache()
    #          .set_model(CNN(3, num_actions))
    #          .set_target_model(CNN(3, num_actions))
    #          .to(device)
    #          .set_memory(buffer_size)
    #          .set_num_workers(multiprocessing.cpu_count()//2)  # Set to 0 for CPU, or adjust based on your system
    #         )
    model = (Model()
            .set_info_path(info_path)
            .set_data_path(data_path)
            .load_info()
            .set_df_cache()
            .set_model(DQN_ResNet50(num_actions, pretrained=True))
            .set_target_model(DQN_ResNet50(num_actions, pretrained=True))
            .to(device)
            .set_memory(buffer_size)
            .set_num_workers(multiprocessing.cpu_count()//2)  # Set to 0 for CPU, or adjust based on your system
        )
    model.train(epsilon_init, epsilon_min, epochs, batch_size, learning_rate, transaction_penalty, gamma, \
        train_start_date, train_end_date, val_start_date, val_end_date, save_dir)
    
    
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