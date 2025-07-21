from train import Train
from cnn import CNN

import torch
import matplotlib
from setproctitle import setproctitle
import multiprocessing

matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib

info_path = '../get_data/data/info/kospi_info.csv'
data_path = '../get_data/data/KOSPI'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    setproctitle("Stock")
    learning_rate = 0.00001
    epsilon_init = 1.0
    epsilon_min = 0.1
    epochs = 3000
    batch_size = 16
    transaction_panalty = 0.05
    gamma = 0.99
    
    num_actions = 3
    buffer_size = 1000
    
    model = (Train()
             .set_info_path(info_path)
             .set_data_path(data_path)
             .load_data()
             .set_model(CNN(3, num_actions))
             .set_target_model(CNN(3, num_actions))
             .to(device)
             .set_memory(buffer_size)
             .set_num_workers(multiprocessing.cpu_count()//2)  # Set to 0 for CPU, or adjust based on your system
            )

    model.train(epsilon_init, epsilon_min, epochs, batch_size, learning_rate, transaction_panalty, gamma)