from train import Train
from cnn import CNN

import torch

info_path = '../get_data/data/info/kospi_info.csv'
data_path = '../get_data/data/KOSPI'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    learning_rate = 0.00001
    epsilon_init = 1
    epsilon_min = 0.1
    epochs = 1
    batch_size = 3
    transaction_panalty = 0
    
    num_actions = 3
    buffer_size = 1000
    
    model = (Train()
             .set_info_path(info_path)
             .set_data_path(data_path)
             .load_data()
             .set_model(CNN(3, num_actions))
             .set_memory(buffer_size)
             .to(device))

    model.train(epsilon_init, epsilon_min, epochs, batch_size, learning_rate, transaction_panalty)