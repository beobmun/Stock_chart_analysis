import random
import pandas as pd
import torch
import torchvision.transforms as transforms
from data_loader import DataInfo, Dataset
from memory import Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch version:", torch.__version__)
print("Using device:", device)
print("gpu count:", torch.cuda.device_count())

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),])

class Train:
    def __init__(self):
        self.info_path = None
        self.data_path = None
        self.data_info = None
        
        self.model = None
        self.memory = None
        
    def set_info_path(self, info_path):
        self.info_path = info_path
        return self
    
    def set_data_path(self, data_path):
        self.data_path = data_path
        return self
    
    def set_model(self, model):
        self.model = model
        return self

    def set_memory(self, buffer_size):
        self.memory = Memory(buffer_size)
        return self
    
    def to(self, device):
        if self.model is not None:
            self.model.to(device)
        else:
            print("Model is not set. Please set the model before calling to().")
        return self
    
    def load_data(self):
        try:
            self.data_info = (DataInfo(self.info_path, self.data_path)
                              .load_info()
                              .get_kospi200_info(listed_date='2020-07-01')
                              .train_test_split(train_size=0.6, val=True, kind=True))
            print("Data loaded successfully.")
            print(f"cnt of train: {len(self.data_info.train_codes)}, val: {len(self.data_info.val_codes)}, test: {len(self.data_info.test_codes)}")
        except Exception as e:
            print(f"Error loading data: {e}")
        return self
    
    def _get_stock_data(self, code):
        try:
            df = pd.read_csv(f"{self.data_path}/{code}.csv")
            df = df.rename(columns={
                    "날짜": "date",
                    "시간": "time",
                    "시가": "open",
                    "고가": "high",
                    "저가": "low",
                    "종가": "close",
                    "거래량": "volume"
                })
            df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df.loc[:, 'time'] = pd.to_datetime(df['time'], format='%H%M%S').dt.time
            df = df[df['date'] >= pd.to_datetime('2020-07-01')]
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            df.set_index('datetime', inplace=True)
            return df
        except Exception as e:
            print(f"Error reading data for {code}: {e}")
            return pd.DataFrame()
   
    def _get_random_date_data(self, df):
        try:
            date = random.choice(df['date'].unique())
            df_d = df[df['date'] == date]  # 랜덤 선택된 날짜의 데이터
            while len(df_d) != 77:
                date = random.choice(df['date'].unique())
                df_d = df[df['date'] == date]
            return df_d
        except Exception as e:
            print(f"Error getting random date data: {e}")
            return pd.DataFrame()

    def _get_random_action(self, num_actions):
        vec = torch.zeros(num_actions)
        idx = random.randrange(0, num_actions)
        vec[idx] = 1
        return vec

    def _get_reward(self, pre_action, cur_action, y, panalty):
        pre_act = 1 - torch.argmax(pre_action).item()
        cur_act = 1 - torch.argmax(cur_action).item()
        return (cur_act * y) - panalty * abs(cur_act - pre_act)
    
    def train(self, epsilon_init, epsilon_min, epochs, batch_size, learning_rate, transation_panalty):
        epsilon = epsilon_init
        num_actions = self.model.fc.fc_layer[-1].out_features
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for code in self.data_info.train_codes:
                print(f"Processing code: {code}")
                try:
                    df = self._get_stock_data(code)
                    df_d = self._get_random_date_data(df) # 랜덤 날짜의 데이터
                    
                    train_dataset = Dataset(df_d, transform=transform)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                    
                    for imgs_batch, yield_batch in train_loader:   # batch_size만큼 imgs, yield가 반환됨.
                        for imgs, y in zip(imgs_batch, yield_batch):
                            pre_state = imgs[0].unsqueeze(0).to(device)
                            cur_state = imgs[1].unsqueeze(0).to(device)
                            next_state = imgs[2].unsqueeze(0).to(device)
                            
                            if round(random.uniform(0, 1), 4) < epsilon:
                                pre_action = self._get_random_action(num_actions)
                            else:
                                with torch.no_grad():
                                    r, pre_action = self.model(pre_state)
                                    print(f"pre_rho: {r}, pre_action: {pre_action}")
                                    
                            if round(random.uniform(0, 1), 4) < epsilon:
                                cur_action = self._get_random_action(num_actions)
                            else:
                                with torch.no_grad():
                                    r, cur_action = self.model(cur_state)
                                    print(f"cur_rho: {r}, cur_action: {cur_action}")
                                    
                            # print(f"pre_action: {pre_action}, cur_action: {cur_action}, yield: {y}")
                            
                            cur_reward = self._get_reward(pre_action, cur_action, y, transation_panalty)
                            print("y", y, "pre_idx: ", torch.argmax(pre_action).item(), "cur_idx: ", torch.argmax(cur_action).item(), "reward: ", cur_reward.item())
                            
                            self.memory.push(cur_state.squeeze(0), cur_action, cur_reward, next_state.squeeze(0))
                            # break
                        states, actions, rewards, next_states = self.memory.get_random_sample(batch_size)
                        states = states.to(device)
                        actions = actions.to(device)
                        rewards = rewards.to(device)
                        next_states = next_states.to(device)
                        
                        print("states shape:", states.shape, "actions shape:", actions.shape, "rewards shape:", rewards.shape, "next_states shape:", next_states.shape)
                        break
                    break
                except Exception as e:
                    print(f"Error processing data for {code}: {e}")
                    continue
            break
