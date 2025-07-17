import os
import random
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_loader import DataInfo, Dataset
from memory import Memory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch version:", torch.__version__)
print("Using device:", device)
print("gpu count:", torch.cuda.device_count())

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),])

class Train:
    def __init__(self):
        self.info_path = None
        self.data_path = None
        self.data_info = None
        
        self.model = None
        self.target_model = None
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
    
    def set_target_model(self, target_model):
        self.target_model = target_model
        try:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
        except Exception as e:
            print(f"Error setting target model: {e}")    
        return self

    def set_memory(self, buffer_size):
        self.memory = Memory(buffer_size)
        return self
    
    def to(self, device):
        if self.model is not None and self.target_model is not None:
            self.model.to(device)
            self.target_model.to(device)
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

    def _get_random_action(self, num_actions, batch_size=1):
        actions = []
        for _ in range(batch_size):
            vec = torch.zeros(num_actions)
            idx = random.randrange(0, num_actions)
            vec[idx] = 1
            actions.append(vec)
        return torch.stack(actions)

    def _get_reward(self, pre_actions, cur_actions, y, panalty):
        pre_acts = 1 - torch.argmax(pre_actions, dim=1)
        cur_acts = 1 - torch.argmax(cur_actions, dim=1)
        return (cur_acts * y) - panalty * torch.abs(cur_acts - pre_acts)

    def _update_target_model(self):
        try:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
        except Exception as e:
            print(f"Error updating target model: {e}")
    
    def _get_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return (accuracy, precision, recall, f1)

    def _save_results(self, model, train_loss, train_metrics, val_loss, val_metrics, results_dir="results"):
        os.makedirs(results_dir, exist_ok=True)

        train_df = pd.DataFrame({
            "loss": train_loss,
            "accuracy": train_metrics["accuracy"],
            "precision": train_metrics["precision"],
            "recall": train_metrics["recall"],
            "f1": train_metrics["f1"]
        })
        train_df.to_csv(os.path.join(results_dir, "train_metrics.csv"), index=False)

        val_df = pd.DataFrame({
            "loss": val_loss,
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"]
        })
        val_df.to_csv(os.path.join(results_dir, "val_metrics.csv"), index=False)

        torch.save(model.state_dict(), os.path.join(results_dir, "trained_model.pth"))

    def train(self, epsilon_init, epsilon_min, epochs, batch_size, learning_rate, transation_panalty, gamma):
        epsilon = epsilon_init
        num_actions = self.model.fc.fc_layer[-1].out_features
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        total_train_loss = list()
        total_train_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        total_val_loss = list()
        total_val_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        
        for epoch in range(epochs):
            epoch_loss = list()
            
            self.model.train()
            pbar_train_codes = tqdm(random.sample(self.data_info.train_codes, 30), desc=f"Epoch {epoch+1}/{epochs}", ncols=100, leave=False)
            y_true = list()
            y_pred = list()
            for code in pbar_train_codes:
                code_loss = list()
                # print(f"Processing code: {code}")
                try:
                    df = self._get_stock_data(code)
                    df_d = self._get_random_date_data(df) # 랜덤 날짜의 데이터
                    
                    train_dataset = Dataset(df_d, transform=transform)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                    
                    with tqdm(total=len(train_dataset), desc=f"Training {code}", ncols=100, leave=False) as pbar:
                        for imgs_batch, yield_batch in train_loader:   # batch_size만큼 imgs, yield가 반환됨.
                            imgs_batch = imgs_batch.to(device)
                            yield_batch = yield_batch.to(device)
                            
                            pre_states = imgs_batch[:, 0, :, :, :]  # pre_state
                            cur_states = imgs_batch[:, 1, :, :, :]  # cur_state
                            next_states = imgs_batch[:, 2, :, :, :]  # next_state
                            
                            if round(random.uniform(0, 1), 4) < epsilon:
                                pre_actions = self._get_random_action(num_actions, batch_size).to(device)
                                cur_actions = self._get_random_action(num_actions, batch_size).to(device)
                                
                            else:
                                with torch.no_grad():
                                    _, pre_actions = self.model(pre_states)
                                    _, cur_actions = self.model(cur_states)
                            
                            cur_rewards = self._get_reward(pre_actions, cur_actions, yield_batch, transation_panalty)
                            for i in range(batch_size):
                                self.memory.push(
                                    cur_states[i].squeeze(0),
                                    cur_actions[i].unsqueeze(0),
                                    cur_rewards[i],
                                    next_states[i].squeeze(0),
                                    yield_batch[i]
                                )
                            
                            if epsilon > epsilon_min:
                                epsilon *= 0.999999
                            
                            pbar.update(batch_size)
                            
                        if len(self.memory) >= batch_size:
                            states, actions, rewards, next_states, y_t = self.memory.get_random_sample(batch_size)
                            states = states.to(device)
                            actions = actions.to(device)
                            rewards = rewards.to(device)
                            next_states = next_states.to(device)
                            
                            q_values, _ = self.model(states)
                            q_values = torch.sum(q_values * actions, dim=1).max(dim=1)[0]
                            
                            with torch.no_grad():
                                next_q_values, _ = self.target_model(next_states)
                                next_q_max = next_q_values.max(dim=1)[0]
                                q_targets = rewards + gamma * next_q_max
                            
                            loss = criterion(q_values, q_targets)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                            code_loss.append(loss.item())
                            
                            y_p = 1 - actions.squeeze(1).argmax(dim=1)
                            z = torch.where(y_t == 0)
                            y_t = torch.where(y_t > 0, 1, -1)
                            y_t[z] = 0
                            y_true += y_t
                            y_pred += y_p
                            # print(f"Loss: {loss.item()}, Epsilon: {epsilon}")
                            # print("states shape:", states.shape, "actions shape:", actions.shape, "rewards shape:", rewards.shape, "next_states shape:", next_states.shape)
                        # break
                    # break
                    self._update_target_model()
                    # print(f"Updated target model for code: {code}")
                    epoch_loss.append(np.mean(code_loss))
                except Exception as e:
                    print(f"(Train)Error processing data for {code}: {e}")
                    continue
                
            total_train_loss.append(np.mean(epoch_loss))
            y_pred = torch.tensor(y_pred)
            y_true = torch.tensor(y_true)
            
            accuracy, precision, recall, f1 = self._get_metrics(y_true, y_pred)
            total_train_metrics["accuracy"].append(accuracy)
            total_train_metrics["precision"].append(precision)
            total_train_metrics["recall"].append(recall)
            total_train_metrics["f1"].append(f1)
            
            print("=" * 50)
            print(f"Epoch {epoch+1}/{epochs} Train Loss: {total_train_loss[-1]:.4f} \t\t| Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
            
            self.model.eval()
            pbar_val_codes = tqdm(self.data_info.val_codes, desc=f"Validation {epoch+1}/{epochs}", ncols=100, leave=False)
            y_true = list()
            y_pred = list()
            val_loss = list()
            for code in pbar_val_codes:
                code_loss = list()
                try:
                    df = self._get_stock_data(code)
                    df_d = self._get_random_date_data(df)  # 랜덤 날짜의 데이터
                    
                    val_dataset = Dataset(df_d, transform=transform)
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    
                    with tqdm(total=len(val_dataset), desc=f"Validating {code}", ncols=100, leave=False) as pbar:
                        for imgs_batch, yield_batch in val_loader:
                            imgs_batch = imgs_batch.to(device)
                            yield_batch = yield_batch.to(device)
                            
                            pre_states = imgs_batch[:, 0, :, :, :]
                            cur_states = imgs_batch[:, 1, :, :, :]
                            next_states = imgs_batch[:, 2, :, :, :]
                            
                            with torch.no_grad():
                                _, pre_actions = self.model(pre_states)
                                q_values, cur_actions = self.model(cur_states)
                                rewards = self._get_reward(pre_actions, cur_actions, yield_batch, 0)
                                cur_actions = cur_actions.unsqueeze(1)
                                q_values = torch.sum(q_values * cur_actions, dim=1).max(dim=1)[0]
                                next_q_values, _ = self.target_model(next_states)
                                next_q_max = next_q_values.max(dim=1)[0]
                            
                            
                            q_targets = rewards + gamma * next_q_max
                            loss = criterion(q_values, q_targets)
                            code_loss.append(loss.item())
                            
                            y_p = 1 - cur_actions.squeeze(1).argmax(dim=1)
                            z = torch.where(yield_batch == 0)
                            y_t = torch.where(yield_batch > 0, 1, -1)
                            y_t[z] = 0
                            y_true += y_t
                            y_pred += y_p
                            
                            pbar.update(batch_size)
                    val_loss.append(np.mean(code_loss))
                except Exception as e:
                    print(f"(Validation)Error reading data for {code}: {e}")
            
            y_pred = torch.tensor(y_pred)
            y_true = torch.tensor(y_true)
            
            accuracy, precision, recall, f1 = self._get_metrics(y_true, y_pred)
            total_val_loss.append(np.mean(val_loss))
            total_val_metrics["accuracy"].append(accuracy)
            total_val_metrics["precision"].append(precision)
            total_val_metrics["recall"].append(recall)
            total_val_metrics["f1"].append(f1)
            
            print(f"Epoch {epoch+1}/{epochs} Validation Loss: {total_val_loss[-1]:.4f} \t| Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
            print("=" * 50)
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f"trained_model_epoch_{epoch+1}.pth")
        print("Training complete.")
        
        self._save_results(self.model, total_train_loss, total_train_metrics, total_val_loss, total_val_metrics)