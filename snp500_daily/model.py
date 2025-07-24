import numpy as np
import pandas as pd
import random
import os
from typing import Literal
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from setproctitle import setproctitle

from data_loader import DataInfo, Dataset
from memory import Memory

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch version:", torch.__version__)
print("Using device:", device)
print("gpu count:", torch.cuda.device_count())

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),])

class Model:
    def __init__(self):
        self.info_path = None
        self.data_path = None
        self.data_info = None
        
        self.model = None
        self.target_model = None
        self.test_model = None
        self.memory = None
        # self.stock_data_cache = dict()
        self.train_df_cache = dict()
        self.val_df_cache = dict()
        self.test_df_cache = dict()
        self.num_workers = 0
        
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
            print(f"Error loading target model state: {e}")
        return self
    
    def set_memory(self, buffer_size=1000):
        self.memory = Memory(buffer_size)
        return self
    
    def set_num_workers(self, num_workers):
        self.num_workers = num_workers
        return self
    
    def set_df_cache(self, type: Literal['train', 'val', 'test', 'all']='all'):
        try:
            if type == 'all':
                pbar_tr = tqdm(self.data_info.train_symbols, desc="Loading train data", ncols=100, leave=False)
                self.train_df_cache = {symbol: self._get_stock_data(symbol) for symbol in pbar_tr}
                pbar_v = tqdm(self.data_info.val_symbols, desc="Loading validation data", ncols=100, leave=False)
                self.val_df_cache = {symbol: self._get_stock_data(symbol) for symbol in pbar_v}
                pbar_te = tqdm(self.data_info.test_symbols, desc="Loading test data", ncols=100, leave=False)
                self.test_df_cache = {symbol: self._get_stock_data(symbol) for symbol in pbar_te}
                print("DataFrame cache set for all dataset.")
            elif type == 'train':
                pbar_tr = tqdm(self.data_info.train_symbols, desc="Loading train data", ncols=100, leave=False)
                self.train_df_cache = {symbol: self._get_stock_data(symbol) for symbol in pbar_tr}
                print("DataFrame cache set for train dataset.")
            elif type == 'val':
                pbar_v = tqdm(self.data_info.val_symbols, desc="Loading validation data", ncols=100, leave=False)
                self.val_df_cache = {symbol: self._get_stock_data(symbol) for symbol in pbar_v}
                print("DataFrame cache set for validation dataset.")
            elif type == 'test':
                pbar_te = tqdm(self.data_info.test_symbols, desc="Loading test data", ncols=100, leave=False)
                self.test_df_cache = {symbol: self._get_stock_data(symbol) for symbol in pbar_te}
                print("DataFrame cache set for test dataset.")
        except Exception as e:
            print(f"Error setting DataFrame cache: {e}")
        return self
        
    def set_test_model(self, model, model_state_path=None):
        try:
            self.test_model = model
            self.test_model.load_state_dict(torch.load(model_state_path))
            self.test_model.eval()
        except Exception as e:
            print(f"Error setting test model: {e}")
        return self
        
    def to(self, device, test=False):
        if test:
            if self.test_model is not None:
                self.test_model.to(device)
            else:
                print("Test model is not set. Please set it before calling to().")
            return self
        if self.model is not None and self.target_model is not None:
            self.model.to(device)
            self.target_model.to(device)
        else:
            print("Model or target model is not set. Please set them before calling to().")
        return self
    
    def load_info(self):
        try:
            self.data_info = (DataInfo(self.info_path, self.data_path)
                              .load_info()
                              .train_test_split(train_size=0.6, val=True, sector=True))
        except Exception as e:
            print(f"Error loading info: {e}")
        return self
    
    def _get_stock_data(self, symbol):
        symbol = symbol.lower().replace(".", "-")
        try:
            df = pd.read_csv(f"{self.data_path}/{symbol}.us.txt")
            df = df.rename(columns={
                "<DATE>": "date",
                "<OPEN>": "open",
                "<HIGH>": "high",
                "<LOW>": "low",
                "<CLOSE>": "close",
                "<VOL>": "volume",
            })
            df.loc[:, "date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df.set_index("date", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]]
            df = df.sort_index()
            return df
        except Exception as e:
            print(f"Error getting stock data for {symbol}: {e}")
            return pd.DataFrame()

    def _get_random_action(self, num_actions, batch_size):
        actions = list()
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

    def _save_results(self, model, train_loss, train_metrics, val_loss_1, val_metrics_1, val_loss_2, val_metrics_2, val_loss_3, val_metrics_3, result_dir="results", epoch="final"):
        os.makedirs(result_dir, exist_ok=True)
        model_dir = os.path.join(result_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        train_df = pd.DataFrame({
            "loss": train_loss,
            "accuracy": train_metrics["accuracy"],
            "precision": train_metrics["precision"],
            "recall": train_metrics["recall"],
            "f1": train_metrics["f1"]
        })
        val_df_1 = pd.DataFrame({
            "loss": val_loss_1,
            "accuracy": val_metrics_1["accuracy"],
            "precision": val_metrics_1["precision"],
            "recall": val_metrics_1["recall"],
            "f1": val_metrics_1["f1"]
        })
        val_df_2 = pd.DataFrame({
            "loss": val_loss_2,
            "accuracy": val_metrics_2["accuracy"],
            "precision": val_metrics_2["precision"],
            "recall": val_metrics_2["recall"],
            "f1": val_metrics_2["f1"]
        })
        val_df_3 = pd.DataFrame({
            "loss": val_loss_3,
            "accuracy": val_metrics_3["accuracy"],
            "precision": val_metrics_3["precision"],
            "recall": val_metrics_3["recall"],
            "f1": val_metrics_3["f1"]
        })
        
        train_df.to_csv(os.path.join(result_dir, f"train_results.csv"), index=False)
        val_df_1.to_csv(os.path.join(result_dir, f"val_results_1.csv"), index=False)
        val_df_2.to_csv(os.path.join(result_dir, f"val_results_2.csv"), index=False)
        val_df_3.to_csv(os.path.join(result_dir, f"val_results_3.csv"), index=False)

        torch.save(model.state_dict(), os.path.join(model_dir, f"s&p500_model_epoch_{epoch}.pth"))

    def _validate(self, dataset, batch_size, gamma, criterion):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.model.eval()
        v_loss = list()
        y_true = list()
        y_pred = list()
        
        with tqdm(total=len(dataset), desc="Validation", ncols=100, leave=False) as pbar:
            with torch.no_grad():
                for imgs_batch, yield_batch in dataloader:
                    imgs_batch = imgs_batch.to(device)
                    yield_batch = yield_batch.to(device)
                    
                    pre_states = imgs_batch[:, 0, :, :, :]
                    cur_states = imgs_batch[:, 1, :, :, :]
                    next_states = imgs_batch[:, 2, :, :, :]
                    
                    _, pre_actions = self.model(pre_states)
                    q_values, cur_actions = self.model(cur_states)
                    rewards = self._get_reward(pre_actions, cur_actions, yield_batch, 0)
                    q_values = torch.sum(q_values * cur_actions, dim=1)
                    next_q_values, _ = self.target_model(next_states)
                    next_q_max = next_q_values.max(dim=1)[0]
                    
                    q_targets = rewards + gamma * next_q_max
                    loss = criterion(q_values, q_targets)
                    v_loss.append(loss.item())
                    y_p = 1 - cur_actions.squeeze(1).argmax(dim=1)
                    z = torch.where(yield_batch == 0)
                    y_t = torch.where(yield_batch > 0, 1, -1)
                    y_t[z] = 0
                    y_true.append(y_t)
                    y_pred.append(y_p)
                    pbar.update(imgs_batch.shape[0])
        y_pred = torch.cat(y_pred).to("cpu")
        y_true = torch.cat(y_true).to("cpu")
        acc, prec, rec, f1 = self._get_metrics(y_true, y_pred)
        return np.mean(v_loss), (acc, prec, rec, f1)

    def train(self, epsilon_init, epsilon_min, epochs, batch_size, learning_rate, transaction_penalty, gamma, train_s, train_e, val_s, val_e):
        epsilon = epsilon_init
        num_actions = self.model.fc.fc_layer[-1].out_features
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # criterion = torch.nn.MSELoss()
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.SmoothL1Loss()
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9995 ** epoch)
        
        total_train_loss = list()
        total_train_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        total_val_loss_1 = list()
        total_val_metrics_1 = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        total_val_loss_2 = list()
        total_val_metrics_2 = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        total_val_loss_3 = list()
        total_val_metrics_3 = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        
        for epoch in range(epochs):
            setproctitle(f"S&P500 Epoch {epoch + 1}/{epochs}, Epsilon: {epsilon:.4f}")
            
            try:
                self.model.train()
                train_loss = list()
                y_true = list()
                y_pred = list()

                train_dataset = Dataset(self.train_df_cache, start_date=train_s, end_date=train_e, transform=transform)
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
                
                with tqdm(total=len(self.train_df_cache), desc=f"Training Epoch {epoch + 1}/{epochs}", ncols=100, leave=False) as pbar:
                    for imgs_batch, yield_batch in train_dataloader:
                        imgs_batch = imgs_batch.to(device)
                        yield_batch = yield_batch.to(device)
                                                
                        pre_states = imgs_batch[:, 0, :, :, :]
                        cur_states = imgs_batch[:, 1, :, :, :]
                        next_states = imgs_batch[:, 2, :, :, :]
                        
                        if round(random.uniform(0, 1), 4) < epsilon:
                            pre_actions = self._get_random_action(num_actions, imgs_batch.shape[0]).to(device)
                            cur_actions = self._get_random_action(num_actions, imgs_batch.shape[0]).to(device)
                        else:
                            with torch.no_grad():
                                _, pre_actions = self.model(pre_states)
                                _, cur_actions = self.model(cur_states)
                        
                        cur_rewards = self._get_reward(pre_actions, cur_actions, yield_batch, transaction_penalty)
                        for i in range(imgs_batch.shape[0]):
                            self.memory.push(
                                cur_states[i].squeeze(0),
                                cur_actions[i].unsqueeze(0),
                                cur_rewards[i],
                                next_states[i].squeeze(0),
                                yield_batch[i]
                            )
                        if len(self.memory) < batch_size:
                            pbar.update(imgs_batch.shape[0])
                            continue
                        states, actions, rewards, next_states, y_t = self.memory.get_random_sample(batch_size)
                        states = states.to(device)
                        actions = actions.squeeze(1).to(device)
                        rewards = rewards.to(device)
                        next_states = next_states.to(device)
                        
                        q_values, _ = self.model(states)
                        q_values = torch.sum(q_values * actions, dim=1)
                        
                        with torch.no_grad():
                            next_q_values, _ = self.target_model(next_states)
                            next_q_max = next_q_values.max(dim=1)[0]
                            q_targets = rewards + gamma * next_q_max
                        
                        loss = criterion(q_values, q_targets)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.item())
                        
                        y_p = 1 - actions.argmax(dim=1)
                        z = torch.where(y_t == 0)
                        y_t = torch.where(y_t > 0, 1, -1)
                        y_t[z] = 0
                        y_true.append(y_t)
                        y_pred.append(y_p)
                        
                        if epsilon > epsilon_min:
                            epsilon *= 0.99999
                        
                        pbar.update(imgs_batch.shape[0])

                scheduler.step()
                total_train_loss.append(np.mean(train_loss))
                y_pred = torch.cat(y_pred).to("cpu")
                y_true = torch.cat(y_true).to("cpu")
                acc, prec, rec, f1 = self._get_metrics(y_true, y_pred)
                total_train_metrics["accuracy"].append(acc)
                total_train_metrics["precision"].append(prec)
                total_train_metrics["recall"].append(rec)
                total_train_metrics["f1"].append(f1)
                
                if (epoch + 1) % 50 == 0:
                    self._update_target_model()

                print("=" * 100)
                print(f"Epoch {epoch + 1}/{epochs} | Epsilon: {epsilon:.4f} | Learning Rate: {scheduler.get_last_lr()[0]:.8f}")
                print(f"Train\tLoss: {total_train_loss[-1]:.4f} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | f1: {f1:.4f}")
                
            except Exception as e:
                print(f"Error during training: {e}")
                continue
            if (epoch + 1) % 100:
                continue
            try:
                val_dataset_1 = Dataset(self.train_df_cache, start_date=val_s, end_date=val_e, transform=transform) # train data와 같은 종목의 학습 시기 이후 
                val_dataset_2 = Dataset(self.val_df_cache, start_date=train_s, end_date=train_e, transform=transform) # train data와 다른 종목의 같은 시기
                val_dataset_3 = Dataset(self.val_df_cache, start_date=val_s, end_date=val_e, transform=transform) # train data와 다른 종목의 학습 시기 이후
                
                val_loss_1, val_metrics_1 = self._validate(val_dataset_1, batch_size, gamma, criterion)
                print(f"Valid_1\tLoss: {val_loss_1:.4f} | Accuracy: {val_metrics_1[0]:.4f} | Precision: {val_metrics_1[1]:.4f} | Recall: {val_metrics_1[2]:.4f} | f1: {val_metrics_1[3]:.4f}")
                val_loss_2, val_metrics_2 = self._validate(val_dataset_2, batch_size, gamma, criterion)
                print(f"Valid_2\tLoss: {val_loss_2:.4f} | Accuracy: {val_metrics_2[0]:.4f} | Precision: {val_metrics_2[1]:.4f} | Recall: {val_metrics_2[2]:.4f} | f1: {val_metrics_2[3]:.4f}")
                val_loss_3, val_metrics_3 = self._validate(val_dataset_3, batch_size, gamma, criterion)
                print(f"Valid_3\tLoss: {val_loss_3:.4f} | Accuracy: {val_metrics_3[0]:.4f} | Precision: {val_metrics_3[1]:.4f} | Recall: {val_metrics_3[2]:.4f} | f1: {val_metrics_3[3]:.4f}")
                
                total_val_loss_1.append(val_loss_1)
                total_val_loss_2.append(val_loss_2)
                total_val_loss_3.append(val_loss_3)
                total_val_metrics_1["accuracy"].append(val_metrics_1[0])
                total_val_metrics_1["precision"].append(val_metrics_1[1])
                total_val_metrics_1["recall"].append(val_metrics_1[2])
                total_val_metrics_1["f1"].append(val_metrics_1[3])
                total_val_metrics_2["accuracy"].append(val_metrics_2[0])
                total_val_metrics_2["precision"].append(val_metrics_2[1])
                total_val_metrics_2["recall"].append(val_metrics_2[2])
                total_val_metrics_2["f1"].append(val_metrics_2[3])
                total_val_metrics_3["accuracy"].append(val_metrics_3[0])
                total_val_metrics_3["precision"].append(val_metrics_3[1])
                total_val_metrics_3["recall"].append(val_metrics_3[2])
                total_val_metrics_3["f1"].append(val_metrics_3[3])
                
                print("=" * 100)

                # if (epoch+1) % 100 == 0:
                self._save_results(self.model, 
                                    total_train_loss, total_train_metrics, 
                                    total_val_loss_1, total_val_metrics_1, 
                                    total_val_loss_2, total_val_metrics_2, 
                                    total_val_loss_3, total_val_metrics_3, 
                                    result_dir="results_sl", epoch=epoch+1)
            except Exception as e:
                print(f"Error during validation setup: {e}")
                continue
        print("Training completed.")
        self._save_results(self.model, 
                           total_train_loss, total_train_metrics, 
                           total_val_loss_1, total_val_metrics_1, 
                           total_val_loss_2, total_val_metrics_2, 
                           total_val_loss_3, total_val_metrics_3, 
                           result_dir="results_sl", epoch="final")
        
    def test_distribution(self, start, end, batch_size, epochs, val_test: Literal['val', 'test'] = "val"):
        if val_test == "val":
            dataset = Dataset(self.val_df_cache, start_date=start, end_date=end, transform=transform)
        elif val_test == "test":
            dataset = Dataset(self.test_df_cache, start_date=start, end_date=end, transform=transform)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
        y_true = list()
        y_yield = list()
        y_pred = list()
        q_values = list()
        
        for epoch in range(epochs):
            with tqdm(total=len(dataset), desc=f"Testing {epoch+1}/{epochs}", ncols=100, leave=False) as pbar:
                with torch.no_grad():
                    for imgs_batch, yield_batch in dataloader:
                        imgs_batch = imgs_batch.to(device)
                        yield_batch = yield_batch.to(device)
                        
                        cur_states = imgs_batch[:, 1, :, :, :]
                        q, cur_actions = self.test_model(cur_states)
                        y_p = 1 - cur_actions.argmax(dim=1)
                        y_z = torch.where(yield_batch == 0)
                        y_t = torch.where(yield_batch > 0, 1, -1)
                        y_t[y_z] = 0
                        y_true.append(y_t)
                        y_yield.append(yield_batch)
                        y_pred.append(y_p)
                        q_values.append(q)
                        pbar.update(imgs_batch.shape[0])
                    
        y_pred = torch.cat(y_pred).to("cpu")
        y_true = torch.cat(y_true).to("cpu")
        y_yield = torch.cat(y_yield).to("cpu")
        q_values = torch.cat(q_values).to("cpu")
        
        print(f"Test Distribution | Accuracy: {accuracy_score(y_true, y_pred):.4f}" )
        return y_pred, y_true, y_yield, q_values