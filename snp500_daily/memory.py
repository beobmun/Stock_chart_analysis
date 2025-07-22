import torch
import numpy as np
import random
from collections import deque

class Memory:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        
    def __len__(self):
        return len(self.buffer)
        
    def push(self, current_state, current_action, current_reward, next_state, y):
        self.buffer.append((current_state, current_action, current_reward, next_state, y))
        
    def get_random_sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, y = zip(*samples)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(y, dtype=torch.float32)
        )