import numpy as np
import random
from collections import deque

class Memory:
    def __init__(self, buffer, width, height):
        self.buffer = buffer
        self.width = width
        self.height = height
        
        self.current_state = deque(maxlen=buffer)
        self.current_action = deque(maxlen=buffer)
        self.current_reward = deque(maxlen=buffer)
        self.next_state = deque(maxlen=buffer)
        
    def add(self, current_state, current_action, current_reward, next_state):
        self.current_state.append(current_state)
        self.current_action.append(current_action)
        self.current_reward.append(current_reward)
        self.next_state.append(next_state)
        
    def get_batch(self, ):
        pass