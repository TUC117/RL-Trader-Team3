# Importing libraries
import os
import random
import pickle
import numpy as np
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

class FFN(nn.Module):
    def __init__(self, state_dim, hidden_size=64, action_dim=3):
        super(FFN, self).__init__()
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim

        self.layer1 = nn.Linear(self.state_dim, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, (self.hidden_size)//2)
        self.layer3 = nn.Linear((self.hidden_size)//2, (self.hidden_size)//8)
        self.final_layer = nn.Linear((self.hidden_size)//8, self.action_dim)

        self.loss_function = None
        self.optmizer = None

    def forward(self, x):
        state_dim = torch.FloatTensor(x)
        l1 = torch.relu(self.layer1(state_dim))
        l1 = torch.relu(self.layer2(l1))
        l1 = torch.relu(self.layer3(l1))
        action = torch.softmax(self.final_layer(l1), dim=-1) # chat gpt gav  dim=-1
        return action
    

class DQNAgent(nn.Module):
    """
        Our agent should have init, a FF nn
    """
    def __init__(self, state_dim, balance, model_name="dqn", action_dim=3, buffer_size=60,
                gamma=0.95, epsilon=1.0, eps_min=0.01, eps_decay=0.995, is_test=False, hidden_size = 64
                ):
        super(DQNAgent, self).__init__() # Calling parent class
        self.model_type = "dqn"
        self.state_dim = state_dim
        self.balance = balance
        self.action_dim = action_dim # buy, hold, sell
        self.memory = deque(maxlen=1000)
        self.buffer_size = buffer_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.is_test = is_test
        if is_test:
            folder_path = "/mnt/d/ML/SoC_RL/Stock/RL-Trader-Team3/saved_models"
            file_path = os.path.join(folder_path, "dqn.pkl")
            with open(file_path, "rb") as f:
                self.model = pickle.load(f)

        else:
            self.model = self._build_model()
    
    def reset(self):
        self.__init__()
        self.epsilon = 1.0

    def _build_model(self):
        model = FFN(state_dim=self.state_dim, hidden_size=self.hidden_size, action_dim=self.action_dim)
        model.loss_function = nn.MSELoss()
        model.optmizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return model
    
    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))
    
    def act(self, state):
        if not self.is_test and np.random.randn() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state)
        possible_actions = self.model.forward(state)
        return torch.argmax(possible_actions)
    
    def replay_buffer(self):
        if len(self.memory) < self.buffer_size:
            return None
        
        mini_batch = random.sample(self.memory, self.buffer_size)
        total_loss = 0

        for state, actions, reward, next_state, done in mini_batch:
            
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            actions = torch.FloatTensor([actions])
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])

            q_val = reward + (1-done)* self.gamma * torch.max(self.model.forward(next_state)).item()

            next_actions = self.model.forward(state)
            action_index = torch.argmax(actions).item()
            # print(action_index)
            target = next_actions.clone()
            target[action_index] = q_val

            loss = self.model.loss_function(next_actions, target)
            total_loss += loss.item()

            self.model.optmizer.zero_grad()
            loss.backward()
            self.model.optmizer.step()

        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

        return total_loss / len(mini_batch)
        