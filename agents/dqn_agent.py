import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.replay_buffer import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.batch_size = config.get("batch_size", 64)
        self.lr = config.get("learning_rate", 1e-3)
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        self.replay_buffer = ReplayBuffer(capacity=config.get("buffer_size", 10000))
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        q_values = self.policy_net(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
        
        expected_q = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())