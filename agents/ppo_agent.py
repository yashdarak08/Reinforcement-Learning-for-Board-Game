import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)

class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.eps_clip = config.get("eps_clip", 0.2)
        self.lr = config.get("learning_rate", 3e-4)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.policy(state_tensor)
        action = np.random.choice(self.action_dim, p=probs.cpu().numpy().flatten())
        return action, probs[:, action].item()
    
    def evaluate(self, states, actions):
        probs, state_values = self.policy(states)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_values), dist_entropy
    
    def update(self, trajectories):
        # trajectories: a dictionary with keys:
        # "states", "actions", "logprobs", "returns", "advantages"
        states = torch.FloatTensor(trajectories["states"]).to(self.device)
        actions = torch.LongTensor(trajectories["actions"]).to(self.device)
        old_logprobs = torch.FloatTensor(trajectories["logprobs"]).to(self.device)
        returns = torch.FloatTensor(trajectories["returns"]).to(self.device)
        advantages = torch.FloatTensor(trajectories["advantages"]).to(self.device)
        
        # Evaluate current policy
        logprobs, state_values, dist_entropy = self.evaluate(states, actions)
        
        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(logprobs - old_logprobs)
        
        # Surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        
        loss = -torch.min(surr1, surr2).mean() + self.value_coef * nn.MSELoss()(state_values, returns) - self.entropy_coef * dist_entropy.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()