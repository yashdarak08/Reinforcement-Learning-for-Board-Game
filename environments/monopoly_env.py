import gym
from gym import spaces
import numpy as np

class MonopolyEnv(gym.Env):
    """
    Simplified environment for a Monopoly-like board game.
    The state may include positions, cash, and properties.
    """
    def __init__(self):
        super(MonopolyEnv, self).__init__()
        # Example state: position (1 value), cash (1 value), properties owned (8 values)
        self.observation_space = spaces.Box(low=0, high=100, shape=(10,), dtype=np.float32)
        # Assume 6 possible actions (buy, sell, trade, etc.)
        self.action_space = spaces.Discrete(6)
        self.state = self.reset()
        self.max_steps = 60
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        # Initialize with random cash, position, etc.
        self.state = np.random.randint(0, 100, size=(10,)).astype(np.float32)
        return self.state

    def step(self, action):
        self.current_step += 1
        
        # Simulate state transition
        self.state = np.random.randint(0, 100, size=(10,)).astype(np.float32)
        
        # Reward could be profit, property gain, etc.
        reward = np.random.rand()
        
        done = self.current_step >= self.max_steps
        info = {}
        return self.state, reward, done, info

    def render(self, mode="human"):
        print("State:", self.state)
        print("Current Step:", self.current_step)
        print("Action Space:", self.action_space)
        print("Observation Space:", self.observation_space)
        