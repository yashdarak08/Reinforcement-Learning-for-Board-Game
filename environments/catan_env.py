import gym
from gym import spaces
import numpy as np

class CatanEnv(gym.Env):
    """
    Simplified environment for a Catan-like board game.
    The state could represent resources, settlements, etc.
    The action space is discrete for simplicity.
    """
    def __init__(self):
        super(CatanEnv, self).__init__()
        # For example, a state vector of 10 features.
        self.observation_space = spaces.Box(low=0, high=10, shape=(10,), dtype=np.float32)
        # Let's assume 5 discrete actions.
        self.action_space = spaces.Discrete(5)
        self.state = self.reset()
        self.max_steps = 50
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.state = np.random.randint(0, 10, size=(10,)).astype(np.float32)
        return self.state

    def step(self, action):
        self.current_step += 1
        
        # Simulate state transition
        self.state = np.random.randint(0, 10, size=(10,)).astype(np.float32)
        
        # Simple reward: reward for a specific action or condition
        reward = 1 if action == np.random.randint(0, self.action_space.n) else 0
        
        # Optional: Apply reward shaping (could also be done via utils.reward_shaping)
        done = self.current_step >= self.max_steps
        info = {}
        return self.state, reward, done, info

    def render(self, mode="human"):
        print("State:", self.state)
        print("Current Step:", self.current_step)
        