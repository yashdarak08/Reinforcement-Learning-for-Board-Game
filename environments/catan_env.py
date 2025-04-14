import gym
from gym import spaces
import numpy as np

class CatanEnv(gym.Env):
    """
    Simplified environment for a Catan-like board game.
    The state represents various resources and game elements.
    """
    def __init__(self):
        super(CatanEnv, self).__init__()
        
        # State definition
        # Indices 0-4: Resources (wood, brick, sheep, wheat, ore)
        # Index 5: Settlement count
        # Index 6: City count
        # Index 7: Road count
        # Index 8: Development cards
        # Index 9: Victory points
        self.observation_space = spaces.Box(low=0, high=20, shape=(10,), dtype=np.float32)
        
        # Actions definition
        # 0: Build road
        # 1: Build settlement
        # 2: Build city
        # 3: Buy development card
        # 4: Trade resources
        self.action_space = spaces.Discrete(5)
        
        self.state = self.reset()
        self.max_steps = 50
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        
        # Initialize with some basic resources (1-3 of each)
        resources = np.random.randint(1, 4, size=(5,)).astype(np.float32)
        
        # Start with 0 buildings and 2 victory points (initial settlements)
        self.state = np.zeros(10, dtype=np.float32)
        self.state[:5] = resources
        self.state[5] = 2  # Start with 2 settlements
        self.state[7] = 2  # Start with 2 roads
        self.state[9] = 2  # Start with 2 victory points
        
        return self.state.copy()

    def step(self, action):
        self.current_step += 1
        
        old_state = self.state.copy()
        resources = self.state[:5]
        
        reward = 0
        done = False
        info = {}
        
        # Process actions
        if action == 0:  # Build road
            if resources[0] >= 1 and resources[1] >= 1:  # Wood and Brick
                self.state[0] -= 1  # Wood
                self.state[1] -= 1  # Brick
                self.state[7] += 1  # Increment road count
                reward = 0.1  # Small reward for building infrastructure
            else:
                reward = -0.05  # Penalty for invalid action
                
        elif action == 1:  # Build settlement
            # Check resource requirements (Wood, Brick, Sheep, Wheat)
            if (resources[0] >= 1 and resources[1] >= 1 and 
                resources[2] >= 1 and resources[3] >= 1):
                self.state[0] -= 1  # Wood
                self.state[1] -= 1  # Brick
                self.state[2] -= 1  # Sheep
                self.state[3] -= 1  # Wheat
                self.state[5] += 1  # Increment settlement count
                self.state[9] += 1  # Add victory point
                reward = 1.0  # Significant reward for settlement
            else:
                reward = -0.05  # Penalty for invalid action
                
        elif action == 2:  # Build city
            # Check resource requirements (Wheat x2, Ore x3)
            if resources[3] >= 2 and resources[4] >= 3:
                self.state[3] -= 2  # Wheat
                self.state[4] -= 3  # Ore
                self.state[5] -= 1  # Decrease settlement count
                self.state[6] += 1  # Increment city count
                self.state[9] += 1  # Add victory point
                reward = 2.0  # Large reward for city
            else:
                reward = -0.05  # Penalty for invalid action
                
        elif action == 3:  # Buy development card
            # Check resource requirements (Sheep, Wheat, Ore)
            if resources[2] >= 1 and resources[3] >= 1 and resources[4] >= 1:
                self.state[2] -= 1  # Sheep
                self.state[3] -= 1  # Wheat
                self.state[4] -= 1  # Ore
                self.state[8] += 1  # Increment development card count
                
                # 20% chance the development card gives a victory point
                if np.random.random() < 0.2:
                    self.state[9] += 1
                    reward = 1.0
                else:
                    reward = 0.5  # Moderate reward for development card
            else:
                reward = -0.05  # Penalty for invalid action
                
        elif action == 4:  # Trade resources
            # Simulate trading - lose one random resource, gain another
            resources_available = np.where(resources >= 1)[0]
            if len(resources_available) > 0:
                resource_to_lose = np.random.choice(resources_available)
                resource_to_gain = np.random.randint(0, 5)
                
                self.state[resource_to_lose] -= 1
                self.state[resource_to_gain] += 1
                reward = 0.2  # Small reward for successful trade
            else:
                reward = -0.05  # Penalty for invalid action
        
        # Add dice roll to simulate resource collection each turn
        dice_roll = np.random.randint(2, 13)  # 2d6
        if dice_roll in [6, 8]:  # Common rolls produce more resources
            resource_index = np.random.randint(0, 5)
            self.state[resource_index] += 1 + self.state[6]  # Cities produce more
        elif dice_roll in [2, 12]:  # Rare rolls
            pass  # No resources or robber
        else:
            if np.random.random() < 0.5:  # 50% chance of resource
                resource_index = np.random.randint(0, 5)
                self.state[resource_index] += 1
        
        # Ensure state variables don't exceed limits
        self.state = np.clip(self.state, 0, 20)
        
        # Check win condition
        if self.state[9] >= 10:  # 10 victory points to win
            reward = 10.0  # Big reward for winning
            done = True
            info["won"] = True
        
        # Check termination
        if self.current_step >= self.max_steps:
            done = True
            info["timeout"] = True
        
        return self.state.copy(), reward, done, info

    def render(self, mode="human"):
        print("Current State:")
        resources = ["Wood", "Brick", "Sheep", "Wheat", "Ore"]
        for i, resource in enumerate(resources):
            print(f"  {resource}: {self.state[i]}")
            
        print(f"  Settlements: {self.state[5]}")
        print(f"  Cities: {self.state[6]}")
        print(f"  Roads: {self.state[7]}")
        print(f"  Development Cards: {self.state[8]}")
        print(f"  Victory Points: {self.state[9]}")
        print(f"Step: {self.current_step}/{self.max_steps}")