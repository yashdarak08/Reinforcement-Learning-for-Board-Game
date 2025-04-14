import gym
from gym import spaces
import numpy as np

class MonopolyEnv(gym.Env):
    """
    Simplified environment for a Monopoly-like board game.
    """
    def __init__(self):
        super(MonopolyEnv, self).__init__()
        
        # State definition
        # Index 0: Position (0-39 for a standard board)
        # Index 1: Cash
        # Indices 2-9: Property ownership (binary flags for 8 property groups)
        self.observation_space = spaces.Box(low=0, high=1500, shape=(10,), dtype=np.float32)
        
        # Actions definition
        # 0: Roll & Move
        # 1: Buy property
        # 2: Skip buying
        # 3: Build house/hotel
        # 4: Mortgage property
        # 5: Trade (simplified)
        self.action_space = spaces.Discrete(6)
        
        # Board properties - simplifying to 40 spaces with 8 property groups
        self.board_size = 40
        self.property_prices = {
            0: 0,    # Go
            1: 60,   # Brown
            2: 0,    # Community Chest
            3: 60,   # Brown
            4: 200,  # Tax
            5: 200,  # Railroad
            # ... and so on (simplified)
        }
        
        self.property_groups = {
            # Mapping from position to group index (2-9 in state vector)
            1: 2,  # Brown
            3: 2,  # Brown
            5: 3,  # Railroad
            6: 4,  # Light Blue
            8: 4,  # Light Blue
            9: 4,  # Light Blue
            # ... and so on (simplified)
        }
        
        self.state = self.reset()
        self.max_steps = 60
        self.current_step = 0
        self.can_buy = False  # Flag to track if player landed on purchasable property

    def reset(self):
        self.current_step = 0
        
        # Start at Go with $1500
        self.state = np.zeros(10, dtype=np.float32)
        self.state[1] = 1500.0  # Starting cash
        
        self.can_buy = False
        return self.state.copy()

    def step(self, action):
        self.current_step += 1
        
        old_state = self.state.copy()
        reward = 0
        done = False
        info = {}
        
        current_position = int(self.state[0])
        
        # Process actions
        if action == 0:  # Roll & Move
            # Roll 2d6
            dice1 = np.random.randint(1, 7)
            dice2 = np.random.randint(1, 7)
            dice_sum = dice1 + dice2
            
            # Move player
            new_position = (current_position + dice_sum) % self.board_size
            self.state[0] = float(new_position)
            
            # Check for passing GO
            if new_position < current_position:
                self.state[1] += 200  # Collect $200
                reward += 1.0
                
            # Check if landed on property
            if new_position in self.property_prices and self.property_prices[new_position] > 0:
                property_group = self.property_groups.get(new_position, -1)
                
                # If property is already owned, pay rent
                if property_group >= 0 and self.state[property_group] > 0:
                    rent = self.property_prices[new_position] * 0.1  # Simplified rent calculation
                    self.state[1] -= rent  # Pay rent
                    reward -= 0.2  # Negative reward for paying rent
                else:
                    self.can_buy = True  # Allow buying in next action
                    
            # Check for special squares (simplified)
            if new_position == 4:  # Income Tax
                tax = min(200, self.state[1] * 0.1)
                self.state[1] -= tax
                reward -= 0.3
                
            if new_position == 30:  # Go to Jail
                self.state[0] = 10  # Jail position
                reward -= 0.5
                
        elif action == 1:  # Buy property
            if self.can_buy:
                property_pos = int(self.state[0])
                if property_pos in self.property_prices:
                    price = self.property_prices[property_pos]
                    
                    if self.state[1] >= price:  # If enough money
                        self.state[1] -= price  # Pay for property
                        
                        # Mark property group as owned
                        property_group = self.property_groups.get(property_pos, -1)
                        if property_group >= 0:
                            self.state[property_group] = 1.0
                            
                        reward += 0.5  # Reward for buying property
                        self.can_buy = False
                    else:
                        reward -= 0.1  # Not enough money
                else:
                    reward -= 0.1  # Not a purchasable property
            else:
                reward -= 0.2  # Penalize invalid purchase attempt
                
        elif action == 2:  # Skip buying
            if self.can_buy:
                self.can_buy = False
                reward -= 0.1  # Small penalty for missed opportunity
            else:
                reward -= 0.05  # No opportunity to skip
                
        elif action == 3:  # Build house/hotel - simplified
            # Count owned property groups
            owned_groups = np.sum(self.state[2:10])
            
            if owned_groups > 0 and self.state[1] >= 100:
                self.state[1] -= 100  # Cost to build
                reward += 0.3  # Reward for development
            else:
                reward -= 0.1  # Cannot build
                
        elif action == 4:  # Mortgage property - simplified
            if np.sum(self.state[2:10]) > 0:  # If any properties owned
                # Select random owned property group
                owned_groups = np.where(self.state[2:10] > 0)[0]
                if len(owned_groups) > 0:
                    group_idx = np.random.choice(owned_groups)
                    self.state[group_idx + 2] = 0  # Mortgage property
                    self.state[1] += 100  # Get cash for mortgage
                    reward += 0.2  # Short-term cash gain but long-term loss
            else:
                reward -= 0.1  # No properties to mortgage
                
        elif action == 5:  # Trade - simplified
            # Random trade outcome
            trade_outcome = np.random.random()
            if trade_outcome < 0.3:  # Good trade
                self.state[1] += 50
                reward += 0.3
            elif trade_outcome < 0.6:  # Bad trade
                self.state[1] -= 50
                reward -= 0.2
            # Otherwise neutral trade
        
        # Ensure cash doesn't go negative
        self.state[1] = max(0, self.state[1])
        
        # Check bankruptcy
        if self.state[1] <= 0:
            done = True
            reward -= 5.0  # Big penalty for bankruptcy
            info["bankrupt"] = True
            
        # Check win condition (simplified - own all property groups)
        if np.all(self.state[2:10] > 0):
            reward += 10.0
            done = True
            info["won"] = True
            
        # Check timeout
        if self.current_step >= self.max_steps:
            done = True
            info["timeout"] = True
            
        return self.state.copy(), reward, done, info

    def render(self, mode="human"):
        print(f"Position: {int(self.state[0])}")
        print(f"Cash: ${self.state[1]:.2f}")
        print("Property Groups Owned:")
        for i in range(2, 10):
            if self.state[i] > 0:
                print(f"  Group {i-1}: Owned")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print("Can Buy Property:" + str(self.can_buy))