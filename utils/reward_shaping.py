import numpy as np

class RewardShaper:
    """
    A configurable reward shaping class for different board game environments.
    
    For Monopoly:
      - Rewards for cash improvements, property acquisition, and strategic positions
    
    For Catan:
      - Rewards for resource collection, building structures, and progress toward victory
    """
    
    def __init__(self, env_type, debug=False):
        self.env_type = env_type.lower()
        self.debug = debug
        
        # Previous state cache for tracking progress (useful for some metrics)
        self.previous_states = []
        self.max_history = 5  # Keep track of last 5 states

    def shape_reward(self, state, action, reward, next_state):
        """
        Compute a shaped reward based on the transition from state to next_state.
        
        Parameters:
          state (np.ndarray): The current state vector.
          action (int): The action taken.
          reward (float): The original environment reward.
          next_state (np.ndarray): The state after taking the action.
          
        Returns:
          float: The modified (shaped) reward.
        """
        shaped_reward = reward
        details = {"original_reward": reward}

        if self.env_type == "monopoly":
            # Calculate improvements
            cash_improvement = next_state[1] - state[1]
            
            # Calculate property acquisition (counting newly owned properties)
            property_before = np.sum(state[2:10])
            property_after = np.sum(next_state[2:10])
            property_gain = property_after - property_before
            
            # Handle board position (track completing a lap)
            position_before = state[0]
            position_after = next_state[0]
            passed_go = position_after < position_before and position_after != 10  # Not jail
            
            # Penalties and bonuses
            cash_bonus = 0.05 * cash_improvement if cash_improvement > 0 else 0.02 * cash_improvement
            property_bonus = 1.0 * property_gain if property_gain > 0 else 0
            position_bonus = 0.5 if passed_go else 0
            
            # Calculate total bonus
            bonus = cash_bonus + property_bonus + position_bonus
            shaped_reward += bonus
            
            details.update({
                'cash_improvement': cash_improvement,
                'property_gain': property_gain,
                'passed_go': passed_go,
                'bonus': bonus
            })

        elif self.env_type == "catan":
            # Calculate resource improvements
            resource_before = np.sum(state[:5])
            resource_after = np.sum(next_state[:5])
            resource_gain = resource_after - resource_before
            
            # Calculate building improvements
            settlements_diff = next_state[5] - state[5]
            cities_diff = next_state[6] - state[6]
            roads_diff = next_state[7] - state[7]
            
            # Calculate victory point progress
            vp_gain = next_state[9] - state[9]
            
            # Apply bonuses
            resource_bonus = 0.05 * resource_gain
            building_bonus = 0.2 * (settlements_diff + 2 * cities_diff + 0.1 * roads_diff)
            vp_bonus = 1.0 * vp_gain if vp_gain > 0 else 0
            
            # Total bonus
            bonus = resource_bonus + building_bonus + vp_bonus
            shaped_reward += bonus
            
            details.update({
                'resource_gain': resource_gain,
                'settlements_diff': settlements_diff,
                'cities_diff': cities_diff,
                'vp_gain': vp_gain,
                'bonus': bonus
            })
        else:
            # Default shaping for unknown environments
            bonus = 0.1 if reward > 0 else 0
            shaped_reward += bonus
            details['bonus'] = bonus

        # Store state in history
        self.previous_states.append(state.copy())
        if len(self.previous_states) > self.max_history:
            self.previous_states.pop(0)
            
        if self.debug:
            print("Reward Shaping Debug Info:", details)
        
        return shaped_reward

def shape_reward(state, action, reward, next_state, env_type="default", debug=False):
    """
    A functional interface for reward shaping.
    
    Parameters:
      state (np.ndarray): The current state vector.
      action (int): The action taken.
      reward (float): The original reward from the environment.
      next_state (np.ndarray): The state after the action.
      env_type (str): The type of board game ("monopoly", "catan", or "default").
      debug (bool): If True, prints debug information.
    
    Returns:
      float: The shaped reward.
    """
    shaper = RewardShaper(env_type, debug)
    return shaper.shape_reward(state, action, reward, next_state)