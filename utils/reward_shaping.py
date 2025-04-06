import numpy as np

class RewardShaper:
    """
    A configurable reward shaping class tailored for different board game environments.
    
    For Monopoly:
      - Assumes the state vector layout:
          index 0: Board position (assumed cyclic; bonus for passing "Go")
          index 1: Cash amount
          indices 2 to N: Property values or ownership indicators
      - The shaping encourages improvements in cash, acquiring properties, and favorable board positions.
    
    For Catan:
      - Assumes the state vector layout:
          indices 0-4: Resource counts (e.g., wood, brick, sheep, wheat, ore)
          index 5: Progress metric (could be development cards, roads built, etc.)
          remaining indices: Other strategic factors
      - The shaping rewards increased resource availability and progress.
    
    The class also supports a debug mode to print detailed shaping information.
    """
    
    def __init__(self, env_type, debug=False):
        self.env_type = env_type.lower()
        self.debug = debug

    def shape_reward(self, state, action, reward, next_state):
        """
        Compute a shaped reward based on the transition from state to next_state.
        
        Parameters:
          state (np.ndarray): The current state vector.
          action (int): The action taken (not directly used in this heuristic, but available for extensions).
          reward (float): The original environment reward.
          next_state (np.ndarray): The state after taking the action.
          
        Returns:
          float: The modified (shaped) reward.
        """
        shaped_reward = reward
        details = {}

        if self.env_type == "monopoly":
            # Heuristic assumptions for Monopoly:
            # - Board position: If the agent passes a certain point (e.g., “Go”), it may get a bonus.
            # - Cash improvement: Increase in cash yields a small bonus.
            # - Property acquisition: Increase in property-related features yields a significant bonus.
            cash_improvement = next_state[1] - state[1]
            # Assume board positions are cyclic with a wrap-around at 100.
            if next_state[0] >= state[0]:
                position_improvement = next_state[0] - state[0]
            else:
                position_improvement = (next_state[0] + 100) - state[0]
            # Count how many property indicators (indices 2 onward) have increased.
            property_gain = sum(1 for i in range(2, len(state)) if next_state[i] > state[i])
            
            # Calculate bonus with tuned coefficients:
            bonus = (0.01 * cash_improvement) + (0.1 * position_improvement) + (0.5 * property_gain)
            shaped_reward += bonus
            
            details['cash_improvement'] = cash_improvement
            details['position_improvement'] = position_improvement
            details['property_gain'] = property_gain
            details['bonus'] = bonus

        elif self.env_type == "catan":
            # Heuristic assumptions for Catan:
            # - Resource improvement: Increase in the total count of resources (indices 0-4).
            # - Progress improvement: Increase in progress-related metric (index 5).
            resource_improvement = np.sum(next_state[:5]) - np.sum(state[:5])
            progress_improvement = next_state[5] - state[5]
            
            # Calculate bonus with tuned coefficients:
            bonus = (0.1 * resource_improvement) + (0.5 * progress_improvement)
            shaped_reward += bonus
            
            details['resource_improvement'] = resource_improvement
            details['progress_improvement'] = progress_improvement
            details['bonus'] = bonus

        else:
            # Default reward shaping if the environment type is unknown.
            bonus = 0.0
            if next_state[0] > 5:
                bonus += 0.5
            shaped_reward += bonus
            details['bonus'] = bonus

        if self.debug:
            print("Reward Shaping Debug Info:", details)
        
        return shaped_reward

def shape_reward(state, action, reward, next_state, env_type="default", debug=False):
    """
    A functional interface for reward shaping.
    This function wraps around the RewardShaper class to allow one-line usage.
    
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
