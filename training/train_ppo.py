import argparse
import yaml
import numpy as np
import torch
from agents.ppo_agent import PPOAgent
from utils.logger import Logger
from utils.reward_shaping import shape_reward
from environments import MonopolyEnv, CatanEnv

def create_env(env_name):
    if env_name.lower() == "monopoly":
        return MonopolyEnv()
    elif env_name.lower() == "catan":
        return CatanEnv()
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def collect_trajectories(env, agent, timesteps, env_type):
    trajectories = {"states": [], "actions": [], "logprobs": [], "rewards": [], "dones": []}
    state = env.reset()
    
    for _ in range(timesteps):
        action, logprob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Apply reward shaping
        shaped_reward = shape_reward(state, action, reward, next_state, env_type=env_type)
        
        trajectories["states"].append(state)
        trajectories["actions"].append(action)
        trajectories["logprobs"].append(logprob)
        trajectories["rewards"].append(shaped_reward)
        trajectories["dones"].append(done)
        
        state = next_state
        if done:
            state = env.reset()
    
    return trajectories

def compute_returns_and_advantages(trajectories, config):
    gamma = config.get("gamma", 0.99)
    rewards = trajectories["rewards"]
    dones = trajectories["dones"]
    
    returns = []
    advantages = []
    
    # Initialize with zero
    G = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1-done)
        returns.insert(0, G)
    
    # Simple advantage estimate (returns - baseline)
    # In a more advanced implementation, you'd use GAE
    # For now, we just use returns as advantages
    advantages = returns.copy()
    
    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    trajectories["returns"] = returns
    trajectories["advantages"] = advantages
    return trajectories

def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    ppo_config = config.get("ppo", {})
    
    env = create_env(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim, ppo_config)
    logger = Logger()
    
    num_updates = ppo_config.get("num_updates", 1000)
    timesteps_per_update = ppo_config.get("timesteps_per_update", 200)
    
    for update in range(num_updates):
        # Collect trajectories using the current policy
        trajectories = collect_trajectories(env, agent, timesteps_per_update, args.env)
        
        # Compute returns and advantages
        trajectories = compute_returns_and_advantages(trajectories, ppo_config)
        
        # Update the policy
        loss = agent.update(trajectories)
        
        # Log progress
        mean_reward = np.mean(trajectories["rewards"])
        logger.log(f"Update {update}: Mean Reward = {mean_reward:.2f}, Loss = {loss:.4f}")
        
        # Save the model periodically
        if update > 0 and update % ppo_config.get("save_interval", 50) == 0:
            torch.save(agent.policy.state_dict(), f"models/ppo_{args.env}_model_update{update}.pth")
    
    # Save the final model
    torch.save(agent.policy.state_dict(), f"models/ppo_{args.env}_model_final.pth")
    logger.log("PPO Training complete and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="catan", help="Environment to train on (monopoly or catan)")
    parser.add_argument("--config", type=str, default="experiments/config_ppo.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    main(args)