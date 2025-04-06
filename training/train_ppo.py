import argparse
import yaml
import gym
import numpy as np
import torch
from agents.ppo_agent import PPOAgent
from utils.logger import Logger
from environments import monopoly_env, catan_env

def create_env(env_name):
    if env_name.lower() == "monopoly":
        return monopoly_env.MonopolyEnv()
    elif env_name.lower() == "catan":
        return catan_env.CatanEnv()
    else:
        raise ValueError("Unknown environment")

def collect_trajectories(env, agent, timesteps):
    trajectories = {"states": [], "actions": [], "logprobs": [], "rewards": [], "dones": []}
    state = env.reset()
    for _ in range(timesteps):
        action, logprob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        trajectories["states"].append(state)
        trajectories["actions"].append(action)
        trajectories["logprobs"].append(logprob)
        trajectories["rewards"].append(reward)
        trajectories["dones"].append(done)
        
        state = next_state
        if done:
            state = env.reset()
    return trajectories

def compute_returns_and_advantages(trajectories, agent, config):
    # A simple implementation; in practice you would use GAE or other techniques.
    gamma = config.get("gamma", 0.99)
    rewards = trajectories["rewards"]
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    # Here, advantages can be computed as returns - baseline (value estimates)
    # For simplicity, we use returns as advantages.
    trajectories["returns"] = returns
    trajectories["advantages"] = returns
    return trajectories

def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    env = create_env(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim, config["ppo"])
    logger = Logger()
    
    num_updates = config["ppo"].get("num_updates", 1000)
    timesteps_per_update = config["ppo"].get("timesteps_per_update", 200)
    
    for update in range(num_updates):
        trajectories = collect_trajectories(env, agent, timesteps_per_update)
        trajectories = compute_returns_and_advantages(trajectories, agent, config["ppo"])
        agent.update(trajectories)
        
        logger.log(f"Update {update} complete.")
    
    torch.save(agent.policy.state_dict(), f"ppo_{args.env}_model.pth")
    logger.log("PPO Training complete and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="catan", help="Environment to train on (monopoly or catan)")
    parser.add_argument("--config", type=str, default="experiments/config_ppo.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    main(args)
