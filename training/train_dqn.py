import argparse
import yaml
import gym
import numpy as np
import torch
from agents.dqn_agent import DQNAgent
from utils.logger import Logger
from environments import monopoly_env, catan_env

def create_env(env_name):
    if env_name.lower() == "monopoly":
        return monopoly_env.MonopolyEnv()
    elif env_name.lower() == "catan":
        return catan_env.CatanEnv()
    else:
        raise ValueError("Unknown environment")

def main(args):
    # Load configuration from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    env = create_env(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, config["dqn"])
    logger = Logger()
    
    num_episodes = config["dqn"].get("num_episodes", 500)
    target_update = config["dqn"].get("target_update", 10)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Optionally, use a reward shaping function from utils.reward_shaping
            # reward = your_reward_shaping_function(state, action, reward)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.update()
        
        if episode % target_update == 0:
            agent.update_target()
        
        logger.log(f"Episode {episode}: Total Reward = {total_reward}")
    
    # Optionally, save the trained model
    torch.save(agent.policy_net.state_dict(), f"dqn_{args.env}_model.pth")
    logger.log("Training complete and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="monopoly", help="Environment to train on (monopoly or catan)")
    parser.add_argument("--config", type=str, default="experiments/config_dqn.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    main(args)
