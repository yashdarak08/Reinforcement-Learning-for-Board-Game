import argparse
import yaml
import numpy as np
import torch
from agents.dqn_agent import DQNAgent
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

def main(args):
    # Load configuration from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    dqn_config = config.get("dqn", {})
    
    env = create_env(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, dqn_config)
    logger = Logger()
    
    num_episodes = dqn_config.get("num_episodes", 500)
    target_update = dqn_config.get("target_update", 10)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        max_steps_per_episode = dqn_config.get("max_steps_per_episode", 200)
        
        while not done and step < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Apply reward shaping
            shaped_reward = shape_reward(state, action, reward, next_state, env_type=args.env)
            
            agent.replay_buffer.push(state, action, shaped_reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            
            agent.update()
        
        if episode % target_update == 0:
            agent.update_target()
        
        logger.log(f"Episode {episode}: Total Reward = {total_reward:.2f}, Steps = {step}")
        
        # Save model periodically
        if episode > 0 and episode % dqn_config.get("save_interval", 50) == 0:
            torch.save(agent.policy_net.state_dict(), f"models/dqn_{args.env}_model_ep{episode}.pth")
    
    # Save the final trained model
    torch.save(agent.policy_net.state_dict(), f"models/dqn_{args.env}_model_final.pth")
    logger.log("Training complete and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="monopoly", help="Environment to train on (monopoly or catan)")
    parser.add_argument("--config", type=str, default="experiments/config_dqn.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    main(args)