import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from training.train_dqn import main as train_dqn
from training.train_ppo import main as train_ppo
from environments import CatanEnv, MonopolyEnv
from agents.dqn_agent import DQNAgent, DQN
from agents.ppo_agent import PPOAgent, ActorCritic
from utils.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser(description="Reinforcement Learning for Board Games")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'play'],
                        help='Mode to run: train, evaluate or play')
    parser.add_argument('--algorithm', type=str, required=True, choices=['dqn', 'ppo'],
                        help='RL algorithm to use: DQN or PPO')
    parser.add_argument('--env', type=str, required=True, choices=['catan', 'monopoly'],
                        help='Environment to use: catan or monopoly')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file for training')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file for evaluation or play')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    return parser.parse_args()

def create_env(env_name):
    """Create environment based on name"""
    if env_name.lower() == 'catan':
        return CatanEnv()
    elif env_name.lower() == 'monopoly':
        return MonopolyEnv()
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def load_agent(env, algorithm, model_path):
    """Load a trained agent"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if algorithm.lower() == 'dqn':
        # Create DQN agent
        agent = DQNAgent(state_dim, action_dim, {})
        # Load model weights
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        agent.epsilon = 0.05  # Low exploration for evaluation
        
    elif algorithm.lower() == 'ppo':
        # Create PPO agent
        agent = PPOAgent(state_dim, action_dim, {})
        # Load model weights
        agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    return agent

def evaluate(env, agent, num_episodes=10, algorithm='dqn'):
    """Evaluate a trained agent"""
    logger = Logger(log_dir=f"logs/eval_{algorithm}")
    rewards = []
    steps = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0
        
        while not done:
            # Select action based on algorithm
            if algorithm.lower() == 'dqn':
                action = agent.select_action(state)
            else:  # PPO
                action, _ = agent.select_action(state)
                
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            episode_steps += 1
            
            # Optional rendering
            if episode == 0:  # Render first episode for visualization
                env.render()
        
        rewards.append(total_reward)
        steps.append(episode_steps)
        logger.log_metrics(episode=episode, reward=total_reward, steps=episode_steps)
    
    # Print evaluation summary
    logger.log(f"Evaluation complete. Average reward: {np.mean(rewards):.2f}")
    logger.log(f"Average episode length: {np.mean(steps):.2f}")
    logger.plot_rewards()
    
    return np.mean(rewards)

def create_directory_structure():
    """Create necessary directories if they don't exist"""
    directories = [
        'logs', 
        'logs/dqn', 
        'logs/ppo', 
        'models',
        'models/dqn',
        'models/ppo'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main function to run the project"""
    args = parse_args()
    create_directory_structure()
    
    if args.mode == 'train':
        # Set default config paths if not provided
        if args.config is None:
            if args.algorithm.lower() == 'dqn':
                args.config = 'experiments/config_dqn.yaml'
            else:
                args.config = 'experiments/config_ppo.yaml'
        
        # Run training
        if args.algorithm.lower() == 'dqn':
            train_args = argparse.Namespace(env=args.env, config=args.config)
            train_dqn(train_args)
        else:
            train_args = argparse.Namespace(env=args.env, config=args.config)
            train_ppo(train_args)
            
    elif args.mode == 'evaluate':
        if args.model is None:
            raise ValueError("Model path must be provided for evaluation")
            
        env = create_env(args.env)
        agent = load_agent(env, args.algorithm, args.model)
        evaluate(env, agent, args.episodes, args.algorithm)
        
    elif args.mode == 'play':
        # Interactive play mode (simplified for now)
        if args.model is None:
            raise ValueError("Model path must be provided for play mode")
            
        env = create_env(args.env)
        agent = load_agent(env, args.algorithm, args.model)
        
        state = env.reset()
        done = False
        total_reward = 0
        
        print("Starting game. Press Ctrl+C to exit.")
        env.render()
        
        try:
            while not done:
                # Get agent's action
                if args.algorithm.lower() == 'dqn':
                    action = agent.select_action(state)
                else:  # PPO
                    action, _ = agent.select_action(state)
                    
                print(f"\nAgent selected action: {action}")
                input("Press Enter to continue...")
                
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
                
                print(f"\nReward: {reward}, Total Reward: {total_reward}")
                env.render()
                
            print(f"\nGame over! Total reward: {total_reward}")
            
        except KeyboardInterrupt:
            print("\nGame manually interrupted.")

if __name__ == "__main__":
    main()