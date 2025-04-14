import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

class Logger:
    def __init__(self, log_dir="logs"):
        """
        Initialize the logger with a log directory.
        
        Args:
            log_dir (str): Directory to save logs and plots
        """
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # Initialize metrics storage
        self.rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # Print and log initialization message
        init_message = f"Logger initialized at {datetime.datetime.now()}"
        print(init_message)
        self._write_to_file(init_message)
    
    def log(self, message):
        """Log a message to console and file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self._write_to_file(log_message)
    
    def _write_to_file(self, message):
        """Write a message to the log file"""
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
    
    def log_metrics(self, episode=None, reward=None, steps=None, loss=None):
        """
        Log training metrics for plotting
        
        Args:
            episode (int): Episode number
            reward (float): Episode reward
            steps (int): Episode steps/length
            loss (float): Training loss
        """
        if reward is not None:
            self.rewards.append(reward)
        
        if steps is not None:
            self.episode_lengths.append(steps)
        
        if loss is not None:
            self.losses.append(loss)
        
        # Log the metrics
        metrics_msg = f"Episode {episode}: "
        if reward is not None:
            metrics_msg += f"Reward={reward:.2f} "
        if steps is not None:
            metrics_msg += f"Steps={steps} "
        if loss is not None:
            metrics_msg += f"Loss={loss:.4f}"
            
        self.log(metrics_msg)
        
    def plot_rewards(self, window_size=10):
        """Plot the rewards over time with a moving average"""
        if not self.rewards:
            self.log("No rewards to plot yet")
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards, alpha=0.6, label='Rewards')
        
        # Calculate and plot moving average if we have enough data
        if len(self.rewards) >= window_size:
            moving_avg = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.rewards)), moving_avg, label=f'{window_size}-episode Moving Avg')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(self.log_dir, 'rewards_plot.png'))
        plt.close()
        
        self.log(f"Rewards plot saved to {self.log_dir}/rewards_plot.png")