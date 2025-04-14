# Reinforcement Learning for Board Games

This repository implements reinforcement learning (RL) agents for strategic board games including simplified versions of Catan and Monopoly. The project demonstrates:

- **DQN (Deep Q-Network)**: Value-based method for decision making
- **PPO (Proximal Policy Optimization)**: Policy gradient method for learning game strategies

## Overview

The agents are trained in simulated environments that capture key aspects of board games, using reward shaping to encourage intelligent play. This project serves as a practical demonstration of applying RL techniques to complex game scenarios.

## Features

- Custom environments with Gym-compatible interfaces
- Flexible agent implementations (DQN and PPO)
- Reward shaping for strategic play
- Comprehensive logging and visualization
- Evaluation tools to measure agent performance

## Repository Structure

```
reinforcement-learning-board-games/
├── README.md
├── main.py              # Main entry point for running experiments
├── requirements.txt     # Dependencies
├── environments/        # Custom board game environments
│   ├── __init__.py
│   ├── catan_env.py     # Catan-like environment
│   └── monopoly_env.py  # Monopoly-like environment
├── agents/              # RL algorithm implementations
│   ├── __init__.py
│   ├── dqn_agent.py     # Deep Q-Network agent
│   └── ppo_agent.py     # Proximal Policy Optimization agent
├── training/            # Training scripts
│   ├── train_dqn.py     # DQN training loop
│   └── train_ppo.py     # PPO training loop
├── experiments/         # Configuration files
│   ├── config_dqn.yaml  # DQN hyperparameters
│   └── config_ppo.yaml  # PPO hyperparameters
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── replay_buffer.py # Experience replay for DQN
│   ├── logger.py        # Logging and visualization
│   └── reward_shaping.py # Reward modifications
└── models/              # Saved model weights (created during training)
    ├── dqn/
    └── ppo/
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/reinforcement-learning-board-games.git
   cd reinforcement-learning-board-games
   ```

2. **Create a Virtual Environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

Train a DQN agent on the Monopoly environment:
```bash
python main.py --mode train --algorithm dqn --env monopoly
```

Train a PPO agent on the Catan environment:
```bash
python main.py --mode train --algorithm ppo --env catan
```

Use a custom configuration:
```bash
python main.py --mode train --algorithm dqn --env monopoly --config path/to/your/config.yaml
```

### Evaluating a Model

Evaluate a trained model:
```bash
python main.py --mode evaluate --algorithm dqn --env monopoly --model models/dqn/dqn_monopoly_model_final.pth --episodes 20
```

### Interactive Play

Watch the agent play in a step-by-step interactive mode:
```bash
python main.py --mode play --algorithm ppo --env catan --model models/ppo/ppo_catan_model_final.pth
```

## Environments

### Catan

A simplified version of the Settlers of Catan board game, where:
- Players collect resources (wood, brick, sheep, wheat, ore)
- Resources are used to build roads, settlements, and cities
- Victory points are earned through buildings and development cards
- First player to reach 10 victory points wins

### Monopoly

A simplified version of the Monopoly board game, where:
- Players move around the board based on dice rolls
- Properties can be purchased and developed
- Rent is collected when other players land on your properties
- The goal is to avoid bankruptcy and accumulate wealth

## Customization

You can modify the environment and agent parameters by:
1. Editing the YAML configuration files in the `experiments/` directory
2. Creating custom reward shaping functions in `utils/reward_shaping.py`
3. Extending the environments with additional game mechanics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.