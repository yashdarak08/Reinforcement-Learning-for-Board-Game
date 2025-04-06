# Datasets for Board Games RL Project

Since there are no comprehensive public datasets for board games like Catan and Monopoly,
this repository uses simulated gameplay to generate training data.

## Generating Data
- Run the training scripts (`train_dqn.py` or `train_ppo.py`) to simulate thousands of episodes.
- The agents interact with the custom environments to generate trajectories which can be saved and analyzed.

## Public Alternatives
For more advanced simulations, consider exploring:
- [OpenSpiel](https://github.com/deepmind/open_spiel): A collection of environments and algorithms for board games and other strategic games.
- Other game-specific simulation libraries available in the community.
