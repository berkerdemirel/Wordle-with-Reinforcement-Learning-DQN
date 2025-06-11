# Wordle RL Agent

This project implements a Reinforcement Learning agent to play the game of Wordle. It uses different RL algorithms like DQN and Q-Learning, and also includes a Random agent for baseline comparison. The project is built using PyTorch Lightning for training and Hydra for configuration management.

## Features

*   Solves Wordle using Reinforcement Learning.
*   Supports multiple agents:
    *   Deep Q-Network (DQN)
    *   Q-Learning (QN)
    *   Random Agent
*   Uses PyTorch Lightning for streamlined training and experiment management.
*   Configuration managed by Hydra, allowing for easy overrides via command line.
*   Tracks and logs metrics like average turns to win, average reward, and loss.
*   Optionally logs to Weights & Biases (WandB) if configured.

## Tools Used

*   Python 3.x
*   PyTorch
*   PyTorch Lightning
*   Hydra
*   NLTK (for word vocabulary)
*   WandB (optional, for logging)

## Project Structure

```
.
├── agent.py            # RL agent implementations (DQN, QN, Random)
├── config.yaml         # Hydra configuration file for all parameters
├── game.py             # Wordle game logic
├── main.py             # Main script to run training/evaluation (deprecated or for non-Lightning use)
├── pl_wordle.py        # PyTorch Lightning module for Wordle RL
├── utils.py            # Utility functions (e.g., optimizer, epsilon schedule, word loading)
├── test_wordle_game.py # Unit tests for the Wordle game logic
├── README.md           # This file
└── outputs/            # Default Hydra output directory for logs and runs
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/berkerdemirel/Wordle-with-Reinforcement-Learning-DQN.git
    cd Wordle-with-Reinforcement-Learning-DQN
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    A `requirements.txt` file is not explicitly provided in the context, but common dependencies would be:
    ```bash
    pip install torch torchvision torchaudio pytorch-lightning hydra-core omegaconf nltk wandb
    ```
    You might need to adjust this based on your specific setup (e.g., CUDA version for PyTorch).

4.  **Download NLTK words corpus (if not already present):**
    The first time you run the code, it will attempt to download the NLTK 'words' corpus if it's not found. You can also do this manually in a Python interpreter:
    ```python
    import nltk
    nltk.download('words')
    ```

## Usage

The primary way to run the project is using main script `main.py`.

**To train/evaluate an agent:**

```bash
python -B main.py
```

This will run with the default configuration specified in `config.yaml` (which defaults to the DQN agent).

**Overriding configurations with Hydra:**

You can easily change parameters from the command line.

*   **Run a different agent (e.g., QN or random):**
    ```bash
    python main.py agent=qn
    python main.py agent=random
    ```

*   **Change training parameters (e.g., number of environment steps, learning rate):**
    ```bash
    python pl_wordle.py train.num_env_steps=100000 optim.lr_initial=0.0001
    ```

*   **Change game parameters (e.g., word length):**
    ```bash
    python pl_wordle.py game.word_length=6
    ```

*   **Multi-run for sweeping parameters (Hydra feature):**
    For example, to run DQN and QN agents:
    ```bash
    python pl_wordle.py -m agent=dqn,qn
    ```
    Results will be saved in `multirun/` inside the Hydra output directory.

## Configuration (`config.yaml`)

The `config.yaml` file contains all the configurable parameters for the game, agents, optimizer, RL-specific settings, and training.

Key sections:

*   `agent`: Specifies the agent type (`dqn`, `qn`, `random`).
*   `seed`: Random seed for reproducibility.
*   `game`: Wordle game settings (`word_length`, `max_turns`).
*   `exploration`: Epsilon-greedy exploration parameters for DQN/QN.
*   `optim`: Optimizer and scheduler settings.
*   `rl`: Reinforcement learning parameters (`gamma`, `n_step` for DQN).
*   `train`: Training loop parameters (`num_env_steps`, `batch_size`, `replay_capacity`, `update_every`, `log_every`).
*   `test`: Parameters for the final testing phase (currently integrated into the main loop logging).

## Logging

*   **Console:** Progress and metrics are printed to the console.
*   **Hydra Logs:** Hydra saves the configuration and a log file (`main.log` or `pl_wordle.log`) for each run in the `outputs/` directory (or `multirun/` for multi-runs).
*   **WandB:** If `WandbLogger` is used (default in `pl_wordle.py` if `wandb` is installed and configured), metrics will be logged to Weights & Biases. You might need to set up your WandB API key (`wandb login`).

## Future Work / Improvements

*   Implement more advanced RL agents (e.g., Actor-Critic).
*   More sophisticated state and action representations.
*   Hyperparameter optimization sweeps using Hydra's capabilities.
*   Detailed analysis of agent performance and learning curves.
*   Add comprehensive unit and integration tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming a standard MIT license was intended).