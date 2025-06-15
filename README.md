# NanoDQN

Reimplementation of CleanRL DQN in JAX+Flax for reinforcement learning research. This implementation focuses on DQN scaling laws and provides bootstrapped ensembles for exploration in deep reinforcement learning.

## Setup

### Option 1: Standard Python Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Option 2: Docker Setup
```bash
# Build the container
docker build -t nanodqn .

# Run with GPU support
docker run -it --gpus all nanodqn

# Run without GPU
docker run -it nanodqn
```

### Option 3: Nix Flake (Full Reproducibility)
```bash
# Enter development shell
nix develop

# Or with direnv (recommended)
direnv allow  # First time only
# Environment automatically activates when entering directory

# Build the package
nix build

# Run the package
nix run
```

## Core Scripts

- **`main.py`** - Main training script implementing DQN with bootstrapped ensembles for exploration. Supports different agent types (boot, bootrp, eps) and handles parallel training across multiple environments. Uses JAX for efficient computation and automatic experiment scheduling.

- **`evaluate.py`** - Analysis and evaluation script for processing experimental results. Creates heatmaps showing convergence probability across different ensemble sizes and environment difficulties. Loads results from experiment folders and generates publication-ready plots.

- **`models.py`** - Neural network architectures and DQN implementations. Includes base Model class, ModelWithPrior for random priors, and Bootstrapped ensemble wrapper. Implements Double DQN and epsilon-greedy exploration strategies.

- **`replay_buffer.py`** - Memory-efficient replay buffer implementation with JAX-compatible operations. Supports observation compression for discrete environments and masking for bootstrapped training. Handles circular buffer logic with JAX's functional programming paradigm.

## Usage

### Basic Usage

```bash
# Run with default configuration
python main.py

# Specify parameters via command line
python main.py env_id=MountainCar-v0 learning_rate=1e-3
```

### Configuration Files

The project uses Hydra for configuration management. Configuration files are stored in the `configs` directory:

- `configs/config.yaml`: Default configuration
- `configs/mountain_car.yaml`: Configuration for Mountain Car environment
- `configs/acrobot.yaml`: Configuration for Acrobot environment

To use a specific configuration:

```bash
python main.py --config-name=mountain_car
```

### Parameter Sweeps

Parameter sweep configurations are available in `configs/sweeps/`:

```bash
# Run learning rate sweep
python main.py --config-name=sweeps/lr_sweep --multirun

# Run environment sweep 
python main.py --config-name=sweeps/env_sweep --multirun

# Run exploration parameters sweep
python main.py --config-name=sweeps/exploration_sweep --multirun
```

## Experiment Tracking

Experiments are tracked with Weights & Biases. The configuration parameters are automatically logged for reproducibility.

To disable tracking:

```bash
python main.py track=false
```