# NanoDQN

Reimplementation of CleanRL DQN in JAX+Flax for reinforcement learning research.

## Setup

```bash
# Install dependencies
pip install -e .
```

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