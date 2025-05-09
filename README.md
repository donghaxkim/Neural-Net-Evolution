# Neural Network Evolution Simulation

This project simulates the evolution of neural networks controlling agents in a competitive environment. Agents controlled by neural networks compete for survival and resources, with the best performers reproducing and passing on their traits with mutations.

## Features

- Neural network-controlled agents
- Evolutionary algorithm for network optimization
- Real-time visualization of agent behavior
- Neural network structure visualization
- Performance tracking over generations

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python main.py
```

## Controls

- Space: Pause/Resume simulation
- R: Reset simulation
- Q: Quit

## Project Structure

- `main.py`: Main simulation loop and visualization
- `agent.py`: Agent class and neural network implementation
- `environment.py`: Environment and food generation
- `visualization.py`: Visualization utilities
- `config.py`: NEAT configuration and simulation parameters 