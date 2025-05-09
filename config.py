"""
NEAT configuration for the neural network evolution simulation.
"""

# NEAT configuration
NEAT_CONFIG = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 100
pop_size             = 50
reset_on_extinction  = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_options     = tanh

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob       = 0.2

# node connection options
connection_add_prob    = 0.5
connection_delete_prob = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation      = 20

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""

# Sim parameters
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Agent parameters
AGENT_RADIUS = 10
AGENT_SPEED = 3
AGENT_ENERGY = 100
ENERGY_DECAY = 0.1

# Food parameters
FOOD_RADIUS = 5
FOOD_ENERGY = 50
FOOD_COUNT = 20

# Neural network inputs
NUM_INPUTS = 8  # Distance to nearest food, angle to food, current energy, etc.
NUM_OUTPUTS = 2  # Movement direction and speed