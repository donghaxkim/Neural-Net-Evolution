import pygame
import neat
import os
import random
import numpy as np
from agent import Agent
from environment import Environment
from visualization import Visualizer
from config import *

def create_agents(config, pop_size):
    """Create a population of agents"""
    agents = []
    for _ in range(pop_size):
        x = random.randint(AGENT_RADIUS, WINDOW_WIDTH - AGENT_RADIUS)
        y = random.randint(AGENT_RADIUS, WINDOW_HEIGHT - AGENT_RADIUS)
        genome = neat.DefaultGenome(0)  # Create a new genome with ID 0
        genome.configure_new(config.genome_config)  # Configure it with our settings
        agents.append(Agent(x, y, genome, config))
    return agents

def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Neural Network Evolution")
    clock = pygame.time.Clock()

    # Initialize NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config.txt')
    
    # Create initial population
    agents = create_agents(config, config.pop_size)
    environment = Environment()
    visualizer = Visualizer(screen)

    # Main simulation loop
    running = True
    paused = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Reset simulation
                    agents = create_agents(config, config.pop_size)
                    environment.reset()
                    visualizer = Visualizer(screen)
                elif event.key == pygame.K_q:
                    running = False

        if not paused:
            # Update environment and agents
            environment.update(agents)
            for agent in agents:
                agent.update(environment.foods)

            # Check if all agents are dead
            if all(not agent.alive for agent in agents):
                # Create next generation
                visualizer.increment_generation()
                
                # Select best genomes
                genomes = [(agent.genome, agent.fitness) for agent in agents]
                genomes.sort(key=lambda x: x[1], reverse=True)
                
                # Create new population
                new_agents = []
                for i in range(config.pop_size):
                    parent = random.choice(genomes[:config.pop_size//2])[0]
                    child = neat.DefaultGenome(i)  # Create new genome with unique ID
                    child.configure_crossover(parent, parent, config.genome_config)  # Self-crossover
                    child.mutate(config.genome_config)  # Mutate the child
                    x = random.randint(AGENT_RADIUS, WINDOW_WIDTH - AGENT_RADIUS)
                    y = random.randint(AGENT_RADIUS, WINDOW_HEIGHT - AGENT_RADIUS)
                    new_agents.append(Agent(x, y, child, config))
                
                agents = new_agents
                environment.reset()

        # Draw everything
        screen.fill((0, 0, 0))
        environment.draw(screen)
        for agent in agents:
            agent.draw(screen)
        
        # Draw statistics and neural network visualization
        visualizer.draw_stats(agents)
        if agents:
            best_agent = max(agents, key=lambda x: x.fitness)
            visualizer.draw_neural_network(best_agent, 
                                        WINDOW_WIDTH - 200, 50, 150, 100)

        pygame.display.flip()
        clock.tick(FPS)

    # Save fitness history plot
    visualizer.plot_fitness_history()
    pygame.quit()

if __name__ == "__main__":
    main() 