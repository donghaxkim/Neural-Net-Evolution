import pygame
import numpy as np
import matplotlib.pyplot as plt
from config import *

class Visualizer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 36)
        self.stats_font = pygame.font.Font(None, 24)
        self.generation = 0
        self.best_fitness = 0
        self.avg_fitness = 0
        self.fitness_history = []
        self.avg_fitness_history = []

    def draw_stats(self, agents):
        """Draw simulation statistics"""
        # Calculate statistics
        alive_count = sum(1 for agent in agents if agent.alive)
        self.best_fitness = max(agent.fitness for agent in agents)
        self.avg_fitness = sum(agent.fitness for agent in agents) / len(agents)
        
        # Update history
        self.fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(self.avg_fitness)
        
        # Draw text
        texts = [
            f"Generation: {self.generation}",
            f"Alive: {alive_count}/{len(agents)}",
            f"Best Fitness: {self.best_fitness:.1f}",
            f"Avg Fitness: {self.avg_fitness:.1f}"
        ]
        
        for i, text in enumerate(texts):
            surface = self.stats_font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, 10 + i * 25))

    def draw_neural_network(self, agent, x, y, width, height):
        """Draw a simplified visualization of the agent's neural network"""
        if not agent.alive:
            return

        # Draw nodes
        node_radius = 5
        layer_sizes = [NUM_INPUTS, 4, NUM_OUTPUTS]  # Simplified network structure
        layer_positions = []
        
        for i, size in enumerate(layer_sizes):
            layer_x = x + (i * width) / (len(layer_sizes) - 1)
            layer_positions.append([])
            
            for j in range(size):
                node_y = y + (j * height) / (size - 1) if size > 1 else y + height/2
                layer_positions[i].append((layer_x, node_y))
                pygame.draw.circle(self.screen, (200, 200, 200), 
                                 (int(layer_x), int(node_y)), node_radius)

        # Draw connections
        for i in range(len(layer_sizes) - 1):
            for j in range(layer_sizes[i]):
                for k in range(layer_sizes[i + 1]):
                    start = layer_positions[i][j]
                    end = layer_positions[i + 1][k]
                    pygame.draw.line(self.screen, (100, 100, 100), 
                                   (int(start[0]), int(start[1])),
                                   (int(end[0]), int(end[1])), 1)

    def plot_fitness_history(self):
        """Plot fitness history using matplotlib"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, label='Best Fitness')
        plt.plot(self.avg_fitness_history, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig('fitness_history.png')
        plt.close()

    def increment_generation(self):
        """Increment generation counter"""
        self.generation += 1 