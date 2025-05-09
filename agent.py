import numpy as np
import pygame
import neat
from config import *

class Agent:
    def __init__(self, x, y, genome, config):
        self.x = x
        self.y = y
        self.radius = AGENT_RADIUS
        self.energy = AGENT_ENERGY
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)  # Create neural network
        self.fitness = 0
        self.alive = True
        self.angle = 0
        self.speed = 0

    def get_inputs(self, foods):
        """Get neural network inputs based on environment state"""
        inputs = []
        
        # Find nearest food
        nearest_food = None
        min_dist = float('inf')
        for food in foods:
            if food.eaten:
                continue
            dist = np.sqrt((food.x - self.x)**2 + (food.y - self.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_food = food

        if nearest_food:
            # Distance to nearest food (normalized)
            inputs.append(min_dist / np.sqrt(WINDOW_WIDTH**2 + WINDOW_HEIGHT**2))
            
            # Angle to nearest food (normalized)
            dx = nearest_food.x - self.x
            dy = nearest_food.y - self.y
            angle = np.arctan2(dy, dx)
            inputs.append(angle / (2 * np.pi))
            
            # Food position (normalized)
            inputs.append(nearest_food.x / WINDOW_WIDTH)
            inputs.append(nearest_food.y / WINDOW_HEIGHT)
        else:
            inputs.extend([1.0, 0.0, 0.5, 0.5])  # No food found

        # Current energy level (normalized)
        inputs.append(self.energy / AGENT_ENERGY)
        
        # Current speed (normalized)
        inputs.append(self.speed / AGENT_SPEED)
        
        # Current angle (normalized)
        inputs.append(self.angle / (2 * np.pi))
        
        # Distance to walls (normalized)
        inputs.append(min(self.x, WINDOW_WIDTH - self.x) / WINDOW_WIDTH)

        return inputs

    def update(self, foods):
        """Update agent state based on neural network output"""
        if not self.alive:
            return

        # Get neural network inputs
        inputs = self.get_inputs(foods)
        
        # Get neural network output
        output = self.net.activate(inputs)
        
        # Update movement
        self.angle += (output[0] - 0.5) * np.pi  # Rotate
        self.speed = output[1] * AGENT_SPEED  # Set speed
        
        # Update position
        self.x += np.cos(self.angle) * self.speed
        self.y += np.sin(self.angle) * self.speed
        
        # Keep agent within bounds
        self.x = np.clip(self.x, self.radius, WINDOW_WIDTH - self.radius)
        self.y = np.clip(self.y, self.radius, WINDOW_HEIGHT - self.radius)
        
        # Decrease energy
        self.energy -= ENERGY_DECAY
        
        # Check if agent is dead
        if self.energy <= 0:
            self.alive = False

    def draw(self, screen):
        """Draw the agent on the screen"""
        if not self.alive:
            return

        # Draw agent body
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.radius)
        
        # Draw direction indicator
        end_x = self.x + np.cos(self.angle) * self.radius
        end_y = self.y + np.sin(self.angle) * self.radius
        pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), (end_x, end_y), 2)

    def eat(self, food):
        """Consume food and gain energy"""
        self.energy = min(AGENT_ENERGY, self.energy + FOOD_ENERGY)
        self.fitness += 1