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
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.fitness = 0
        self.alive = True
        self.angle = 0
        self.speed = 0
        self.memory = np.zeros((MEMORY_STEPS, MEMORY_SIZE))  # Memory buffer
        self.vision_angle = VISION_ANGLE
        self.vision_range = VISION_RANGE

    def get_vision_cone(self, foods):
        """Simulate vision cone and return visible foods"""
        visible_foods = []
        for food in foods:
            if food.eaten:
                continue
                
            # Calculate distance and angle to food
            dx = food.x - self.x
            dy = food.y - self.y
            dist = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            
            # Normalize angle difference to [-pi, pi]
            angle_diff = (angle - self.angle + np.pi) % (2 * np.pi) - np.pi
            
            # Check if food is within vision cone
            if (dist <= self.vision_range and 
                abs(angle_diff) <= self.vision_angle / 2):
                visible_foods.append((food, dist, angle_diff))
                
        return visible_foods

    def get_inputs(self, foods):
        """Get neural network inputs based on environment state"""
        inputs = []
        
        # Vision cone inputs
        visible_foods = self.get_vision_cone(foods)
        for i in range(VISION_SECTORS):
            sector_found = False
            min_dist = float('inf')
            sector_angle = (-self.vision_angle/2) + (i * self.vision_angle/VISION_SECTORS)
            
            for food, dist, angle in visible_foods:
                if (sector_angle <= angle < sector_angle + self.vision_angle/VISION_SECTORS and 
                    dist < min_dist):
                    inputs.extend([dist / self.vision_range, 1.0])  # Distance and presence
                    sector_found = True
                    min_dist = dist
                    
            if not sector_found:
                inputs.extend([1.0, 0.0])  # No food in this sector
        
        # Current energy level (normalized)
        inputs.append(self.energy / AGENT_ENERGY)
        
        # Current speed (normalized)
        inputs.append(self.speed / AGENT_SPEED)
        
        # Current angle (normalized)
        inputs.append(self.angle / (2 * np.pi))
        
        # Memory inputs
        inputs.extend(self.memory.flatten())
        
        return inputs

    def update(self, foods):
        """Update agent state based on neural network output"""
        if not self.alive:
            return

        # Get neural network inputs
        inputs = self.get_inputs(foods)
        
        # Update memory - shift and add new state
        self.memory = np.roll(self.memory, 1, axis=0)
        self.memory[0] = inputs[:MEMORY_SIZE]  # Store first MEMORY_SIZE inputs
        
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
        """Draw the agent and its vision cone on the screen"""
        if not self.alive:
            return

        # Draw agent body
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.radius)
        
        # Draw direction indicator
        end_x = self.x + np.cos(self.angle) * self.radius
        end_y = self.y + np.sin(self.angle) * self.radius
        pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), (end_x, end_y), 2)
        
        # Draw vision cone
        start_angle = self.angle - self.vision_angle / 2
        end_angle = self.angle + self.vision_angle / 2
        points = [(self.x, self.y)]
        
        for angle in np.linspace(start_angle, end_angle, 20):
            points.append((
                self.x + np.cos(angle) * self.vision_range,
                self.y + np.sin(angle) * self.vision_range
            ))
        
        pygame.draw.polygon(screen, (100, 255, 100, 50), points, 1)

    def eat(self, food):
        """Consume food and gain energy"""
        self.energy = min(AGENT_ENERGY, self.energy + FOOD_ENERGY)
        self.fitness += 1