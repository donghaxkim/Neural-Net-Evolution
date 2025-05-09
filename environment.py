import numpy as np
import pygame
from config import *

class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = FOOD_RADIUS
        self.eaten = False

    def draw(self, screen):
        if not self.eaten:
            pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.radius)

class Environment:
    def __init__(self):
        self.foods = []
        self.spawn_food()

    def spawn_food(self):
        """Spawn new food particles"""
        while len(self.foods) < FOOD_COUNT:
            x = np.random.randint(FOOD_RADIUS, WINDOW_WIDTH - FOOD_RADIUS)
            y = np.random.randint(FOOD_RADIUS, WINDOW_HEIGHT - FOOD_RADIUS)
            self.foods.append(Food(x, y))

    def update(self, agents):
        """Update environment state and handle interactions"""
        # Check for food consumption
        for agent in agents:
            if not agent.alive:
                continue
                
            for food in self.foods:
                if food.eaten:
                    continue
                    
                # Check if agent is close enough to eat food
                dist = np.sqrt((food.x - agent.x)**2 + (food.y - agent.y)**2)
                if dist < agent.radius + food.radius:
                    agent.eat(food)
                    food.eaten = True

        # Remove eaten food and spawn new food
        self.foods = [food for food in self.foods if not food.eaten]
        self.spawn_food()

    def draw(self, screen):
        """Draw all food particles"""
        for food in self.foods:
            food.draw(screen)

    def reset(self):
        """Reset environment state"""
        self.foods = []
        self.spawn_food() 