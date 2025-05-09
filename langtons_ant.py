import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 800
CELL_SIZE = 4
GRID_SIZE = WINDOW_SIZE // CELL_SIZE
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Initialize the window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Langton's Ant")
clock = pygame.time.Clock()

class Ant:
    def __init__(self):
        self.x = GRID_SIZE // 2
        self.y = GRID_SIZE // 2
        self.direction = 0  # 0: up, 1: right, 2: down, 3: left
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.steps = 0

    def move(self):
        # Get current cell color
        current_color = self.grid[self.y, self.x]
        
        # Change direction based on current cell color
        if current_color == 0:  # White cell
            self.direction = (self.direction + 1) % 4  # Turn right
        else:  # Black cell
            self.direction = (self.direction - 1) % 4  # Turn left
        
        # Flip the color of the current cell
        self.grid[self.y, self.x] = 1 - current_color
        
        # Move forward
        if self.direction == 0:  # Up
            self.y = (self.y - 1) % GRID_SIZE
        elif self.direction == 1:  # Right
            self.x = (self.x + 1) % GRID_SIZE
        elif self.direction == 2:  # Down
            self.y = (self.y + 1) % GRID_SIZE
        else:  # Left
            self.x = (self.x - 1) % GRID_SIZE
        
        self.steps += 1

    def draw(self):
        # Draw the grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = BLACK if self.grid[y, x] == 1 else WHITE
                pygame.draw.rect(screen, color, 
                               (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw the ant
        ant_rect = pygame.Rect(self.x * CELL_SIZE, self.y * CELL_SIZE, 
                             CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, ant_rect)

def main():
    ant = Ant()
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
                    ant = Ant()  # Reset
                elif event.key == pygame.K_q:
                    running = False

        if not paused:
            ant.move()

        # Draw everything
        screen.fill(WHITE)
        ant.draw()
        
        # Draw step counter
        font = pygame.font.Font(None, 36)
        text = font.render(f"Steps: {ant.steps}", True, BLACK)
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main() 