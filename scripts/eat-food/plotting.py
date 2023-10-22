import pygame
from math import sin, cos, radians

# Colors
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 100)

def initialize_window(width=960, height=540, caption='Simulation'):
    pygame.init()
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption(caption)
    return window

def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True

def draw_organism(window, x, y, r):
    pygame.draw.circle(window, GREEN, (x, y), 5)
    pygame.draw.circle(window, DARK_GREEN, (x, y), 5, 1)
    tail_len = 7.5
    x2 = int(cos(radians(r)) * tail_len + x)
    y2 = int(sin(radians(r)) * tail_len + y)
    pygame.draw.line(window, DARK_GREEN, (x, y), (x2, y2), 1)

def draw_food(window, x, y):
    pygame.draw.circle(window, BLUE, (x, y), 3)
    pygame.draw.circle(window, DARK_BLUE, (x, y), 3, 1)

def update_display(window, organisms, foods):
    # Clear the screen with a white background
    window.fill((255, 255, 255))
    
    for organism in organisms:
        # Assuming the organism object has x, y, and r attributes
        draw_organism(window, organism.x, organism.y, organism.r)
    for food in foods:
        # Assuming the food object has x and y attributes
        draw_food(window, food.x, food.y)
    
    # Update the display
    pygame.display.update()