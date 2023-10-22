import pygame
from math import sin, cos, radians
from organisms import Prey, Predator

# Default Colors
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 100)
RED = (255, 0, 0)
DARK_RED = (100, 0, 0)

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

def draw_prey(window, x, y, r, color=GREEN, tail_color=DARK_GREEN):
    pygame.draw.circle(window, color, (x, y), 5)
    pygame.draw.circle(window, tail_color, (x, y), 5, 1)
    tail_len = 7.5
    x2 = int(cos(radians(r)) * tail_len + x)
    y2 = int(sin(radians(r)) * tail_len + y)
    pygame.draw.line(window, tail_color, (x, y), (x2, y2), 1)

def draw_predator(window, x, y, r, color=RED, tail_color=DARK_RED):
    pygame.draw.circle(window, color, (x, y), 10)  # 8 yerine 10 kullandık
    pygame.draw.circle(window, tail_color, (x, y), 7, 1)  # 5 yerine 7 kullandık
    tail_len = 10.5  # 7.5 yerine 10.5 kullandık
    x2 = int(cos(radians(r)) * tail_len + x)
    y2 = int(sin(radians(r)) * tail_len + y)
    pygame.draw.line(window, tail_color, (x, y), (x2, y2), 1)


def draw_food(window, x, y, color=BLUE, border_color=DARK_BLUE):
    pygame.draw.circle(window, color, (x, y), 3)
    pygame.draw.circle(window, border_color, (x, y), 3, 1)

def update_display(window, organisms, foods):
    # Clear the screen with a white background
    window.fill((255, 255, 255))
    
    for organism in organisms:
        if isinstance(organism, Prey):
            draw_prey(window, organism.x, organism.y, organism.r)
        elif isinstance(organism, Predator):
            draw_predator(window, organism.x, organism.y, organism.r)
    
    for food in foods:
        draw_food(window, food.x, food.y)
    
    # Update the display using pygame.display.flip() for potentially better performance
    pygame.display.flip()
