import pygame
import random
import math
from pygame.locals import *

BLUE = (0, 0, 255)
GREEN = (0, 180, 0)

class Food(pygame.sprite.Sprite):
    def __init__(self, screen, width, height, color=BLUE):
        super().__init__()
        self.screen = screen
        self.radius = 5
        self.color = color
        self.x = random.randint(self.radius, width - self.radius)
        self.y = random.randint(self.radius, height - self.radius)

    @classmethod
    def near_organism(cls, screen, width, height, organism, minimum_distance=80, maximum_distance=120):
        food = cls(screen, width, height)
        angle = random.uniform(0, 2 * math.pi)
        distance = random.randint(minimum_distance, maximum_distance)
        food.x = int(max(food.radius, min(width - food.radius, organism.x + math.cos(angle) * distance)))
        food.y = int(max(food.radius, min(height - food.radius, organism.y + math.sin(angle) * distance)))
        return food

    def draw(self):
        pygame.draw.circle(self.screen, self.color, (int(self.x), int(self.y)), self.radius)
