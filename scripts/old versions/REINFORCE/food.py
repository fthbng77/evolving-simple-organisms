import pygame
import random
from pygame.locals import *

BLUE = (0, 0, 255)

class Food(pygame.sprite.Sprite):
    def __init__(self, screen, width, height):
        super().__init__()
        self.screen = screen
        self.radius = 5
        self.x = random.randint(self.radius, width - self.radius)
        self.y = random.randint(self.radius, height - self.radius)

    def draw(self):
        pygame.draw.circle(self.screen, BLUE, (int(self.x), int(self.y)), self.radius)