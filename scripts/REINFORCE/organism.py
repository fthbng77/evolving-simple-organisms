import pygame
import math
import random
from agent import Agent

# Renkler
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Organism(pygame.sprite.Sprite):
    def __init__(self, screen, WIDTH, HEIGHT, agent=None):
        super().__init__()
        self.screen = screen
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.radius = 20
        self.reset()
        self.direction = random.uniform(0, 2 * math.pi)

        self.score = 0.0
        self.agent = agent or Agent(input_dim=7, output_dim=4)
        self.writer = self.agent.writer

    def sense_food(self, target_food):
        return [
            (target_food.x - self.x) / self.WIDTH,
            (target_food.y - self.y) / self.HEIGHT,
            self.energy / 100.0,
            (self.x - self.radius) / self.WIDTH,
            (self.WIDTH - self.radius - self.x) / self.WIDTH,
            (self.y - self.radius) / self.HEIGHT,
            (self.HEIGHT - self.radius - self.y) / self.HEIGHT,
        ]

    def food_distance(self, food):
        return math.hypot(food.x - self.x, food.y - self.y)

    def decide_move(self, sensed_distances):
        return self.agent.select_action(sensed_distances)

    def reset(self):
        self.x = random.randint(self.radius, self.WIDTH - self.radius)
        self.y = random.randint(self.radius, self.HEIGHT - self.radius)
        self.energy = 100
        self.score = 0

    def learn_from_experience(self, reward, log_prob, value, entropy):
        self.agent.store_step(reward, log_prob, value, entropy)

    def execute_action(self, action):
        collision_penalty = 0  # Cezanın başlangıç değeri

        if action == 0:  # Up
            self.y = self.y - 5
            self.direction = -3.14/2
        elif action == 1:  # Down
            self.y = self.y + 5
            self.direction = 3.14/2
        elif action == 2:  # Left
            self.x = self.x - 5
            self.direction = 3.14
        elif action == 3:  # Right
            self.x = self.x + 5
            self.direction = 0

        self.energy = self.energy - 1

        if self.x <= self.radius or self.x >= self.WIDTH - self.radius or \
        self.y <= self.radius or self.y >= self.HEIGHT - self.radius:
            collision_penalty = -0.1
            self.energy = self.energy - 0.1

        # Ekran sınırları içinde kalmasını sağlama
        self.x = max(self.radius, min(self.x, self.WIDTH - self.radius))
        self.y = max(self.radius, min(self.y, self.HEIGHT - self.radius))

        # Skoru güncelleme (kenarlara çarpma durumunda)
        self.score = self.score + collision_penalty

        return collision_penalty

    def draw(self):
        pygame.draw.circle(self.screen, RED, (self.x, self.y), self.radius)
        pygame.draw.line(self.screen, BLACK, (self.x, self.y),
                        (self.x + self.radius*math.cos(self.direction),
                        self.y + self.radius*math.sin(self.direction)), 3)
