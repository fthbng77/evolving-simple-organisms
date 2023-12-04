import pygame
import torch
import math
import random
from agent import Agent

WIDTH, HEIGHT = 800, 600

# Renkler
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Organism Learning Game")

class Organism(pygame.sprite.Sprite):
    def __init__(self, screen, WIDTH, HEIGHT):
        super().__init__()
        self.screen = screen
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        
        self.radius = 20
        self.reset()
        self.direction = random.uniform(0, 2 * math.pi)
        
        self.score = 0.0
        self.agent = Agent(input_dim=8, output_dim=4)
        self.writer = self.agent.writer

    def sense_food(self, foods):
        distances = []
        for food in foods:
            dist = math.sqrt((food.x - self.x) ** 2 + (food.y - self.y) ** 2)
            distances.append(dist)
        distances.sort()
        return distances[:8]

    def decide_move(self, sensed_distances):
        if not isinstance(sensed_distances, torch.Tensor):
            state = torch.FloatTensor(sensed_distances).unsqueeze(0).cuda()
        else:
            state = sensed_distances

        action, log_prob = self.agent.select_action(state)  # İki değeri de döndür
        return action, log_prob
    
    def reset(self):
        self.x = random.randint(self.radius, self.WIDTH - self.radius)
        self.y = random.randint(self.radius, self.HEIGHT - self.radius)
        self.energy = 100
        self.score = 0
    
    def learn_from_experience(self, sensed_distances, action, reward, next_sensed_distances, log_prob):
        state = torch.FloatTensor(sensed_distances).unsqueeze(0).cuda()
        next_state = torch.FloatTensor(next_sensed_distances).unsqueeze(0).cuda()
        self.agent.store_experience(state, action, reward, next_state, log_prob)
        self.agent.update_policy_gradient()

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

        self.energy = self.energy - 10

        if self.x <= self.radius or self.x >= WIDTH - self.radius or \
        self.y <= self.radius or self.y >= HEIGHT - self.radius:
            collision_penalty = -0.1
            self.energy = self.energy - 0.1

        # Ekran sınırları içinde kalmasını sağlama
        self.x = max(self.radius, min(self.x, WIDTH - self.radius))
        self.y = max(self.radius, min(self.y, HEIGHT - self.radius))

        # Skoru güncelleme (kenarlara çarpma durumunda)
        self.score = self.score + collision_penalty

        return collision_penalty

    def draw(self):
        pygame.draw.circle(self.screen, RED, (self.x, self.y), self.radius)
        pygame.draw.line(self.screen, BLACK, (self.x, self.y), 
                        (self.x + self.radius*math.cos(self.direction), 
                        self.y + self.radius*math.sin(self.direction)), 3)