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

# Ekran oluşturma
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
        self.agent = Agent(input_dim=4, output_dim=4)
        
    def sense_food(self, foods):
        distances = []
        for food in foods:
            dist = math.sqrt((food.x - self.x) ** 2 + (food.y - self.y) ** 2)
            distances.append(dist)
        distances.sort()
        return distances[:4]

    def decide_move(self, sensed_distances):
        state = torch.FloatTensor(sensed_distances).unsqueeze(0).cuda()
        return self.agent.select_action(state)
    
    def reset(self):
        self.x = random.randint(self.radius, self.WIDTH - self.radius)
        self.y = random.randint(self.radius, self.HEIGHT - self.radius)
        self.energy = 100
        self.score = 0
    
    def learn_from_experience(self, sensed_distances, action, reward, next_sensed_distances):
        state = torch.FloatTensor(sensed_distances).unsqueeze(0).cuda()
        next_state = torch.FloatTensor(next_sensed_distances).unsqueeze(0).cuda()
        self.agent.update(state, action, reward, next_state)

    def execute_action(self, action):
        collision_penalty = 0  # Cezanın başlangıç değeri
            
        if action == 0:  # Up
            self.y -= 5
            self.direction = -3.14/2
        elif action == 1:  # Down
            self.y += 5
            self.direction = 3.14/2
        elif action == 2:  # Left
            self.x -= 5
            self.direction = 3.14
        elif action == 3:  # Right
            self.x += 5
            self.direction = 0

        # Enerji tüketimi (hareket başına)
        self.energy -= 0.1

        if self.x <= self.radius or self.x >= WIDTH - self.radius or \
        self.y <= self.radius or self.y >= HEIGHT - self.radius:
            collision_penalty = -0.1
            print("Organism collided with the wall!")

        # Ekran sınırları içinde kalmasını sağlama
        self.x = max(self.radius, min(self.x, WIDTH - self.radius))
        self.y = max(self.radius, min(self.y, HEIGHT - self.radius))

        # Skoru güncelleme (kenarlara çarpma durumunda)
        self.score += collision_penalty

        return collision_penalty


    def draw_score(self):
        font = pygame.font.SysFont(None, 25)
        text = font.render(f"Score: {self.score:.2f}", True, BLACK)
        self.screen.blit(text, (10, 30))

    def draw(self):
        pygame.draw.circle(self.screen, RED, (self.x, self.y), self.radius)
        pygame.draw.line(self.screen, BLACK, (self.x, self.y), 
                        (self.x + self.radius*math.cos(self.direction), 
                        self.y + self.radius*math.sin(self.direction)), 3)