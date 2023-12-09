import pygame
import torch
import math
import random
from agent import Agent

WIDTH, HEIGHT = 800, 600
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
        self.agent = Agent(input_dim=16, output_dim=4)
        self.writer = self.agent.writer

    def sense_food(self, foods):
        sensed_info = []
        for food in foods:
            dist = math.sqrt((food.x - self.x) ** 2 + (food.y - self.y) ** 2)
            angle = math.atan2(food.y - self.y, food.x - self.x) - self.direction
            sensed_info.append((dist, angle))
        sensed_info.sort(key=lambda x: x[0])
        return sensed_info[:8]
    
    #yapay zekanın karar mekanizması burada kullanılıyor.
    def decide_move(self, sensed_info):
        state = torch.FloatTensor([item for sublist in sensed_info for item in sublist]).unsqueeze(0).to(self.agent.device)
        action, log_prob = self.agent.select_action(state)
        return action, log_prob
    
    def reset(self):
        self.x = random.randint(self.radius, self.WIDTH - self.radius)
        self.y = random.randint(self.radius, self.HEIGHT - self.radius)
        self.energy = 100
        self.score = 0
    
    def learn_from_experience(self, sensed_info, action, reward, next_sensed_info, log_prob):
        state = torch.FloatTensor([item for sublist in sensed_info for item in sublist]).unsqueeze(0).to(self.agent.device)
        next_state = torch.FloatTensor([item for sublist in next_sensed_info for item in sublist]).unsqueeze(0).to(self.agent.device)
        self.agent.store_experience(state, action, reward, next_state, log_prob)
        self.agent.update_policy_gradient()

    def execute_action(self, action):
        speed = 5
        collision_penalty = 0
        angle_change = math.pi / 8
            
        if action == 0:  # Up
            self.y -= speed
        elif action == 1:  # Down
            self.y += speed
        elif action == 2:  # Left
            self.x -= speed
            self.direction -= angle_change
        elif action == 3:  # Right
            self.x += speed
            self.direction += angle_change

        self.energy -= 0.25

        # Ekran sınırlarına çarpma cezası
        if self.x <= self.radius or self.x >= WIDTH - self.radius or \
           self.y <= self.radius or self.y >= HEIGHT - self.radius:
            collision_penalty = -5
            self.energy -= 5

        # Ekran sınırları içinde kalmasını sağlama
        self.x = max(self.radius, min(self.x, WIDTH - self.radius))
        self.y = max(self.radius, min(self.y, HEIGHT - self.radius))

        self.score += collision_penalty
        return collision_penalty

    def draw(self):
        pygame.draw.circle(self.screen, RED, (self.x, self.y), self.radius)
        pygame.draw.line(self.screen, BLACK, (self.x, self.y), 
                        (self.x + self.radius*math.cos(self.direction), 
                        self.y + self.radius*math.sin(self.direction)), 3)