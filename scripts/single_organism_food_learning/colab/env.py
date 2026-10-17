import gymnasium as gym
from gymnasium import spaces
import pygame
import math
import random
import numpy as np
class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()

        pygame.init()
        self.WIDTH, self.HEIGHT = 320, 240
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.action_space = spaces.Discrete(4)  # 4 hareket (yukarı, aşağı, sağ, sol)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

        self.organism_size = 20
        self.organism_direction = 0  # Başlangıç yönü
        self.organism_speed = 5
        self.organism_radius = self.organism_size // 2
        self.organism_position = [self.WIDTH // 2, self.HEIGHT // 2]

        self.goal_size = 20
        self.goal_radius = self.organism_radius
        self.goal_position = [100, 100]

        self.score = 0
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.3

        self.state_space_size = (self.WIDTH // self.organism_size, self.HEIGHT // self.organism_size)
        self.q_table = np.zeros((self.state_space_size[0], self.state_space_size[1], self.action_space.n))

    def step(self, action):
        reward = -0.1
        done = False
        info = {}

        angle_change = math.pi / 8
        x, y = self.organism_position

        if action == 0:
            y -= self.organism_speed
        elif action == 1:
            y += self.organism_speed
        elif action == 2:
            x -= self.organism_speed
            self.organism_direction -= angle_change
        elif action == 3:
            x += self.organism_speed
            self.organism_direction += angle_change

        x = max(self.organism_radius, min(x, self.WIDTH - self.organism_radius))
        y = max(self.organism_radius, min(y, self.HEIGHT - self.organism_radius))

        self.organism_position = [x, y]

        distance_to_goal = math.sqrt((x - self.goal_position[0])**2 + (y - self.goal_position[1])**2)
        if distance_to_goal < self.goal_radius:
            reward += 10
            done = True
            self.goal_position = [random.randint(0, (self.WIDTH - self.goal_size) // self.organism_size) * self.organism_size,
                                  random.randint(0, (self.HEIGHT - self.goal_size) // self.organism_size) * self.organism_size]

        observation = np.array(pygame.surfarray.array3d(self.screen))
        return observation, reward, done, info

    def reset(self):
        self.organism_position = [self.WIDTH // 2, self.HEIGHT // 2]
        self.score = 0
        observation = np.array(pygame.surfarray.array3d(self.screen))
        return observation

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))
        pygame.draw.circle(self.screen, (255, 0, 0), self.organism_position, self.organism_radius)
        pygame.draw.circle(self.screen, (0, 255, 0), self.goal_position, self.goal_radius)
        pygame.display.update()

    def close(self):
        pygame.quit()
