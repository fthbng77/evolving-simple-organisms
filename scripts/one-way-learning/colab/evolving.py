!pip install gymnasium wandb

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import wandb

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()

        pygame.init()
        self.WIDTH, self.HEIGHT = 320, 240
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.action_space = spaces.Discrete(4)
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

class DQN:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

        wandb.config = {
            "learning_rate": self.learning_rate,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma,
            "epsilon_initial": self.epsilon,
            "epsilon_min": self.epsilon_min
        }

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(self.state_space[0], self.state_space[1], self.state_space[2])))
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(48, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0  # Yeterli deneyim yoksa, eğitim yapma

        minibatch = random.sample(self.memory, batch_size)
        start_time = time.time()

        # Tüm mini-batch için hedefleri hesapla
        states = np.array([i[0] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        targets = np.array([self.calculate_target(state, action, reward, next_state, done)
                            for (state, action, reward, next_state, done) in minibatch])

        # Tüm mini-batch üzerinde modeli eğit
        history = self.model.fit(states, np.vstack(targets), epochs=1, verbose=0, batch_size=batch_size)

        training_duration = time.time() - start_time
        average_loss = np.mean(history.history['loss'])

        wandb.log({
            "average_loss": average_loss,
            "epsilon": self.epsilon,
            "training_duration": training_duration
        })

        self.update_epsilon()
        return average_loss


    def calculate_target(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        return target_f

import os

wandb.init(project='evolving', entity='fth123bng')

env = CustomEnv()  # CustomEnv sınıfınızın adını kullanın
state_size = env.observation_space.shape
action_size = env.action_space.n

# DQN ajanınızı başlatın
dqn_agent = DQN(state_size, action_size)

# Eğitim parametreleri
episodes = 100
batch_size = 32

model_save_path = 'dqn_models'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size[0], state_size[1], state_size[2]])
    total_reward = 0

    for time in range(500):
        action = dqn_agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])
        # Deneyimi hafızaya kaydet
        dqn_agent.remember(state, action, reward, next_state, done)
        total_reward += reward

        state = next_state

        if done:
            print(f"Episode: {e}/{episodes}, Score: {time}, Total Reward: {total_reward}")
            wandb.log({"episode": e, "score": time, "total_reward": total_reward})
            break

        if len(dqn_agent.memory) > batch_size:
            dqn_agent.replay(batch_size)

    if dqn_agent.epsilon > dqn_agent.epsilon_min:
        dqn_agent.epsilon *= dqn_agent.epsilon_decay

    # Her 10 bölümde bir modeli kaydet
    if e % 10 == 0:
        model_filename = os.path.join(model_save_path, f'dqn_model_{e}.h5')
        dqn_agent.model.save(model_filename)
        print(f"Model saved: {model_filename}")


dqn_agent.model.save(os.path.join(model_save_path, 'dqn_model_final.h5'))
print("Final model saved")
wandb.save(os.path.join(model_save_path, 'dqn_model_final.h5'))
