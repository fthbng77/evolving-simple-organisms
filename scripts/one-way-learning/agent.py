import pygame
import numpy as np
import random
import math

# Pygame başlangıcı ve boyut tanımlamaları
pygame.init()
WIDTH, HEIGHT = 320, 240
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Renkler
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0,0,255)

# Organizma özellikleri
organism_size = 20
organism_direction = 0  # Başlangıç yönü
organism_speed = 5
organism_radius = organism_size // 2
organism_position = [WIDTH // 2, HEIGHT // 2]

# Yem özellikleri
goal_size = 20
goal_radius = organism_radius 
goal_position = [100, 100]

# Oyun değişkenleri
score = 0
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.3

# Q-learning için Q-table
action_space_size = 4  # Yukarı, aşağı, sola, sağa
state_space_size = (WIDTH // organism_size, HEIGHT // organism_size)
q_table = np.zeros((state_space_size[0], state_space_size[1], action_space_size))

def execute_action_and_calculate_reward(x, y, direction, action, goal_pos):
    angle_change = math.pi / 8
    reward = -0.1  # Her hareket için küçük bir ceza

    # Eylemlere göre pozisyon ve yön değişikliği
    if action == 0:  # Yukarı
        y -= organism_speed
    elif action == 1:  # Aşağı
        y += organism_speed
    elif action == 2:  # Sola
        x -= organism_speed
        direction -= angle_change
    elif action == 3:  # Sağa
        x += organism_speed
        direction += angle_change

    # Ekran sınırları içinde kalmasını sağlama ve büyük ceza uygulama
    if x < organism_radius or x > WIDTH - organism_radius or y < organism_radius or y > HEIGHT - organism_radius:
        reward -= 5  # Ekran sınırlarını aşma cezası

    # Hedefe yakınlığa göre ödül hesaplama
    distance_to_goal = math.sqrt((x - goal_pos[0])**2 + (y - goal_pos[1])**2)
    if distance_to_goal < goal_radius:
        reward += 10  # Hedefe ulaşma ödülü
    else:
        reward += -distance_to_goal / 100

    x = max(organism_radius, min(x, WIDTH - organism_radius))
    y = max(organism_radius, min(y, HEIGHT - organism_radius))

    return x, y, direction, reward

# Oyun döngüsü
running = True
while running:
    screen.fill(WHITE)

    # Organizma ve yem çizimi
    pygame.draw.circle(screen, RED, organism_position, organism_radius)
    pygame.draw.circle(screen, GREEN, goal_position, goal_radius)

    state = (organism_position[0] // organism_size, organism_position[1] // organism_size)
    
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, 3)  # Rastgele bir aksiyon seç
    else:
        action = np.argmax(q_table[state])

    new_x, new_y, new_direction, reward = execute_action_and_calculate_reward(
        organism_position[0], organism_position[1], organism_direction, action, goal_position)

    new_state = (new_x // organism_size, new_y // organism_size)

    max_future_q = np.max(q_table[new_state])
    current_q = q_table[state + (action,)]
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
    q_table[state + (action,)] = new_q

    organism_position[0], organism_position[1], organism_direction = new_x, new_y, new_direction

    # Hedefe ulaşma durumu
    if reward > 0:
        score += 1
        goal_position = [random.randint(0, (WIDTH - goal_size) // organism_size) * organism_size,
                         random.randint(0, (HEIGHT - goal_size) // organism_size) * organism_size]

    font = pygame.font.SysFont(None, 36)
    score_text = font.render(f'Score: {score}', True, BLUE)
    reward_text = font.render(f'Reward: {reward}', True, BLUE)
    screen.blit(reward_text, (10, 30))
    screen.blit(score_text, (10, 10))

    pygame.display.update()

    # Oyunu kapatma
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.time.delay(100)

pygame.quit()
