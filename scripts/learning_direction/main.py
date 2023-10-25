import pygame
import random
from organism import Organism
from food import Food
import math

WIDTH, HEIGHT = 800, 600
FPS = 30  # Frames per second
screen = pygame.display.set_mode((WIDTH, HEIGHT))
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Pygame başlatma
pygame.init()

organism = Organism(screen, WIDTH, HEIGHT)
foods = [Food(screen, WIDTH, HEIGHT) for _ in range(50)]
death_count = 0 

def compute_reward(org, foods_before, foods_after, collision_penalty):
    if org.energy <= 0:  # Eğer enerji sıfırsa büyük bir ceza döndür
        return -100
    elif len(foods_after) < len(foods_before):
        return 10 + collision_penalty  # yem yendiğinde daha büyük bir ödül
    else:
        return collision_penalty  # hareket edildiğinde ve yem yenilmediğinde ceza (veya ödül)

running = True
score_font = pygame.font.SysFont(None, 35)
energy_font = pygame.font.SysFont(None, 35)  # Enerjiyi göstermek için yazı fontu
clock = pygame.time.Clock()  # FPS için saat nesnesi oluşturma

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    sensed_distances = organism.sense_food(foods)
    action = organism.decide_move(sensed_distances)

    foods_before = foods.copy()
    collision_penalty = organism.execute_action(action)

    foods = [food for food in foods if math.sqrt((food.x - organism.x) ** 2 + (food.y - organism.y) ** 2) > organism.radius + food.radius]
    foods_after = foods

    reward = compute_reward(organism, foods_before, foods_after, collision_penalty)

    # Yem yendiğinde skoru artır ve yeni yiyecek ekle
    if len(foods_after) < len(foods_before):
        foods.append(Food(screen, WIDTH, HEIGHT))
        organism.energy += 10

    if organism.energy <= 0:
        organism.reset()
        death_count += 1
        print(death_count)

    next_sensed_distances = organism.sense_food(foods)
    organism.learn_from_experience(sensed_distances, action, reward, next_sensed_distances)

    organism.draw()
    for food in foods:
        food.draw()

    # Skoru, enerjiyi ve ölüm sayısını ekranda gösterme
    score_text = score_font.render(f"Score: {organism.score:.2f}", True, BLACK)
    energy_text = energy_font.render(f"Energy: {organism.energy:.2f}", True, BLACK)  # Enerjiyi göstermek için
    death_text = score_font.render(f"Deaths: {death_count}", True, BLACK)
    
    screen.blit(score_text, (10, 10))
    screen.blit(energy_text, (10, 50))
    screen.blit(death_text, (WIDTH - 150, 10))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()