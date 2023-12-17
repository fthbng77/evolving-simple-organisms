import pygame
import random
from organism import Organism
from food import Food
import math

WIDTH, HEIGHT = 800, 600
FPS = 30 
screen = pygame.display.set_mode((WIDTH, HEIGHT))
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Pygame başlatma
pygame.init()

organism = Organism(screen, WIDTH, HEIGHT)
foods = [Food(screen, WIDTH, HEIGHT) for _ in range(50)]
death_count = 0 

def compute_reward(org, food_eaten, collision_penalty):
    if org.energy <= 0:
        return -100
    elif food_eaten:
        return 10 + collision_penalty
    else:
        return collision_penalty

def update_food(foods, organism):
    food_eaten = False
    new_foods = []
    for food in foods:
        if math.sqrt((food.x - organism.x) ** 2 + (food.y - organism.y) ** 2) > organism.radius + food.radius:
            new_foods.append(food)
        else:
            food_eaten = True
    return new_foods, food_eaten


running = True
score_font = pygame.font.SysFont(None, 35)
energy_font = pygame.font.SysFont(None, 35)
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    sensed_distances = organism.sense_food(foods)
    action, log_prob = organism.decide_move(sensed_distances)

    collision_penalty = organism.execute_action(action)

    foods, food_eaten = update_food(foods, organism)
    reward = compute_reward(organism, food_eaten, collision_penalty)

    if food_eaten:
        foods.append(Food(screen, WIDTH, HEIGHT))
        organism.energy += 10
        organism.score += 10
        organism.writer.add_scalar('Score', organism.score, organism.agent.global_step)

    if organism.energy <= 0:
        organism.reset()
        foods = [Food(screen, WIDTH, HEIGHT) for _ in range(50)]
        death_count += 1
        organism.agent.death_count = death_count
        organism.agent.end_of_episode()

    next_sensed_distances = organism.sense_food(foods)
    organism.learn_from_experience(sensed_distances, action, reward, next_sensed_distances, log_prob)

    organism.agent.global_step += 1
    organism.draw()
    for food in foods:
        food.draw()

    score_text = score_font.render(f"Score: {organism.score:.2f}", True, BLACK)
    energy_text = energy_font.render(f"Energy: {organism.energy:.2f}", True, BLACK)
    death_text = score_font.render(f"Deaths: {death_count}", True, BLACK)
    epsilon_text = score_font.render(f"Epsilon: {organism.agent.epsilon:.4f}", True, BLACK)
    screen.blit(score_text, (10, 10))
    screen.blit(energy_text, (10, 50))
    screen.blit(epsilon_text, (10, 90))
    screen.blit(death_text, (WIDTH - 150, 10))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()