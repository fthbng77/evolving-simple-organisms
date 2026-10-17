"""Visually test a stage-1 food-navigation checkpoint."""

import argparse

import pygame

from evolutionary_ppo_main import EvolvingCell, Food, HEIGHT, WIDTH, random_traits
from ppo_agent import PPOAgent


def parse_arguments():
    parser = argparse.ArgumentParser(description='Watch a pretrained food-navigation policy.')
    parser.add_argument('checkpoint')
    return parser.parse_args()


def reset(screen, agent):
    cell = EvolvingCell(screen, agent, traits=random_traits(), energy=140.0)
    food = Food.near_organism(screen, WIDTH, HEIGHT, cell, 60, 150)
    return cell, food


def main():
    arguments = parse_arguments()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Food navigation checkpoint viewer')
    agent = PPOAgent(23, 4, log_dir=None)
    agent.load_policy(arguments.checkpoint)
    cell, food = reset(screen, agent)
    eaten, action, action_steps = 0, 0, 0
    clock, font = pygame.time.Clock(), pygame.font.SysFont(None, 26)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if action_steps == 0:
            action = agent.select_action(cell.state(food, [cell]), deterministic=True)
            action_steps = 5
        cell.move(action); action_steps -= 1
        if cell.distance_to(food) <= cell.radius + food.radius:
            eaten += 1
            cell, food = reset(screen, agent)
        elif cell.energy <= 0:
            cell, food = reset(screen, agent)
        screen.fill((246, 248, 247)); food.draw(); cell.draw()
        label = font.render(f'Ön eğitim testi | Yenilen yem: {eaten}', True, (30, 42, 48))
        screen.blit(label, (14, 14)); pygame.display.flip(); clock.tick(30)
    pygame.quit()


if __name__ == '__main__':
    main()
