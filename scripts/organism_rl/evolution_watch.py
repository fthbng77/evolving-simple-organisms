"""Play an evolutionary PPO checkpoint at normal speed."""

import argparse
from pathlib import Path

import pygame

from evolutionary_ppo_main import EvolvingCell, Food, HEIGHT, WIDTH, spawn_foods
from ppo_agent import PPOAgent


def parse_arguments():
    parser = argparse.ArgumentParser(description='Watch an evolutionary PPO checkpoint at normal speed.')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--population', type=int, default=20)
    parser.add_argument('--uniform-food', action='store_true', help='Distribute food across the arena instead of near cells.')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if not arguments.checkpoint.exists():
        raise SystemExit(f'Checkpoint not found: {arguments.checkpoint}')

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Evolution checkpoint viewer: {arguments.checkpoint.name}')
    agent = PPOAgent(23, 4, log_dir=None)
    metadata = agent.load_policy(arguments.checkpoint)
    traits = metadata.get('mean_traits')
    population = [EvolvingCell(screen, agent, traits=dict(traits) if traits else None, energy=140.0) for _ in range(arguments.population)]
    foods = [Food(screen, WIDTH, HEIGHT) for _ in range(80)] if arguments.uniform_food else spawn_foods(screen, 80, population)
    clock, font = pygame.time.Clock(), pygame.font.SysFont(None, 25)
    eaten_total = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        candidates = []
        for cell in population:
            target = min(foods, key=cell.distance_to)
            candidates.append((cell, target, cell.state(target, population)))
        # The trainer samples from the PPO distribution.  Do the same here so
        # this playback evaluates the checkpoint under its trained behaviour.
        actions = agent.select_actions([state for _, _, state in candidates])
        for (cell, target, _), (action, _, _) in zip(candidates, actions):
            for _ in range(5):
                cell.move(action)
                eaten = next((food for food in foods if cell.distance_to(food) <= cell.radius + food.radius), None)
                if eaten:
                    foods.remove(eaten)
                    foods.append(
                        Food(screen, WIDTH, HEIGHT)
                        if arguments.uniform_food
                        else Food.near_organism(screen, WIDTH, HEIGHT, cell, 30, 90)
                    )
                    cell.energy += cell.traits['food_energy']
                    eaten_total += 1
            if cell.energy <= 0:
                cell.x, cell.y, cell.energy = WIDTH / 2, HEIGHT / 2, 140.0

        screen.fill((246, 248, 247))
        for food in foods:
            food.draw()
        for cell in population:
            cell.draw()
        title = font.render(f'Test: {arguments.checkpoint.name} | Yenilen yem: {eaten_total}', True, (30, 42, 48))
        layout = 'Yemler haritaya esit dagitiliyor.' if arguments.uniform_food else 'Yemler egitimdeki gibi hucresinin yakininda doguyor.'
        detail = font.render(layout, True, (88, 102, 108))
        screen.blit(title, (12, 12))
        screen.blit(detail, (12, 40))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    main()
