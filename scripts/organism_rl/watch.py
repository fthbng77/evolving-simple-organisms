import argparse
from pathlib import Path

import pygame

from agent import Agent
from food import BLUE, GREEN, Food
from organism import Organism
from scenario import load_scenario


WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SCENARIO_DIR = Path(__file__).with_name('scenarios')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Watch a saved organism policy at normal speed.')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--scenario', type=Path, default=SCENARIO_DIR / 'one_food.json')
    return parser.parse_args()


def create_foods(screen, organism, scenario):
    return [
        Food.near_organism(
            screen, WIDTH, HEIGHT, organism,
            scenario.minimum_food_distance, scenario.maximum_food_distance,
        )
        for _ in range(scenario.food_count)
    ]


def choose_target(organism, foods):
    for food in foods:
        food.color = BLUE
    target = min(foods, key=organism.food_distance)
    target.color = GREEN
    return target


def main():
    arguments = parse_arguments()
    if not arguments.checkpoint.exists():
        raise SystemExit(f'Checkpoint not found: {arguments.checkpoint}')
    scenario = load_scenario(arguments.scenario)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Organism Policy Watcher')
    agent = Agent(input_dim=7, output_dim=4, log_dir=None)
    agent.load_policy(arguments.checkpoint)
    organism = Organism(screen, WIDTH, HEIGHT, agent=agent)
    foods = create_foods(screen, organism, scenario)
    target_food = choose_target(organism, foods)
    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont(None, 35)
    current_action = None
    action_steps_remaining = 0
    deaths = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if action_steps_remaining == 0:
            state = organism.sense_food(target_food)
            current_action, _, _, _ = organism.agent.select_action(state, deterministic=True)
            action_steps_remaining = scenario.action_repeat

        organism.execute_action(current_action)
        action_steps_remaining -= 1
        if organism.food_distance(target_food) <= organism.radius + target_food.radius:
            foods.remove(target_food)
            foods.append(Food.near_organism(
                screen, WIDTH, HEIGHT, organism,
                scenario.minimum_food_distance, scenario.maximum_food_distance,
            ))
            target_food = choose_target(organism, foods)
            organism.energy += 10
            organism.score += 10
            action_steps_remaining = 0

        if organism.energy <= 0:
            organism.reset()
            foods = create_foods(screen, organism, scenario)
            target_food = choose_target(organism, foods)
            action_steps_remaining = 0
            deaths += 1

        screen.fill(WHITE)
        organism.draw()
        for food in foods:
            food.draw()
        status = score_font.render(
            f'Watch: {arguments.checkpoint.name}  Score: {organism.score:.0f}  Deaths: {deaths}',
            True, BLACK,
        )
        scenario_text = score_font.render(f'Scenario: {scenario.name}; green: target', True, BLACK)
        screen.blit(status, (10, 10))
        screen.blit(scenario_text, (10, 50))
        pygame.display.flip()
        clock.tick(scenario.fps)

    pygame.quit()


if __name__ == '__main__':
    main()
