import argparse
from pathlib import Path

import pygame

from agent import Agent
from food import Food
from multi_main import (
    BLACK,
    ORGANISM_COLORS,
    SCENARIO_DIR,
    WHITE,
    WIDTH,
    HEIGHT,
    choose_target,
    create_foods,
    recolor_foods,
)
from organism import Organism
from scenario import load_scenario


def parse_arguments():
    parser = argparse.ArgumentParser(description='Watch a multi-organism checkpoint at normal speed.')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--scenario', type=Path, default=SCENARIO_DIR / 'multi_organism.json')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if not arguments.checkpoint.exists():
        raise SystemExit(f'Checkpoint not found: {arguments.checkpoint}')
    scenario = load_scenario(arguments.scenario)
    if scenario.organism_count < 2:
        raise SystemExit('multi_watch.py requires organism_count of at least 2')

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Multi-organism watcher: {arguments.checkpoint.name}')
    agent = Agent(input_dim=10, output_dim=4, log_dir=None)
    metadata = agent.load_policy(arguments.checkpoint)
    organisms = [
        Organism(screen, WIDTH, HEIGHT, agent=agent, color=ORGANISM_COLORS[index % len(ORGANISM_COLORS)])
        for index in range(scenario.organism_count)
    ]
    foods = create_foods(screen, organisms, scenario)
    targets = {organism: choose_target(organism, foods) for organism in organisms}
    recolor_foods(foods, targets)
    action_steps = [0] * scenario.organism_count
    actions = [0] * scenario.organism_count
    deaths = [0] * scenario.organism_count
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for index, organism in enumerate(organisms):
            if action_steps[index] == 0:
                state = organism.sense_environment(targets[organism], organisms)
                actions[index], _, _, _ = agent.select_action(state, deterministic=True)
                action_steps[index] = scenario.action_repeat

            organism.execute_action(actions[index])
            action_steps[index] -= 1
            eaten_food = next(
                (
                    food for food in foods
                    if organism.food_distance(food) <= organism.radius + food.radius
                ),
                None,
            )
            if eaten_food:
                foods.remove(eaten_food)
                foods.append(Food.near_organism(
                    screen, WIDTH, HEIGHT, organism,
                    scenario.minimum_food_distance, scenario.maximum_food_distance,
                ))
                for other in organisms:
                    if targets[other] is eaten_food:
                        targets[other] = choose_target(other, foods)
                targets[organism] = choose_target(organism, foods)
                recolor_foods(foods, targets)
                organism.energy += 10
                organism.score += 10
                action_steps[index] = 0

            if organism.energy <= 0:
                organism.reset()
                targets[organism] = choose_target(organism, foods)
                recolor_foods(foods, targets)
                action_steps[index] = 0
                deaths[index] += 1

        screen.fill(WHITE)
        for food in foods:
            food.draw()
        for organism in organisms:
            organism.draw()
        score_text = ' | '.join(f'{index + 1}: {organism.score:.0f}' for index, organism in enumerate(organisms))
        death_text = ' | '.join(f'{index + 1}: {count}' for index, count in enumerate(deaths))
        title = font.render(f'Watch: {arguments.checkpoint.name}', True, BLACK)
        score = font.render(f'Scores: {score_text}', True, BLACK)
        death = font.render(f'Deaths: {death_text}', True, BLACK)
        result = metadata.get('rolling_average_foods')
        result_text = font.render(
            f'Training rolling avg food/organism: {result:.2f}' if result is not None else '',
            True, BLACK,
        )
        legend = font.render('Green: active target | Red/blue/yellow: shared policy', True, BLACK)
        screen.blit(title, (10, 10))
        screen.blit(score, (10, 40))
        screen.blit(death, (10, 70))
        screen.blit(result_text, (10, 100))
        screen.blit(legend, (10, 130))
        pygame.display.flip()
        clock.tick(scenario.fps)

    pygame.quit()


if __name__ == '__main__':
    main()
