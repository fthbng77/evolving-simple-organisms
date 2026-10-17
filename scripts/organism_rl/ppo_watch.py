import argparse
import random
from pathlib import Path

import pygame

from multi_main import ORGANISM_COLORS, choose_target, recolor_foods
from obstacles import move_with_obstacles, obstacle_distances, spawn_food
from organism import Organism
from ppo_agent import PPOAgent
from ppo_curriculum_main import HEIGHT, WIDTH, load_curriculum, rectangles


def parse_arguments():
    parser = argparse.ArgumentParser(description='Watch a PPO obstacle checkpoint at normal speed.')
    parser.add_argument('checkpoints', type=Path, nargs='*')
    parser.add_argument('--seed', type=int, default=100_000)
    return parser.parse_args()


def reset_episode(screen, agent, curriculum, obstacles):
    organisms = [Organism(screen, WIDTH, HEIGHT, agent=agent, color=color) for color in ORGANISM_COLORS]
    foods = []
    for index in range(curriculum['food_count']):
        source = organisms[index % len(organisms)]
        foods.append(spawn_food(
            screen, WIDTH, HEIGHT, source, curriculum['food_distance']['minimum'],
            curriculum['food_distance']['maximum'], obstacles,
        ))
    targets = {organism: choose_target(organism, foods) for organism in organisms}
    recolor_foods(foods, targets)
    return organisms, foods, targets


def main():
    arguments = parse_arguments()
    checkpoints = arguments.checkpoints or [
        Path('runs/ppo_obstacle_curriculum/20261017-025536/checkpoints/episode_040000.pt'),
        Path('runs/ppo_obstacle_curriculum/20261017-025536/checkpoints/episode_060000.pt'),
        Path('runs/ppo_obstacle_curriculum/20261017-025536/checkpoints/episode_085000.pt'),
    ]
    checkpoints = [checkpoint.resolve() for checkpoint in checkpoints]
    checkpoint_index = 0
    checkpoint = checkpoints[checkpoint_index]
    run_directory = checkpoint.parent.parent
    curriculum = load_curriculum(run_directory / 'scenario.json')
    stage = curriculum['stages'][-1]
    obstacles = rectangles(stage)
    random.seed(arguments.seed)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('PPO checkpoint viewer — two obstacles')
    fonts = {'body': pygame.font.SysFont(None, 25), 'small': pygame.font.SysFont(None, 21)}
    agent = PPOAgent(18, 4, log_dir=None)
    agent.load_policy(checkpoint)
    organisms, foods, targets = reset_episode(screen, agent, curriculum, obstacles)
    actions, repeats, food_counts, collisions, steps, trial = [0] * 3, [0] * 3, [0] * 3, [0] * 3, 0, 1
    clock, running = pygame.time.Clock(), True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                random.seed(arguments.seed)
                organisms, foods, targets = reset_episode(screen, agent, curriculum, obstacles)
                actions, repeats, food_counts, collisions, steps, trial = [0] * 3, [0] * 3, [0] * 3, [0] * 3, 0, 1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                checkpoint_index = (checkpoint_index + 1) % len(checkpoints)
                checkpoint = checkpoints[checkpoint_index]
                agent.load_policy(checkpoint)
                random.seed(arguments.seed)
                organisms, foods, targets = reset_episode(screen, agent, curriculum, obstacles)
                actions, repeats, food_counts, collisions, steps, trial = [0] * 3, [0] * 3, [0] * 3, [0] * 3, 0, 1

        for index, organism in enumerate(organisms):
            if repeats[index] == 0:
                state = organism.sense_environment(targets[organism], organisms, obstacle_distances(organism, obstacles))
                actions[index] = agent.select_action(state, deterministic=True)
                repeats[index] = curriculum['action_repeat']
            collision_penalty = move_with_obstacles(organism, actions[index], obstacles)
            if collision_penalty:
                collisions[index] += 1
            repeats[index] -= 1
            eaten = next((food for food in foods if organism.food_distance(food) <= organism.radius + food.radius), None)
            if eaten:
                foods.remove(eaten)
                replacement = spawn_food(
                    screen, WIDTH, HEIGHT, organism, curriculum['food_distance']['minimum'],
                    curriculum['food_distance']['maximum'], obstacles,
                )
                foods.append(replacement)
                for other in organisms:
                    if targets[other] is eaten:
                        targets[other] = choose_target(other, foods)
                targets[organism] = choose_target(organism, foods)
                recolor_foods(foods, targets)
                organism.energy += 10
                organism.score += 10
                food_counts[index] += 1
        steps += 1
        if steps >= curriculum['max_decisions_per_episode'] * curriculum['action_repeat'] or any(organism.energy <= 0 for organism in organisms):
            random.seed(arguments.seed + steps)
            organisms, foods, targets = reset_episode(screen, agent, curriculum, obstacles)
            actions, repeats, food_counts, collisions, steps = [0] * 3, [0] * 3, [0] * 3, [0] * 3, 0
            trial += 1

        screen.fill((246, 248, 247))
        for obstacle in obstacles:
            pygame.draw.rect(screen, (105, 116, 124), obstacle, border_radius=6)
            pygame.draw.rect(screen, (69, 79, 87), obstacle, 2, border_radius=6)
        for food in foods:
            food.draw()
        for organism in organisms:
            target = targets[organism]
            pygame.draw.circle(screen, organism.color, (int(target.x), int(target.y)), target.radius + 7, 2)
            organism.draw()
        title = fonts['body'].render(
            f'PPO test • {checkpoint.name} • deneme {trial} • R: başlat • N: checkpoint değiştir',
            True, (30, 42, 48),
        )
        status = fonts['small'].render(
            f"Yem: {' / '.join(map(str, food_counts))}    Çarpışma: {' / '.join(map(str, collisions))}",
            True, (86, 99, 105),
        )
        screen.blit(title, (14, 12)); screen.blit(status, (14, 40))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == '__main__':
    main()
