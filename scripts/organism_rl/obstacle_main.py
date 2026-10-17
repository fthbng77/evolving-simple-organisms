import argparse
import json
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

import pygame

from agent import Agent
from food import BLUE, GREEN
from multi_main import ORGANISM_COLORS, choose_target, recolor_foods
from obstacles import NavigationField, move_with_obstacles, obstacle_distances, spawn_food
from organism import Organism
from scenario import load_scenario


WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
OBSTACLE_COLOR = (75, 84, 105)
SCENARIO_DIR = Path(__file__).with_name('scenarios')
RUNS_DIR = Path(__file__).with_name('runs')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train shared-policy organisms to navigate fixed obstacles.')
    parser.add_argument('--scenario', type=Path, default=SCENARIO_DIR / 'multi_organism_obstacles.json')
    return parser.parse_args()


def load_obstacles(path):
    with Path(path).open(encoding='utf-8') as scenario_file:
        data = json.load(scenario_file)
    return [
        pygame.Rect(obstacle['x'], obstacle['y'], obstacle['width'], obstacle['height'])
        for obstacle in data.get('obstacles', [])
    ]


def create_foods(screen, organisms, scenario, obstacles):
    return [
        spawn_food(
            screen, WIDTH, HEIGHT, organisms[index % len(organisms)],
            scenario.minimum_food_distance, scenario.maximum_food_distance, obstacles,
        )
        for index in range(scenario.food_count)
    ]


def create_fields(foods, obstacles):
    return {
        food: NavigationField(food, WIDTH, HEIGHT, 20, obstacles)
        for food in foods
    }


def compute_reward(route_distance_before, route_distance_after, ate_food, collision_penalty, died):
    reward = -0.01
    if not ate_food:
        reward += 0.25 * (route_distance_before - route_distance_after)
    if ate_food:
        reward += 10.0
    if collision_penalty:
        reward -= 1.0
    if died:
        reward -= 10.0
    return reward


def main():
    arguments = parse_arguments()
    scenario = load_scenario(arguments.scenario)
    if scenario.organism_count < 2:
        raise SystemExit('obstacle_main.py requires organism_count of at least 2')
    obstacles = load_obstacles(arguments.scenario)
    if len(obstacles) != 2:
        raise SystemExit('This experiment requires exactly two fixed obstacles')

    run_directory = RUNS_DIR / scenario.name / datetime.now().strftime('%Y%m%d-%H%M%S')
    checkpoint_directory = run_directory / 'checkpoints'
    best_checkpoint = checkpoint_directory / 'best_policy.pt'
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(arguments.scenario, run_directory / 'scenario.json')

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Organism Learning Game: fixed obstacles')
    agent = Agent(input_dim=18, output_dim=4, log_dir=run_directory / 'tensorboard')
    organisms = [
        Organism(screen, WIDTH, HEIGHT, agent=agent, color=ORGANISM_COLORS[index % len(ORGANISM_COLORS)])
        for index in range(scenario.organism_count)
    ]
    foods = create_foods(screen, organisms, scenario, obstacles)
    fields = create_fields(foods, obstacles)
    targets = {organism: choose_target(organism, foods) for organism in organisms}
    recolor_foods(foods, targets)
    active = [True] * scenario.organism_count
    episode_foods_eaten = [0] * scenario.organism_count
    population_deaths = 0
    best_average_foods = float('-inf')
    best_return = float('-inf')
    recent_average_foods = deque(maxlen=100)
    recent_returns = deque(maxlen=100)
    print(f'Scenario: {scenario.name}')
    print(f'Run output: {run_directory}')

    running = True
    font = pygame.font.SysFont(None, 30)
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        simulation_steps = 0
        while running and simulation_steps < scenario.simulation_steps_per_frame:
            decisions = {}
            for index, organism in enumerate(organisms):
                if not active[index]:
                    continue
                sensors = obstacle_distances(organism, obstacles)
                state = organism.sense_environment(targets[organism], organisms, sensors)
                action, log_prob, value, entropy = organism.decide_move(state)
                decisions[index] = {
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'entropy': entropy,
                    'reward': 0.0,
                    'steps': 0,
                }

            for _ in range(scenario.action_repeat):
                if simulation_steps >= scenario.simulation_steps_per_frame:
                    break
                for index, decision in decisions.items():
                    if not active[index]:
                        continue
                    organism = organisms[index]
                    target_food = targets[organism]
                    route_distance_before = fields[target_food].distance(organism)
                    collision_penalty = move_with_obstacles(organism, decision['action'], obstacles)
                    eaten_food = next(
                        (
                            food for food in foods
                            if organism.food_distance(food) <= organism.radius + food.radius
                        ),
                        None,
                    )
                    ate_food = eaten_food is not None
                    if ate_food:
                        foods.remove(eaten_food)
                        fields.pop(eaten_food)
                        new_food = spawn_food(
                            screen, WIDTH, HEIGHT, organism,
                            scenario.minimum_food_distance, scenario.maximum_food_distance, obstacles,
                        )
                        foods.append(new_food)
                        fields[new_food] = NavigationField(new_food, WIDTH, HEIGHT, organism.radius, obstacles)
                        for other in organisms:
                            if targets[other] is eaten_food:
                                targets[other] = choose_target(other, foods)
                        targets[organism] = choose_target(organism, foods)
                        recolor_foods(foods, targets)
                        organism.energy += 10
                        organism.score += 10
                        episode_foods_eaten[index] += 1

                    died = organism.energy <= 0
                    route_distance_after = fields[targets[organism]].distance(organism)
                    decision['reward'] += compute_reward(
                        route_distance_before, route_distance_after, ate_food, collision_penalty, died,
                    )
                    decision['steps'] += 1
                    if died:
                        active[index] = False
                simulation_steps += 1

            for index, decision in decisions.items():
                organisms[index].learn_from_experience(
                    decision['reward'], decision['log_prob'], decision['value'], decision['entropy'], index,
                )
                agent.global_step += decision['steps']

            if not any(active):
                metrics = agent.finish_episodes(range(scenario.organism_count))
                completed_episodes = population_deaths + 1
                average_foods = sum(episode_foods_eaten) / scenario.organism_count
                food_success = sum(food_count > 0 for food_count in episode_foods_eaten) / scenario.organism_count
                recent_average_foods.append(average_foods)
                recent_returns.append(metrics['return'])
                rolling_foods = sum(recent_average_foods) / len(recent_average_foods)
                rolling_return = sum(recent_returns) / len(recent_returns)
                metadata = {
                    'episode': completed_episodes,
                    'organism_count': scenario.organism_count,
                    'food_count': scenario.food_count,
                    'obstacle_count': len(obstacles),
                    'average_foods_eaten': average_foods,
                    'food_success': food_success,
                    'return': metrics['return'],
                    'rolling_average_foods': rolling_foods,
                    'rolling_return': rolling_return,
                    'run_directory': str(run_directory),
                }
                agent.writer.add_scalar('Population/average_foods_eaten', average_foods, population_deaths)
                agent.writer.add_scalar('Population/food_success', food_success, population_deaths)
                agent.writer.add_scalar('Population/return', metrics['return'], population_deaths)
                agent.writer.add_scalar('Progress/rolling_100_average_foods', rolling_foods, population_deaths)
                agent.writer.add_scalar('Progress/rolling_100_return', rolling_return, population_deaths)
                if (average_foods, metrics['return']) > (best_average_foods, best_return):
                    agent.save_policy(best_checkpoint, metadata)
                    best_average_foods = average_foods
                    best_return = metrics['return']
                if scenario.checkpoint_at(completed_episodes):
                    agent.save_policy(checkpoint_directory / f'death_{completed_episodes:06d}.pt', metadata)

                for organism in organisms:
                    organism.reset()
                foods = create_foods(screen, organisms, scenario, obstacles)
                fields = create_fields(foods, obstacles)
                targets = {organism: choose_target(organism, foods) for organism in organisms}
                recolor_foods(foods, targets)
                active = [True] * scenario.organism_count
                episode_foods_eaten = [0] * scenario.organism_count
                population_deaths += 1
                if population_deaths >= scenario.max_episodes:
                    running = False

        screen.fill(WHITE)
        for obstacle in obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle, border_radius=5)
        for food in foods:
            food.draw()
        for organism in organisms:
            organism.draw()
        status = font.render(
            f'Population deaths: {population_deaths}  Foods: {sum(episode_foods_eaten)}  Obstacles: 2',
            True, BLACK,
        )
        legend = font.render('Gray: obstacle | Green: target food | Shared policy', True, BLACK)
        screen.blit(status, (10, 10))
        screen.blit(legend, (10, 45))
        pygame.display.flip()
        clock.tick(scenario.fps)

    with (run_directory / 'summary.json').open('w', encoding='utf-8') as summary_file:
        json.dump({'scenario': scenario.name, 'population_episodes_completed': population_deaths}, summary_file)
    agent.writer.close()
    pygame.quit()


if __name__ == '__main__':
    main()
