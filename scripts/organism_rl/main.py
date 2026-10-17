import argparse
import json
import shutil
from collections import deque
from datetime import datetime
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
RUNS_DIR = Path(__file__).with_name('runs')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train an organism with a named scenario.')
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


def compute_reward(distance_before, distance_after, ate_food, collision_penalty, died):
    reward = -0.01
    if not ate_food:
        reward += 0.2 * (distance_before - distance_after) / 5.0
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
    run_directory = RUNS_DIR / scenario.name / datetime.now().strftime('%Y%m%d-%H%M%S')
    checkpoint_directory = run_directory / 'checkpoints'
    best_checkpoint = checkpoint_directory / 'best_policy.pt'
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(arguments.scenario, run_directory / 'scenario.json')

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Organism Learning Game: {scenario.name}')
    agent = Agent(input_dim=7, output_dim=4, log_dir=run_directory / 'tensorboard')
    organism = Organism(screen, WIDTH, HEIGHT, agent=agent)
    foods = create_foods(screen, organism, scenario)
    target_food = choose_target(organism, foods)
    death_count = 0
    episode_foods_eaten = 0
    episode_steps = 0
    best_foods_eaten = -1
    best_return = float('-inf')
    recent_foods_eaten = deque(maxlen=100)
    recent_returns = deque(maxlen=100)
    best_rolling_foods_eaten = float('-inf')
    print(f'Scenario: {scenario.name}')
    print(f'Run output: {run_directory}')

    running = True
    score_font = pygame.font.SysFont(None, 35)
    energy_font = pygame.font.SysFont(None, 35)
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        simulation_steps = 0
        while running and simulation_steps < scenario.simulation_steps_per_frame:
            state = organism.sense_food(target_food)
            action, log_prob, value, entropy = organism.decide_move(state)
            action_reward = 0.0
            action_steps = 0
            died = False
            while action_steps < scenario.action_repeat and simulation_steps < scenario.simulation_steps_per_frame:
                distance_before = organism.food_distance(target_food)
                collision_penalty = organism.execute_action(action)
                ate_food = organism.food_distance(target_food) <= organism.radius + target_food.radius
                distance_after = organism.food_distance(target_food)

                if ate_food:
                    foods.remove(target_food)
                    foods.append(Food.near_organism(
                        screen, WIDTH, HEIGHT, organism,
                        scenario.minimum_food_distance, scenario.maximum_food_distance,
                    ))
                    target_food = choose_target(organism, foods)
                    organism.energy += 10
                    organism.score += 10
                    episode_foods_eaten += 1
                    organism.writer.add_scalar('Score', organism.score, organism.agent.global_step)

                died = organism.energy <= 0
                action_reward += compute_reward(
                    distance_before, distance_after, ate_food, collision_penalty, died,
                )
                action_steps += 1
                simulation_steps += 1
                episode_steps += 1
                if ate_food or died:
                    break

            organism.learn_from_experience(action_reward, log_prob, value, entropy)
            organism.agent.global_step += action_steps

            if died:
                metrics = organism.agent.finish_episode()
                if metrics:
                    completed_episodes = death_count + 1
                    recent_foods_eaten.append(episode_foods_eaten)
                    recent_returns.append(metrics['return'])
                    rolling_foods_eaten = sum(recent_foods_eaten) / len(recent_foods_eaten)
                    rolling_return = sum(recent_returns) / len(recent_returns)
                    rolling_food_success = sum(count > 0 for count in recent_foods_eaten) / len(recent_foods_eaten)
                    best_rolling_foods_eaten = max(best_rolling_foods_eaten, rolling_foods_eaten)
                    metadata = {
                        'episode': completed_episodes,
                        'foods_eaten': episode_foods_eaten,
                        'return': metrics['return'],
                        'steps': episode_steps,
                        'scenario': scenario.name,
                        'food_count': scenario.food_count,
                        'target_lock': scenario.target_lock,
                        'rolling_foods_eaten': rolling_foods_eaten,
                        'rolling_return': rolling_return,
                        'rolling_food_success': rolling_food_success,
                        'run_directory': str(run_directory),
                    }
                    organism.writer.add_scalar('Episode/return', metrics['return'], death_count)
                    organism.writer.add_scalar('Episode/actor_loss', metrics['actor_loss'], death_count)
                    organism.writer.add_scalar('Episode/critic_loss', metrics['critic_loss'], death_count)
                    organism.writer.add_scalar('Episode/entropy', metrics['entropy'], death_count)
                    organism.writer.add_scalar('Episode/entropy_coefficient', metrics['entropy_coefficient'], death_count)
                    organism.writer.add_scalar('Episode/foods_eaten', episode_foods_eaten, death_count)
                    organism.writer.add_scalar('Episode/steps', episode_steps, death_count)
                    organism.writer.add_scalar('Progress/rolling_100_foods_eaten', rolling_foods_eaten, death_count)
                    organism.writer.add_scalar('Progress/rolling_100_return', rolling_return, death_count)
                    organism.writer.add_scalar('Progress/rolling_100_food_success', rolling_food_success, death_count)
                    organism.writer.add_scalar('Progress/best_rolling_foods_eaten', best_rolling_foods_eaten, death_count)
                    if (episode_foods_eaten, metrics['return']) > (best_foods_eaten, best_return):
                        organism.agent.save_policy(best_checkpoint, metadata)
                        best_foods_eaten = episode_foods_eaten
                        best_return = metrics['return']
                    if scenario.checkpoint_at(completed_episodes):
                        organism.agent.save_policy(
                            checkpoint_directory / f'death_{completed_episodes:06d}.pt', metadata,
                        )
                organism.reset()
                foods = create_foods(screen, organism, scenario)
                target_food = choose_target(organism, foods)
                episode_foods_eaten = 0
                episode_steps = 0
                death_count += 1
                if death_count >= scenario.max_episodes:
                    running = False

        screen.fill(WHITE)
        organism.draw()
        for food in foods:
            food.draw()
        score_text = score_font.render(f'Score: {organism.score:.2f}', True, BLACK)
        energy_text = energy_font.render(f'Energy: {organism.energy:.2f}', True, BLACK)
        entropy_text = energy_font.render(f'Entropy coeff: {organism.agent.entropy_coefficient:.4f}', True, BLACK)
        death_text = score_font.render(f'Deaths: {death_count}', True, BLACK)
        scenario_text = score_font.render(f'Scenario: {scenario.name}; green: target', True, BLACK)
        screen.blit(score_text, (10, 10))
        screen.blit(energy_text, (10, 50))
        screen.blit(entropy_text, (10, 90))
        screen.blit(scenario_text, (10, 130))
        screen.blit(death_text, (WIDTH - 150, 10))
        pygame.display.flip()
        clock.tick(scenario.fps)

    with (run_directory / 'summary.json').open('w', encoding='utf-8') as summary_file:
        json.dump({'scenario': scenario.name, 'episodes_completed': death_count}, summary_file)
    organism.writer.close()
    pygame.quit()


if __name__ == '__main__':
    main()
