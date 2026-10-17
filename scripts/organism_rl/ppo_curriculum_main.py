import argparse
import json
import random
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

import pygame

from multi_main import ORGANISM_COLORS, choose_target, recolor_foods
from obstacles import NavigationField, move_with_obstacles, obstacle_distances, spawn_food
from organism import Organism
from ppo_agent import PPOAgent


WIDTH, HEIGHT = 800, 600
WHITE, BLACK, OBSTACLE_COLOR = (246, 248, 247), (30, 42, 48), (105, 116, 124)
SCENARIO_DIR = Path(__file__).with_name('scenarios')
RUNS_DIR = Path(__file__).with_name('runs')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train obstacle navigation with PPO curriculum learning.')
    parser.add_argument('--scenario', type=Path, default=SCENARIO_DIR / 'ppo_curriculum.json')
    return parser.parse_args()


def load_curriculum(path):
    with Path(path).open(encoding='utf-8') as input_file:
        data = json.load(input_file)
    for key in ('name', 'organism_count', 'food_count', 'food_distance', 'action_repeat', 'simulation_steps_per_frame', 'fps', 'stages'):
        if key not in data:
            raise ValueError(f'Missing curriculum field: {key}')
    if data['organism_count'] != 3:
        raise ValueError('The visual curriculum currently requires exactly three organisms')
    if not data['stages']:
        raise ValueError('Curriculum requires at least one stage')
    return data


def rectangles(stage):
    return [pygame.Rect(item['x'], item['y'], item['width'], item['height']) for item in stage['obstacles']]


def create_foods(screen, organisms, curriculum, obstacles):
    distances = curriculum['food_distance']
    return [
        spawn_food(screen, WIDTH, HEIGHT, organisms[index % len(organisms)], distances['minimum'], distances['maximum'], obstacles)
        for index in range(curriculum['food_count'])
    ]


def reset_population(screen, agent, curriculum, obstacles):
    organisms = [
        Organism(screen, WIDTH, HEIGHT, agent=agent, color=color)
        for color in ORGANISM_COLORS
    ]
    foods = create_foods(screen, organisms, curriculum, obstacles)
    fields = {food: NavigationField(food, WIDTH, HEIGHT, 20, obstacles) for food in foods}
    targets = {organism: choose_target(organism, foods) for organism in organisms}
    recolor_foods(foods, targets)
    return organisms, foods, fields, targets


def reward(route_before, route_after, ate_food, collision_penalty, died):
    value = -0.01
    if not ate_food:
        value += 0.25 * (route_before - route_after)
    if ate_food:
        value += 10.0
    if collision_penalty:
        value -= 1.0
    if died:
        value -= 10.0
    return value


def run_episode(screen, agent, curriculum, obstacles, training, on_step=None):
    organisms, foods, fields, targets = reset_population(screen, agent, curriculum, obstacles)
    food_counts, collision_counts = [0] * 3, [0] * 3
    alive = [True] * 3
    decision_count = 0
    while any(alive) and decision_count < curriculum['max_decisions_per_episode']:
        decisions = {}
        for index, organism in enumerate(organisms):
            if not alive[index]:
                continue
            state = organism.sense_environment(targets[organism], organisms, obstacle_distances(organism, obstacles))
            if training:
                action, old_log_prob, value = agent.select_action(state)
            else:
                action, old_log_prob, value = agent.select_action(state, deterministic=True), None, None
            decisions[index] = {'state': state, 'action': action, 'log_prob': old_log_prob, 'value': value, 'reward': 0.0}

        for _ in range(curriculum['action_repeat']):
            for index, decision in decisions.items():
                if not alive[index]:
                    continue
                organism = organisms[index]
                target = targets[organism]
                route_before = fields[target].distance(organism)
                collision_penalty = move_with_obstacles(organism, decision['action'], obstacles)
                if collision_penalty:
                    collision_counts[index] += 1
                eaten = next((food for food in foods if organism.food_distance(food) <= organism.radius + food.radius), None)
                ate_food = eaten is not None
                if ate_food:
                    foods.remove(eaten)
                    fields.pop(eaten)
                    replacement = spawn_food(
                        screen, WIDTH, HEIGHT, organism, curriculum['food_distance']['minimum'],
                        curriculum['food_distance']['maximum'], obstacles,
                    )
                    foods.append(replacement)
                    fields[replacement] = NavigationField(replacement, WIDTH, HEIGHT, organism.radius, obstacles)
                    for other in organisms:
                        if targets[other] is eaten:
                            targets[other] = choose_target(other, foods)
                    targets[organism] = choose_target(organism, foods)
                    recolor_foods(foods, targets)
                    organism.energy += 10
                    organism.score += 10
                    food_counts[index] += 1
                died = organism.energy <= 0
                route_after = route_before if ate_food else fields[target].distance(organism)
                decision['reward'] += reward(route_before, route_after, ate_food, collision_penalty, died)
                if died:
                    alive[index] = False
        decision_count += 1
        if on_step:
            on_step(organisms, foods, targets)

        if training:
            for index, decision in decisions.items():
                terminal = not alive[index] or decision_count >= curriculum['max_decisions_per_episode']
                agent.store_step(index, decision['state'], decision['action'], decision['log_prob'], decision['value'], decision['reward'], terminal)
                agent.global_step += curriculum['action_repeat']
    return food_counts, collision_counts


def evaluate(screen, agent, curriculum, obstacles, episode_count):
    random_state = random.getstate()
    all_foods, all_collisions = [], []
    for seed in range(episode_count):
        random.seed(100_000 + seed)
        foods, collisions = run_episode(screen, agent, curriculum, obstacles, training=False)
        all_foods.extend(foods)
        all_collisions.extend(collisions)
    random.setstate(random_state)
    return {
        'foods_per_organism': sum(all_foods) / len(all_foods),
        'food_success': sum(value > 0 for value in all_foods) / len(all_foods),
        'collisions_per_organism': sum(all_collisions) / len(all_collisions),
    }


def draw(screen, fonts, stage, stage_episode, global_episode, organisms, foods, targets, obstacles, rolling_foods, last_evaluation):
    screen.fill(WHITE)
    pygame.draw.rect(screen, (224, 230, 232), (0, 0, WIDTH, HEIGHT), 2)
    for obstacle in obstacles:
        pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle, border_radius=6)
        pygame.draw.rect(screen, (69, 79, 87), obstacle, 2, border_radius=6)
    for food in foods:
        food.draw()
    for organism in organisms:
        target = targets[organism]
        target_center = int(target.x), int(target.y)
        pygame.draw.circle(screen, organism.color, target_center, target.radius + 7, 2)
        organism.draw()
    hud = fonts['small'].render(
        f"PPO • {stage['name']} • bölüm {global_episode} • yem/canlı {rolling_foods:.2f}",
        True, (95, 107, 112),
    )
    screen.blit(hud, (14, HEIGHT - 26))
    pygame.display.flip()


def main():
    arguments = parse_arguments()
    curriculum = load_curriculum(arguments.scenario)
    run_directory = RUNS_DIR / curriculum['name'] / datetime.now().strftime('%Y%m%d-%H%M%S')
    checkpoint_directory = run_directory / 'checkpoints'
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(arguments.scenario, run_directory / 'scenario.json')
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Organism Learning: PPO curriculum')
    fonts = {'title': pygame.font.SysFont(None, 30), 'body': pygame.font.SysFont(None, 25), 'small': pygame.font.SysFont(None, 21)}
    agent = PPOAgent(input_dim=18, output_dim=4, log_dir=run_directory / 'tensorboard')
    recent_foods = deque(maxlen=100)
    history, global_episode, best_score, running = [], 0, float('-inf'), True

    for stage_index, stage in enumerate(curriculum['stages'], start=1):
        if not running:
            break
        obstacles = rectangles(stage)
        stage_episode, last_evaluation = 0, None
        promoted = False
        while running and stage_episode < stage['maximum_episodes']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running:
                break
            def draw_step(organisms, foods, targets):
                draw(
                    screen, fonts, stage, stage_episode, global_episode, organisms, foods, targets,
                    obstacles, sum(recent_foods) / len(recent_foods) if recent_foods else 0.0, last_evaluation,
                )

            food_counts, collision_counts = run_episode(
                screen, agent, curriculum, obstacles, training=True, on_step=draw_step,
            )
            metrics = agent.finish_rollouts(range(curriculum['organism_count']))
            stage_episode += 1; global_episode += 1
            foods_per_organism = sum(food_counts) / curriculum['organism_count']
            food_success = sum(value > 0 for value in food_counts) / curriculum['organism_count']
            recent_foods.append(foods_per_organism)
            agent.writer.add_scalar('Training/foods_per_organism', foods_per_organism, global_episode)
            agent.writer.add_scalar('Training/food_success', food_success, global_episode)
            agent.writer.add_scalar('Training/collisions_per_organism', sum(collision_counts) / curriculum['organism_count'], global_episode)
            agent.writer.add_scalar('Training/rolling_100_foods', sum(recent_foods) / len(recent_foods), global_episode)
            for name, value in metrics.items():
                agent.writer.add_scalar(f'PPO/{name}', value, global_episode)

            evaluate_now = stage_episode % curriculum['evaluation_interval'] == 0
            if evaluate_now:
                last_evaluation = evaluate(screen, agent, curriculum, obstacles, curriculum['evaluation_episodes'])
                for name, value in last_evaluation.items():
                    agent.writer.add_scalar(f'Evaluation/{name}', value, global_episode)
                metadata = {'global_episode': global_episode, 'stage': stage['name'], 'stage_episode': stage_episode, 'evaluation': last_evaluation}
                if last_evaluation['foods_per_organism'] > best_score:
                    best_score = last_evaluation['foods_per_organism']
                    agent.save_policy(checkpoint_directory / 'best_policy.pt', metadata)
                agent.save_policy(checkpoint_directory / f'episode_{global_episode:06d}.pt', metadata)
                promoted = (
                    stage_episode >= stage['minimum_episodes']
                    and last_evaluation['foods_per_organism'] >= stage['promotion_foods_per_organism']
                    and last_evaluation['food_success'] >= stage['promotion_food_success']
                )
            if promoted:
                break
        history.append({'stage': stage['name'], 'episodes': stage_episode, 'promoted_by_evaluation': promoted, 'last_evaluation': last_evaluation})
        agent.save_policy(checkpoint_directory / f'stage_{stage_index}_complete.pt', {'stage': stage['name'], 'global_episode': global_episode, 'evaluation': last_evaluation})
        if not promoted and stage_index < len(curriculum['stages']):
            print(f"Stopped at stage '{stage['name']}': blind-test target was not reached.")
            running = False
            break

    with (run_directory / 'summary.json').open('w', encoding='utf-8') as output_file:
        json.dump({'scenario': curriculum['name'], 'global_episodes_completed': global_episode, 'stages': history, 'completed': running}, output_file, ensure_ascii=False, indent=2)
    agent.writer.close()
    pygame.quit()
    print(f'Run output: {run_directory}')


if __name__ == '__main__':
    main()
