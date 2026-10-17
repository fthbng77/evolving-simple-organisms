"""Stage 1: train a reusable PPO policy to reach visible food."""

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import pygame

from evolutionary_ppo_main import EvolvingCell, Food, HEIGHT, RUNS_DIR, SCENARIOS_DIR, WIDTH, random_traits
from ppo_agent import PPOAgent


def parse_arguments():
    parser = argparse.ArgumentParser(description='Pretrain food navigation before evolution.')
    parser.add_argument('--scenario', type=Path, default=SCENARIOS_DIR / 'food_navigation_pretrain.json')
    parser.add_argument('--render', action='store_true', help='Show training instead of running headlessly.')
    return parser.parse_args()


def load_scenario(path):
    with Path(path).open(encoding='utf-8') as source:
        data = json.load(source)
    required = {
        'name', 'training_episodes', 'max_decisions_per_episode', 'food_distance',
        'action_repeat', 'evaluation_interval', 'evaluation_episodes', 'success_target',
        'checkpoint_episodes', 'fps',
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f'Missing scenario fields: {sorted(missing)}')
    return data


def new_episode(screen, agent, scenario):
    cell = EvolvingCell(screen, agent, traits=random_traits(), energy=140.0)
    food = Food.near_organism(
        screen, WIDTH, HEIGHT, cell,
        scenario['food_distance']['minimum'], scenario['food_distance']['maximum'],
    )
    return cell, food


def run_episode(screen, agent, scenario, training):
    cell, food = new_episode(screen, agent, scenario)
    for decision in range(scenario['max_decisions_per_episode']):
        state = cell.state(food, [cell])
        if training:
            action, log_prob, value = agent.select_action(state)
        else:
            action, log_prob, value = agent.select_action(state, deterministic=True), None, None
        distance_before = cell.distance_to(food)
        reward, reached = -0.01, False
        for _ in range(scenario['action_repeat']):
            collided = cell.move(action)
            distance_after = cell.distance_to(food)
            reward += 0.08 * (distance_before - distance_after) / max(1.0, cell.traits['speed'])
            reward -= 0.10 if collided else 0.0
            distance_before = distance_after
            if distance_after <= cell.radius + food.radius:
                reward += 10.0
                reached = True
                break
        terminal = reached or cell.energy <= 0 or decision + 1 == scenario['max_decisions_per_episode']
        if training:
            agent.store_step(0, state, action, log_prob, value, reward, terminal)
            agent.global_step += scenario['action_repeat']
        if terminal:
            return reached, decision + 1, cell, food
    raise RuntimeError('Episode did not terminate.')


def evaluate(screen, agent, scenario):
    saved_state = random.getstate()
    outcomes, decisions = [], []
    for seed in range(scenario['evaluation_episodes']):
        random.seed(10_000 + seed)
        reached, decision_count, _, _ = run_episode(screen, agent, scenario, training=False)
        outcomes.append(reached)
        decisions.append(decision_count)
    random.setstate(saved_state)
    return {
        'success_rate': sum(outcomes) / len(outcomes),
        'mean_decisions': sum(decisions) / len(decisions),
    }


def main():
    arguments = parse_arguments()
    scenario = load_scenario(arguments.scenario)
    run_dir = RUNS_DIR / scenario['name'] / datetime.now().strftime('%Y%m%d-%H%M%S')
    checkpoint_dir = run_dir / 'checkpoints'
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(arguments.scenario, run_dir / 'scenario.json')
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT)) if arguments.render else pygame.Surface((WIDTH, HEIGHT))
    if arguments.render:
        pygame.display.set_caption('Stage 1: PPO food navigation pretraining')
    agent = PPOAgent(23, 4, log_dir=run_dir / 'tensorboard')
    clock, font = pygame.time.Clock(), pygame.font.SysFont(None, 26)
    best_success, last_evaluation, completed = -1.0, None, 0

    for episode in range(1, scenario['training_episodes'] + 1):
        for event in pygame.event.get() if arguments.render else []:
            if event.type == pygame.QUIT:
                completed = episode - 1
                break
        else:
            reached, decisions, cell, food = run_episode(screen, agent, scenario, training=True)
            metrics = agent.finish_rollouts([0])
            completed = episode
            agent.writer.add_scalar('Training/success', float(reached), episode)
            agent.writer.add_scalar('Training/decisions', decisions, episode)
            for name, value in metrics.items():
                agent.writer.add_scalar(f'PPO/{name}', value, episode)
            if arguments.render:
                screen.fill((246, 248, 247)); food.draw(); cell.draw()
                hud = font.render(f'PPO ön eğitim | bölüm {episode} | başarılı: {int(reached)}', True, (30, 42, 48))
                screen.blit(hud, (14, 14)); pygame.display.flip(); clock.tick(scenario['fps'])
            if episode % scenario['evaluation_interval'] == 0:
                last_evaluation = evaluate(screen, agent, scenario)
                for name, value in last_evaluation.items():
                    agent.writer.add_scalar(f'Evaluation/{name}', value, episode)
                metadata = {'episode': episode, 'evaluation': last_evaluation, 'scenario': scenario['name']}
                if last_evaluation['success_rate'] > best_success:
                    best_success = last_evaluation['success_rate']
                    agent.save_policy(checkpoint_dir / 'best_policy.pt', metadata)
                if episode in scenario['checkpoint_episodes']:
                    agent.save_policy(checkpoint_dir / f'episode_{episode:06d}.pt', metadata)
                if last_evaluation['success_rate'] >= scenario['success_target']:
                    break
            continue
        break

    final_metadata = {'episode': completed, 'evaluation': last_evaluation, 'scenario': scenario['name']}
    agent.save_policy(checkpoint_dir / 'food_navigation_pretrained.pt', final_metadata)
    with (run_dir / 'summary.json').open('w', encoding='utf-8') as output:
        json.dump(final_metadata, output, ensure_ascii=False, indent=2)
    agent.writer.close(); pygame.quit()
    print(f'Run output: {run_dir}')


if __name__ == '__main__':
    main()
