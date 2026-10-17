import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import pygame

from food import Food
from ppo_agent import PPOAgent


WIDTH, HEIGHT = 1200, 800
RUNS_DIR = Path(__file__).with_name('runs')
SCENARIOS_DIR = Path(__file__).with_name('scenarios')
TRAIT_BOUNDS = {
    'speed': (3.0, 8.0),
    'metabolism': (0.45, 1.5),
    'sensor_range': (120.0, 380.0),
    'food_energy': (10.0, 28.0),
    'reproduction_energy': (150.0, 220.0),
}
MUTATION_COLORS = {
    'speed': (177, 126, 255),
    'metabolism': (67, 200, 170),
    'sensor_range': (91, 181, 255),
    'food_energy': (255, 185, 70),
    'reproduction_energy': (255, 99, 146),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train shared PPO movement with evolving body traits.')
    parser.add_argument('--scenario', type=Path, default=SCENARIOS_DIR / 'evolutionary_body_traits.json')
    parser.add_argument('--render', action='store_true', help='Show the simulation window while training.')
    parser.add_argument('--pretrained-policy', type=Path, help='Stage-1 food-navigation policy to reuse.')
    parser.add_argument('--train-policy', action='store_true', help='Fine-tune a loaded policy instead of freezing it.')
    return parser.parse_args()


def load_scenario(path):
    with Path(path).open(encoding='utf-8') as source:
        data = json.load(source)
    required = {'name', 'initial_population', 'population_cap', 'food_count', 'max_simulation_steps', 'max_deaths', 'death_checkpoints', 'action_repeat', 'simulation_steps_per_frame', 'fps', 'ppo_update_transitions', 'metric_interval', 'mutation_rate'}
    missing = required - data.keys()
    if missing:
        raise ValueError(f'Missing scenario fields: {sorted(missing)}')
    return data


def random_traits():
    return {
        'speed': random.uniform(4.0, 6.0),
        'metabolism': random.uniform(0.75, 1.15),
        'sensor_range': random.uniform(180.0, 300.0),
        'food_energy': random.uniform(15.0, 23.0),
        'reproduction_energy': random.uniform(165.0, 205.0),
    }


def mutate(traits, mutation_rate):
    child = dict(traits)
    if random.random() < mutation_rate:
        name = random.choice(tuple(TRAIT_BOUNDS))
        lower, upper = TRAIT_BOUNDS[name]
        child[name] = min(upper, max(lower, child[name] * random.uniform(0.90, 1.10)))
        return child, name
    return child, None


class EvolvingCell:
    next_id = 0

    def __init__(self, screen, agent, traits=None, generation=0, energy=100.0, mutation=None, energy_costs=None):
        self.id = EvolvingCell.next_id
        EvolvingCell.next_id += 1
        self.screen, self.agent = screen, agent
        self.radius = 13
        self.x = random.uniform(self.radius, WIDTH - self.radius)
        self.y = random.uniform(self.radius, HEIGHT - self.radius)
        self.direction = random.uniform(0, 6.283)
        self.energy, self.age, self.generation = energy, 0, generation
        self.traits = traits or random_traits()
        self.mutation = mutation
        self.energy_costs = energy_costs
        self.color = MUTATION_COLORS.get(mutation, (255, 128, 110))

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def state(self, target, population, observe_other_cells=True):
        distance = self.distance_to(target)
        visible = distance <= self.traits['sensor_range']
        dx = (target.x - self.x) / WIDTH if visible else 0.0
        dy = (target.y - self.y) / HEIGHT if visible else 0.0
        nearest = min((other for other in population if other is not self), key=self.distance_to, default=None) if observe_other_cells else None
        other_state = [0.0, 0.0, 1.0] if nearest is None else [
            (nearest.x - self.x) / WIDTH, (nearest.y - self.y) / HEIGHT, self.distance_to(nearest) / WIDTH,
        ]
        return [
            dx, dy, self.energy / 220.0,
            (self.x - self.radius) / WIDTH, (WIDTH - self.radius - self.x) / WIDTH,
            (self.y - self.radius) / HEIGHT, (HEIGHT - self.radius - self.y) / HEIGHT,
            *other_state,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            self.traits['speed'] / 8.0, self.traits['metabolism'] / 1.5,
            self.traits['sensor_range'] / 380.0, self.traits['food_energy'] / 28.0,
            self.traits['reproduction_energy'] / 220.0,
        ]

    def move(self, action):
        speed = self.traits['speed']
        if action == 0:
            self.y -= speed; self.direction = -1.571
        elif action == 1:
            self.y += speed; self.direction = 1.571
        elif action == 2:
            self.x -= speed; self.direction = 3.142
        else:
            self.x += speed; self.direction = 0.0
        collision = self.x <= self.radius or self.x >= WIDTH - self.radius or self.y <= self.radius or self.y >= HEIGHT - self.radius
        self.x = max(self.radius, min(self.x, WIDTH - self.radius))
        self.y = max(self.radius, min(self.y, HEIGHT - self.radius))
        # One visual movement is a short biological time slice, not a full day.
        if self.energy_costs:
            cost = self.energy_costs['base']
            cost += self.energy_costs['metabolism'] * self.traits['metabolism']
            cost += self.energy_costs['speed'] * (self.traits['speed'] / 8.0) ** 2
            cost += self.energy_costs['sensor_range'] * (self.traits['sensor_range'] / 380.0) ** 2
            cost += self.energy_costs['food_energy'] * (self.traits['food_energy'] / 28.0) ** 2
            self.energy -= cost + (0.15 if collision else 0.0)
        else:
            self.energy -= 0.25 * self.traits['metabolism'] + (0.15 if collision else 0.0)
        self.age += 1
        return collision

    def draw(self):
        center = int(self.x), int(self.y)
        for index in range(8):
            angle = 6.283 * index / 8
            inner = int(self.x + (self.radius + 1) * __import__('math').cos(angle)), int(self.y + (self.radius + 1) * __import__('math').sin(angle))
            outer = int(self.x + (self.radius + 7) * __import__('math').cos(angle)), int(self.y + (self.radius + 7) * __import__('math').sin(angle))
            pygame.draw.line(self.screen, (50, 50, 55) if index % 2 == 0 else (92, 103, 186), inner, outer, 2)
        tail = int(self.x - (self.radius + 34) * __import__('math').cos(self.direction)), int(self.y - (self.radius + 34) * __import__('math').sin(self.direction))
        pygame.draw.line(self.screen, (102, 223, 139), center, tail, 3)
        pygame.draw.circle(self.screen, self.color, center, self.radius + 2)


def spawn_foods(screen, count, population, minimum_distance=20, maximum_distance=50):
    """Seed early food close enough for random newborn behavior to survive."""
    return [
        Food.near_organism(screen, WIDTH, HEIGHT, population[index % len(population)], minimum_distance, maximum_distance)
        for index in range(count)
    ]


def mean_trait(population, name):
    return sum(cell.traits[name] for cell in population) / len(population) if population else 0.0


def main():
    arguments = parse_arguments()
    scenario = load_scenario(arguments.scenario)
    run = RUNS_DIR / scenario['name'] / datetime.now().strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(arguments.scenario, run / 'scenario.json')
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT)) if arguments.render else pygame.Surface((WIDTH, HEIGHT))
    if arguments.render:
        pygame.display.set_caption('PPO + evolving body traits')
    fonts = {'body': pygame.font.SysFont(None, 25), 'small': pygame.font.SysFont(None, 20)}
    agent = PPOAgent(23, 4, log_dir=run / 'tensorboard')
    if arguments.pretrained_policy:
        if not arguments.pretrained_policy.exists():
            raise SystemExit(f'Pretrained policy not found: {arguments.pretrained_policy}')
        agent.load_policy(arguments.pretrained_policy)
        shutil.copy2(arguments.pretrained_policy, run / 'pretrained_policy.pt')
    energy_costs = scenario.get('energy_costs')
    food_spawn = scenario.get('food_spawn_distance', {'minimum': 20, 'maximum': 50})
    population = [EvolvingCell(screen, agent, energy=140.0, energy_costs=energy_costs) for _ in range(scenario['initial_population'])]
    foods = spawn_foods(screen, scenario['food_count'], population, food_spawn['minimum'], food_spawn['maximum'])
    births = deaths = simulation_steps = transitions = 0
    saved_death_checkpoints = set()
    rollout_ids, running = set(), True
    clock = pygame.time.Clock()

    while running and population and simulation_steps < scenario['max_simulation_steps'] and deaths < scenario['max_deaths']:
        if arguments.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        for _ in range(scenario['simulation_steps_per_frame']):
            if not population or simulation_steps >= scenario['max_simulation_steps']:
                break
            candidates = []
            for cell in population:
                target = min(foods, key=cell.distance_to)
                state = cell.state(target, population, observe_other_cells=not scenario.get('ignore_other_cells', False))
                candidates.append((cell, target, cell.distance_to(target), state))
            decisions = {
                cell.id: [cell, target, distance, state, action, log_prob, value, 0.0]
                for (cell, target, distance, state), (action, log_prob, value) in zip(
                    candidates,
                    agent.select_actions([item[3] for item in candidates], deterministic=bool(arguments.pretrained_policy and not arguments.train_policy)),
                )
            }
            for _ in range(scenario['action_repeat']):
                for decision in decisions.values():
                    cell, target, distance_before, *_ = decision
                    if cell not in population:
                        continue
                    collided = cell.move(decision[4])
                    eaten = next((food for food in foods if cell.distance_to(food) <= cell.radius + food.radius), None)
                    if eaten:
                        foods.remove(eaten)
                        foods.append(Food.near_organism(screen, WIDTH, HEIGHT, cell, food_spawn['minimum'], food_spawn['maximum']))
                        cell.energy += cell.traits['food_energy']; decision[7] += 8.0
                    elif cell.distance_to(target) <= cell.traits['sensor_range']:
                        decision[7] += 0.04 * (distance_before - cell.distance_to(target)) / max(1.0, cell.traits['speed'])
                    decision[7] -= 0.01 + (0.15 if collided else 0.0)
            newborns = []
            for decision in decisions.values():
                cell = decision[0]
                terminal = cell.energy <= 0
                if not arguments.pretrained_policy or arguments.train_policy:
                    agent.store_step(cell.id, decision[3], decision[4], decision[5], decision[6], decision[7] + (-5.0 if terminal else 0.0), terminal)
                    rollout_ids.add(cell.id); transitions += 1
                if terminal:
                    population.remove(cell); deaths += 1
                elif cell.energy >= cell.traits['reproduction_energy'] and len(population) + len(newborns) < scenario['population_cap']:
                    cell.energy *= 0.5
                    child_traits, mutation = mutate(cell.traits, scenario['mutation_rate'])
                    newborns.append(EvolvingCell(screen, agent, child_traits, cell.generation + 1, cell.energy, mutation, energy_costs))
                    births += 1
            population.extend(newborns)
            simulation_steps += scenario['action_repeat']
            reached_checkpoints = [
                death_count for death_count in scenario['death_checkpoints']
                if death_count <= deaths and death_count not in saved_death_checkpoints
            ]
            for death_count in reached_checkpoints:
                metadata = {
                    'simulation_steps': simulation_steps, 'deaths': death_count, 'births': births,
                    'population': len(population), 'mean_traits': {name: mean_trait(population, name) for name in TRAIT_BOUNDS},
                }
                agent.save_policy(run / 'checkpoints' / f'death_{death_count:06d}.pt', metadata)
                saved_death_checkpoints.add(death_count)
            if transitions >= scenario['ppo_update_transitions']:
                agent.finish_rollouts(rollout_ids); rollout_ids.clear(); transitions = 0
            if simulation_steps % scenario['metric_interval'] == 0 and population:
                for name in TRAIT_BOUNDS:
                    agent.writer.add_scalar(f'Evolution/mean_{name}', mean_trait(population, name), simulation_steps)
                agent.writer.add_scalar('Evolution/population', len(population), simulation_steps)
                agent.writer.add_scalar('Evolution/births', births, simulation_steps)
                agent.writer.add_scalar('Evolution/deaths', deaths, simulation_steps)

        if arguments.render:
            screen.fill((246, 248, 247))
            for food in foods:
                food.draw()
            for cell in population:
                cell.draw()
            policy_mode = 'ince ayar' if arguments.train_policy else ('dondurulmuş beyin' if arguments.pretrained_policy else 'PPO öğreniyor')
            header = fonts['body'].render(f'PPO + EVRİM ({policy_mode})  |  Popülasyon: {len(population)}  Doğum: {births}  Ölüm: {deaths}', True, (30, 42, 48))
            traits = fonts['small'].render(f"Ort. hız {mean_trait(population, 'speed'):.2f}  |  metabolizma {mean_trait(population, 'metabolism'):.2f}  |  sensör {mean_trait(population, 'sensor_range'):.0f}", True, (88, 102, 108))
            legend = fonts['small'].render('Mercan: mutasyonsuz | Mor: hız | Turkuaz: metabolizma | Mavi: sensör | Sarı: yem enerjisi | Pembe: üreme eşiği', True, (88, 102, 108))
            screen.blit(header, (12, 12)); screen.blit(traits, (12, 42))
            screen.blit(legend, (12, 68))
            pygame.display.flip(); clock.tick(scenario['fps'])

    if rollout_ids:
        agent.finish_rollouts(rollout_ids)
    summary = {'simulation_steps': simulation_steps, 'population': len(population), 'births': births, 'deaths': deaths, 'stop_reason': 'death_limit' if deaths >= scenario['max_deaths'] else 'population_extinct_or_step_limit', 'mean_traits': {name: mean_trait(population, name) for name in TRAIT_BOUNDS}}
    with (run / 'summary.json').open('w', encoding='utf-8') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    agent.save_policy(run / 'checkpoints' / 'final_policy.pt', summary)
    agent.writer.close(); pygame.quit()
    print(f'Run output: {run}')


if __name__ == '__main__':
    main()
