import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    food_count: int
    target_lock: bool
    minimum_food_distance: int
    maximum_food_distance: int
    max_episodes: int
    checkpoint_episodes: tuple[int, ...]
    checkpoint_interval_after: int
    checkpoint_interval: int
    action_repeat: int
    simulation_steps_per_frame: int
    fps: int

    def checkpoint_at(self, episode: int) -> bool:
        if episode in self.checkpoint_episodes:
            return True
        return episode > self.checkpoint_interval_after and episode % self.checkpoint_interval == 0


def load_scenario(path: Path) -> Scenario:
    path = Path(path)
    with path.open(encoding='utf-8') as scenario_file:
        data = json.load(scenario_file)
    required_fields = {
        'name', 'food_count', 'target_lock', 'food_distance', 'max_episodes',
        'checkpoint_episodes', 'checkpoint_interval_after', 'checkpoint_interval',
        'action_repeat', 'simulation_steps_per_frame', 'fps',
    }
    missing_fields = required_fields - data.keys()
    if missing_fields:
        raise ValueError(f'Missing scenario fields: {sorted(missing_fields)}')
    if data['food_count'] < 1:
        raise ValueError('food_count must be at least 1')
    if data['food_distance']['minimum'] > data['food_distance']['maximum']:
        raise ValueError('food_distance.minimum must not exceed maximum')
    return Scenario(
        name=data['name'], food_count=data['food_count'], target_lock=data['target_lock'],
        minimum_food_distance=data['food_distance']['minimum'],
        maximum_food_distance=data['food_distance']['maximum'],
        max_episodes=data['max_episodes'], checkpoint_episodes=tuple(data['checkpoint_episodes']),
        checkpoint_interval_after=data['checkpoint_interval_after'],
        checkpoint_interval=data['checkpoint_interval'], action_repeat=data['action_repeat'],
        simulation_steps_per_frame=data['simulation_steps_per_frame'], fps=data['fps'],
    )
