from collections import deque
import math
import random

import pygame

from food import Food


SENSOR_DIRECTIONS = tuple(
    (math.cos(index * math.tau / 8), math.sin(index * math.tau / 8))
    for index in range(8)
)


def expanded_obstacle(rectangle, radius):
    return rectangle.inflate(radius * 2, radius * 2)


def collides_with_obstacle(x, y, radius, obstacles):
    return any(expanded_obstacle(obstacle, radius).collidepoint(x, y) for obstacle in obstacles)


def move_with_obstacles(organism, action, obstacles):
    previous_position = organism.x, organism.y
    collision_penalty = organism.execute_action(action)
    if collides_with_obstacle(organism.x, organism.y, organism.radius, obstacles):
        organism.x, organism.y = previous_position
        organism.energy -= 0.5
        organism.score -= 0.5
        collision_penalty -= 0.5
    return collision_penalty


def obstacle_distances(organism, obstacles, maximum_distance=220, step=10):
    distances = []
    for direction_x, direction_y in SENSOR_DIRECTIONS:
        distance = maximum_distance
        for candidate_distance in range(step, maximum_distance + step, step):
            x = organism.x + direction_x * candidate_distance
            y = organism.y + direction_y * candidate_distance
            outside_world = x < organism.radius or x > organism.WIDTH - organism.radius or y < organism.radius or y > organism.HEIGHT - organism.radius
            if outside_world or collides_with_obstacle(x, y, organism.radius, obstacles):
                distance = candidate_distance
                break
        distances.append(min(distance, maximum_distance) / maximum_distance)
    return distances


def spawn_food(screen, width, height, organism, minimum_distance, maximum_distance, obstacles):
    for _ in range(50):
        food = Food.near_organism(
            screen, width, height, organism, minimum_distance, maximum_distance,
        )
        if not collides_with_obstacle(food.x, food.y, food.radius, obstacles):
            return food
    while True:
        food = Food(screen, width, height)
        if not collides_with_obstacle(food.x, food.y, food.radius, obstacles):
            return food


class NavigationField:
    """Coarse shortest-path map. It rewards valid detours around obstacles."""

    cell_size = 20

    def __init__(self, target_food, width, height, organism_radius, obstacles):
        self.width = width
        self.height = height
        self.organism_radius = organism_radius
        self.obstacles = obstacles
        self.columns = math.ceil(width / self.cell_size)
        self.rows = math.ceil(height / self.cell_size)
        self.distances = self._build(target_food.x, target_food.y)

    def _cell(self, x, y):
        return (
            max(0, min(self.columns - 1, int(x // self.cell_size))),
            max(0, min(self.rows - 1, int(y // self.cell_size))),
        )

    def _blocked(self, cell):
        column, row = cell
        x = column * self.cell_size + self.cell_size / 2
        y = row * self.cell_size + self.cell_size / 2
        return (
            x < self.organism_radius or x > self.width - self.organism_radius
            or y < self.organism_radius or y > self.height - self.organism_radius
            or collides_with_obstacle(x, y, self.organism_radius, self.obstacles)
        )

    def _build(self, target_x, target_y):
        target_cell = self._cell(target_x, target_y)
        distances = {target_cell: 0}
        queue = deque([target_cell])
        while queue:
            column, row = queue.popleft()
            for offset_x, offset_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = column + offset_x, row + offset_y
                if (
                    not (0 <= neighbor[0] < self.columns and 0 <= neighbor[1] < self.rows)
                    or neighbor in distances or self._blocked(neighbor)
                ):
                    continue
                distances[neighbor] = distances[(column, row)] + 1
                queue.append(neighbor)
        return distances

    def distance(self, organism):
        return self.distances.get(self._cell(organism.x, organism.y), self.columns + self.rows)
