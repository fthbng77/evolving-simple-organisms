from math import atan2, degrees, radians
import pygame
from utils import dist, calc_heading, remove_dead_organisms
from modules import NeuralNetwork, Predator, Prey, Food
from plotting import draw_prey, draw_predator, draw_food, update_display, handle_events, initialize_window

def simulate(settings, preys, predators, foods, gen, window):

    def update_organisms(organisms, predators=None, foods=None):
        for org in organisms:
            if isinstance(org, Prey):
                org.sense_environment(predators, foods)
                org.think(predators, foods)
            elif isinstance(org, Predator):
                org.sense_environment(preys)
                org.think(predators)
            org.update_velocity(org.dv)
            org.update_angle(org.dr)
            org.update_position()

    total_time_steps = int(settings['gen_time'] / settings['dt'])
    danger_distance = settings['danger_distance']
    food_distance = settings['food_distance']
    eat_distance = settings['eat_distance']
 # Yeme mesafesi eşiği

    for t_step in range(0, total_time_steps, 1):
        if settings['plot']:
            update_display(window, preys + predators, foods)
            pygame.display.update()
            if not handle_events():
                break

        update_organisms(organisms=predators)
        update_organisms(organisms=preys, predators=predators, foods=foods)

        # Prey için yaklaşma mekanizması
        for prey in preys:
            closest_predator_distance = float('inf')
            closest_food_distance = float('inf')
            closest_food_item = None
        
            for predator in predators:
                predator_prey_dist = dist(prey.x, prey.y, predator.x, predator.y)
                if predator_prey_dist < closest_predator_distance:
                    closest_predator_distance = predator_prey_dist

            for food_item in foods:
                food_prey_dist = dist(prey.x, prey.y, food_item.x, food_item.y)
                if food_prey_dist < closest_food_distance:
                    closest_food_distance = food_prey_dist
                    closest_food_item = food_item

            if closest_predator_distance < danger_distance:
                angle_to_predator = atan2(predator.y - prey.y, predator.x - prey.x)
                flee_angle = angle_to_predator + radians(180)
                prey.dr = degrees(flee_angle) - prey.r
                prey.dv = settings['v_max']
            elif closest_food_distance < food_distance:
                angle_to_food = atan2(closest_food_item.y - prey.y, closest_food_item.x - prey.x)
                prey.dr = degrees(angle_to_food) - prey.r
                prey.dv = settings['v_max'] / 2

        # Yemi yeme
        for prey in preys:
            for food_item in foods:
                if dist(prey.x, prey.y, food_item.x, food_item.y) < eat_distance:
                    foods.remove(food_item)
                    break

        # Avcı için av yaklaşma mekanizması
        for predator in predators:
            closest_prey_distance = float('inf')
            closest_prey_item = None

            for prey in preys:
                predator_prey_dist = dist(predator.x, predator.y, prey.x, prey.y)
                if predator_prey_dist < closest_prey_distance:
                    closest_prey_distance = predator_prey_dist
                    closest_prey_item = prey

            if closest_prey_distance < danger_distance:
                angle_to_prey = atan2(closest_prey_item.y - predator.y, closest_prey_item.x - predator.x)
                predator.dr = degrees(angle_to_prey) - predator.r
                predator.dv = settings['v_max']

        # Avı yeme
        for predator in predators:
            for prey in preys:
                if dist(predator.x, predator.y, prey.x, prey.y) < eat_distance:
                    preys.remove(prey)
                    break

        preys = [prey for prey in preys if prey.alive]
        predators = [predator for predator in predators if predator.alive]

    return preys, predators