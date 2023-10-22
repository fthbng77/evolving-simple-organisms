from math import atan2, degrees, radians, sqrt, sin, cos
from random import uniform
from plotting import initialize_window, draw_prey, draw_predator, draw_food, handle_events, update_display
import pygame
from organisms import Prey, Predator

class food():
    def __init__(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1

    def respawn(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1

def dist(x1, y1, x2, y2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calc_heading(org, target):
    d_x = target.x - org.x
    d_y = target.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180: theta_d += 360
    return theta_d / 180

def simulate(settings, preys, predators, foods, gen, window):
    total_time_steps = int(settings['gen_time'] / settings['dt'])
    danger_distance = 20  # You need to set an appropriate value
    food_distance = 10    # You need to set an appropriate value

    for t_step in range(0, total_time_steps, 1):
        # PLOT SIMULATION FRAME
        if settings['plot']:
            update_display(window, preys + predators, foods)
            pygame.display.update()
            if not handle_events():
                break

        for predator in predators:
            closest_prey_distance = float('inf')
            closest_prey_heading = 0
            for prey in preys:
                prey_predator_dist = dist(predator.x, predator.y, prey.x, prey.y)
                if prey_predator_dist < closest_prey_distance:
                    closest_prey_distance = prey_predator_dist
                    closest_prey_heading = calc_heading(predator, prey)

            predator.d_prey = closest_prey_distance
            predator.r_prey = closest_prey_heading
            predator.sense_environment(preys)
            predator.think(preys)
            predator.update_velocity(predator.dv)
            predator.update_angle(predator.dr)
            predator.update_position()

        for prey in preys:
            closest_predator_distance = float('inf')
            closest_predator_heading = 0
            prey.sense_environment(predators, foods)  # foods parametresini ekledik
            for predator in predators:
                predator_prey_dist = dist(prey.x, prey.y, predator.x, predator.y)
                if predator_prey_dist < closest_predator_distance:
                    closest_predator_distance = predator_prey_dist
                    closest_predator_heading = calc_heading(prey, predator)

            closest_food_distance = float('inf')
            closest_food_heading = 0
            
            for food_item in foods:
                food_prey_dist = dist(prey.x, prey.y, food_item.x, food_item.y)
                if food_prey_dist < closest_food_distance:
                    closest_food_distance = food_prey_dist
                    closest_food_heading = calc_heading(prey, food_item)

            if closest_predator_distance < danger_distance:
                # Code to make prey flee from predator
                angle_to_predator = atan2(predator.y - prey.y, predator.x - prey.x)
                flee_angle = angle_to_predator + radians(180)  # opposite direction
                prey.dr = degrees(flee_angle) - prey.r  # adjusting the angle to flee
                prey.dv = settings['v_max']  # setting the velocity to max to flee quickly

            elif closest_food_distance < food_distance:
                # Code to make prey move towards food
                angle_to_food = atan2(food_item.y - prey.y, food_item.x - prey.x)  # used food_item instead of food
                prey.dr = degrees(angle_to_food) - prey.r  # adjusting the angle to move towards food
                prey.dv = settings['v_max'] / 2  # setting a moderate velocity to move towards food

            prey.d_predator = closest_predator_distance
            prey.r_predator = closest_predator_heading
            prey.d_food = closest_food_distance
            prey.r_food = closest_food_heading
            prey.think(predators,foods)
            prey.update_velocity(prey.dv)
            prey.update_angle(prey.dr)
            prey.update_position()

        # Remove dead preys and predators from the simulation
        preys = [prey for prey in preys if prey.alive]
        predators = [predator for predator in predators if predator.alive]

    return preys, predators


