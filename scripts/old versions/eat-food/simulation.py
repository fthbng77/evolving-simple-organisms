import numpy as np
from math import atan2, degrees, radians, sqrt, sin, cos
from random import uniform
from plotting import initialize_window, draw_organism, draw_food, handle_events, update_display
import pygame

class food():
    def __init__(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1

    def respawn(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1

class organism():
    def __init__(self, settings, wih=None, who=None, name=None):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.r = uniform(0, 360)
        self.v = uniform(0, settings['v_max'])
        self.dv = uniform(-settings['dv_max'], settings['dv_max'])
        self.d_food = 100
        self.r_food = 0
        self.fitness = 0
        self.wih = wih
        self.who = who
        self.name = name

    def think(self):
        # SIMPLE MLP bu kısım üzerine biraz daha detaylı çalışma yapmam gerekiyor.
        af = lambda x: np.tanh(x)  # activation function
        h1 = af(np.dot(self.wih, self.r_food))
        out = af(np.dot(self.who, h1))
        self.nn_dv = float(out[0])
        self.nn_dr = float(out[1])

    def update_r(self, settings):
        self.r += self.nn_dr * settings['dr_max'] * settings['dt']
        self.r = self.r % 360

    def update_vel(self, settings):
        self.v += self.nn_dv * settings['dv_max'] * settings['dt']
        if self.v < 0: self.v = 0
        if self.v > settings['v_max']: self.v = settings['v_max']

    def update_pos(self, settings):
        dx = self.v * cos(radians(self.r)) * settings['dt']
        dy = self.v * sin(radians(self.r)) * settings['dt']
        self.x += dx
        self.y += dy

def dist(x1, y1, x2, y2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calc_heading(org, food):
    d_x = food.x - org.x
    d_y = food.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180: theta_d += 360
    return theta_d / 180

def simulate(settings, organisms, foods, gen, window):

    total_time_steps = int(settings['gen_time'] / settings['dt'])

    for t_step in range(0, total_time_steps, 1):

        # PLOT SIMULATION FRAME
        if settings['plot']:
            update_display(window, organisms, foods)
            pygame.display.update()
            if not handle_events():
                break

        for org in organisms:
            closest_food_distance = float('inf')
            closest_food_heading = 0

            # FIND CLOSEST FOOD AND UPDATE FITNESS IF NECESSARY
            for food in foods:
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                if food_org_dist < closest_food_distance:
                    closest_food_distance = food_org_dist
                    closest_food_heading = calc_heading(org, food)

                # UPDATE FITNESS FUNCTION
                if food_org_dist <= 0.1:
                    org.fitness += food.energy
                    food.respawn(settings)

            org.d_food = closest_food_distance
            org.r_food = closest_food_heading

            # GET ORGANISM RESPONSE
            org.think()
            org.update_r(settings)
            org.update_vel(settings)
            org.update_pos(settings)

    return organisms