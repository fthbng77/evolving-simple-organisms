import numpy as np
from math import atan2, degrees, radians, sqrt, sin, cos
from random import uniform


import numpy as np
from math import atan2, degrees, radians, sqrt, sin, cos
from random import uniform

class Organism:
    def __init__(self, settings, wih=None, who=None, name=None):
        self.settings = settings
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.r = uniform(0, 360)
        self.v = uniform(0, settings['v_max'])
        self.dv = uniform(-settings['dv_max'], settings['dv_max'])
        self.fitness = 0
        self.wih = wih
        self.who = who
        self.name = name
        self.energy = 100
        self.alive = True
    
    def update_energy(self, amount):
        self.energy += amount
        if self.energy <= 0:
            self.alive = False

    def update_velocity(self, dv):
        self.v += dv
        self.v = max(0, min(self.v, self.settings['v_max']))

    def update_position(self):
        self.x += self.v * cos(radians(self.r))
        self.y += self.v * sin(radians(self.r))
        self.x = min(max(self.x, self.settings['x_min']), self.settings['x_max'])
        self.y = min(max(self.y, self.settings['y_min']), self.settings['y_max'])

    def update_angle(self, dr):
        self.r += dr
        self.r %= 360

    def think(self, others):
        raise NotImplementedError

    def update(self, others):
        self.sense_environment(others)
        self.think(others)
        self.update_velocity(self.dv)
        self.update_angle(self.dr)
        self.update_position()

class Prey(Organism):
    def __init__(self, settings, wih=None, who=None, name=None):
        super().__init__(settings, wih, who, name)
        if wih is None:
            self.wih = np.random.rand(5, 4)  # Bu satırı ekleyin
        if who is None:
            self.who = np.random.rand(2, 5)
            
    def sense_environment(self, predators, foods):
        closest_pred_distance = float('inf')
        closest_pred_angle = 0
        closest_food_distance = float('inf')
        closest_food_angle = 0
        
        for predator in predators:
            distance = sqrt((self.x - predator.x)**2 + (self.y - predator.y)**2)
            angle = atan2(predator.y - self.y, predator.x - self.x) - radians(self.r)
            angle = (angle + 2 * np.pi) % (2 * np.pi)  # Normalize angle to [0, 2*pi)
            
            if distance < closest_pred_distance:
                closest_pred_distance = distance
                closest_pred_angle = angle
        
        for food in foods:
            distance = sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
            angle = atan2(food.y - self.y, food.x - self.x) - radians(self.r)
            angle = (angle + 2 * np.pi) % (2 * np.pi)  # Normalize angle to [0, 2*pi)
            
            if distance < closest_food_distance:
                closest_food_distance = distance
                closest_food_angle = angle
        
        self.d_predator = closest_pred_distance
        self.r_predator = closest_pred_angle
        self.d_food = closest_food_distance
        self.r_food = closest_food_angle

    def think(self, predators, foods):
        inputs = np.array([self.d_predator, self.r_predator, self.d_food, self.r_food])
        inputs_normalized = inputs / np.linalg.norm(inputs)
        
        af = lambda x: np.tanh(x)
        
        hidden_layer = af(np.dot(self.wih, inputs_normalized))
        output_layer = af(np.dot(self.who, hidden_layer))
        
        self.dv = output_layer[0] * self.settings['dv_max']
        self.dr = output_layer[1] * self.settings['dr_max']

    def interact(self, predator, foods):
        # Interaction with predator
        predator_distance = sqrt((self.x - predator.x)**2 + (self.y - predator.y)**2)
        if predator_distance < 1.0:
            self.update_energy(-10)
        
        # Interaction with food
        for food in foods:
            food_distance = sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
            if food_distance < 1.0:  # Assume 1.0 is the distance at which prey can eat food
                food.respawn(self.settings)  # Respawn food
                self.update_energy(10)  # Prey gains energy when eating food



class Predator(Organism):
    def __init__(self, settings, wih=None, who=None, name=None):
        super().__init__(settings, wih, who, name)
        if wih is None:
            self.wih = np.random.rand(settings['hnodes'], settings['inodes'])  # Bu satırı ekleyin
        if who is None:
            self.who = np.random.rand(settings['onodes'], settings['hnodes'])  # Bu satırı ekleyin
    
    def sense_environment(self, preys):
        closest_prey_distance = float('inf')
        closest_prey_angle = 0
        
        for prey in preys:
            distance = sqrt((self.x - prey.x)**2 + (self.y - prey.y)**2)
            angle = atan2(prey.y - self.y, prey.x - self.x) - radians(self.r)
            angle = (angle + 2 * np.pi) % (2 * np.pi)  # Normalize angle to [0, 2*pi)
            
            if distance < closest_prey_distance:
                closest_prey_distance = distance
                closest_prey_angle = angle
        
        self.d_prey = closest_prey_distance
        self.r_prey = closest_prey_angle

    def think(self, preys):
        inputs = np.array([self.d_prey, self.r_prey])
        inputs_normalized = inputs / np.linalg.norm(inputs)
        
        # Activation function (you may choose others like ReLU, etc.)
        af = lambda x: np.tanh(x)
        
        # Forward pass through the neural network
        hidden_layer = af(np.dot(self.wih, inputs_normalized))
        output_layer = af(np.dot(self.who, hidden_layer))
        
        self.dv = output_layer[0] * self.settings['dv_max']
        self.dr = output_layer[1] * self.settings['dr_max']

    def interact(self, prey):
        distance = sqrt((self.x - prey.x)**2 + (self.y - prey.y)**2)
        contact_threshold = 10

        if distance < contact_threshold:
            # Predator catches the prey
            energy_transfer = 20  # You can adjust this value
            self.energy += energy_transfer  # Predator gains energy
            prey.update_energy(-energy_transfer)
