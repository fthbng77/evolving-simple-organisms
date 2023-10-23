from simulation import simulate, initialize_window
from evolution import evolve
from random import uniform
import numpy as np
from modules import Predator, Prey, Food
# Simulation settings
settings = {
    'pop_size': 100,       # Number of organisms
    'food_num': 400,      # Number of food particles
    'gens': 50,          # Number of generations
    'elitism': 0.20,      # Elitism (selection bias)
    'mutate': 0.10,       # Mutation rate
    'gen_time': 50,      # Generation length (seconds)
    'dt': 0.04,           # Simulation time step (dt)
    'dr_max': 720,        # Max rotational speed (degrees per second)
    'v_max': 0.75,        # Max velocity (units per second)
    'dv_max': 0.5,        # Max acceleration (+/-) (units per second^2)
    'x_min': 0.0,         # Arena western border
    'x_max': 960.0,       # Arena eastern border
    'y_min': 0.0,         # Arena southern border
    'y_max': 540.0,       # Arena northern border
    'danger_distance': 15,
    'food_distance': 10,
    'eat_distance': 3,
    'plot': True,         # Plot final generation?
    'inodes': 2,          # Number of input nodes
    'hnodes': 5,          # Number of hidden nodes
    'onodes': 2           # Number of output nodes
}

def run(settings):
    window = initialize_window()

    # Populate the environment with food
    foods = [Food(settings) for _ in range(settings['food_num'])]

    preys = [Prey(settings, name=f'gen[x]-prey[{i}]') 
             for i in range(settings['pop_size'])]

    predators = [Predator(settings, name=f'gen[x]-predator[{i}]')
                 for i in range(settings['pop_size'])]

    # Cycle through each generation
    for gen in range(settings['gens']):
        # Simulate
        simulation_results = simulate(settings, preys, predators, foods, gen, window)

        preys, predators = simulation_results

        # Evolve
        preys, stats_preys = evolve(settings, preys, gen, 'prey')
        predators, stats_predators = evolve(settings, predators, gen, 'predator')


        print(f'> GEN: {gen} BEST_PREY: {stats_preys["BEST"]} AVG_PREY: {stats_preys["AVG"]} WORST_PREY: {stats_preys["WORST"]}')
        print(f'> GEN: {gen} BEST_PREDATOR: {stats_predators["BEST"]} AVG_PREDATOR: {stats_predators["AVG"]} WORST_PREDATOR: {stats_predators["WORST"]}')

if __name__ == "__main__":
    run(settings)