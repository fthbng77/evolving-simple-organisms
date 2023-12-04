from simulation import organism, food, simulate, initialize_window
from evolution import evolve
from random import uniform
import numpy as np

# Simulation settings
settings = {
    'pop_size': 100,       # Number of organisms
    'food_num': 400,      # Number of food particles
    'gens': 50,          # Number of generations
    'elitism': 0.20,      # Elitism (selection bias)
    'mutate': 0.10,       # Mutation rate
    'gen_time': 200,      # Generation length (seconds)
    'dt': 0.04,           # Simulation time step (dt)
    'dr_max': 720,        # Max rotational speed (degrees per second)
    'v_max': 0.75,        # Max velocity (units per second)
    'dv_max': 0.5,        # Max acceleration (+/-) (units per second^2)
    'x_min': 0.0,         # Arena western border
    'x_max': 960.0,       # Arena eastern border
    'y_min': 0.0,         # Arena southern border
    'y_max': 540.0,       # Arena northern border
    'plot': True,         # Plot final generation?
    'inodes': 1,          # Number of input nodes
    'hnodes': 5,          # Number of hidden nodes
    'onodes': 2           # Number of output nodes
}

def run(settings):
    window = initialize_window()

    # Populate the environment with food
    foods = [food(settings) for _ in range(settings['food_num'])]

    # Populate the environment with organisms
    organisms = [organism(settings, 
                          wih=np.random.uniform(-1, 1, (settings['hnodes'], settings['inodes'])),
                          who=np.random.uniform(-1, 1, (settings['onodes'], settings['hnodes'])),
                          name=f'gen[x]-org[{i}]')
                 for i in range(settings['pop_size'])]

    # Cycle through each generation
    for gen in range(settings['gens']):
        # Simulate
        organisms = simulate(settings, organisms, foods, gen, window)  # Pencereyi parametre olarak ilettik.

        # Evolve
        organisms, stats = evolve(settings, organisms, gen)
        print(f'> GEN: {gen} BEST: {stats["BEST"]} AVG: {stats["AVG"]} WORST: {stats["WORST"]}')

if __name__ == "__main__":
    run(settings)