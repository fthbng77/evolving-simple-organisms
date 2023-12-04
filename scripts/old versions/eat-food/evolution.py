import operator
from collections import defaultdict
from math import floor
from random import randint, random, sample, uniform
from simulation import organism
import numpy as np

def mutate_weights(settings, weights, weight_type):
    if weight_type == 'wih':  # MUTATE: WIH WEIGHTS
        index_row = randint(0, settings['hnodes']-1)
        weights[index_row] = weights[index_row] * uniform(0.9, 1.1)
    else:  # MUTATE: WHO WEIGHTS
        index_row = randint(0, settings['onodes']-1)
        index_col = randint(0, settings['hnodes']-1)
        weights[index_row][index_col] = weights[index_row][index_col] * uniform(0.9, 1.1)
    
    weights = np.clip(weights, -1, 1)  # Ensures weights stay in the range [-1, 1]
    return weights

def crossover_weights(org1_weights, org2_weights):
    crossover_weight = random()
    return (crossover_weight * org1_weights) + ((1 - crossover_weight) * org2_weights)

def evolve(settings, organisms_old, gen):
    elitism_num = int(floor(settings['elitism'] * settings['pop_size']))
    new_orgs = settings['pop_size'] - elitism_num

    #--- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(int)
    for org in organisms_old:
        if org.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = org.fitness
        if org.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = org.fitness
        stats['SUM'] += org.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']

    orgs_sorted = sorted(organisms_old, key=operator.attrgetter('fitness'), reverse=True)
    organisms_new = [organism(settings, wih=orgs_sorted[i].wih, who=orgs_sorted[i].who, name=orgs_sorted[i].name) for i in range(elitism_num)]

    #--- GENERATE NEW ORGANISMS ---------------------------+
    for w in range(0, new_orgs):
        # SELECTION (TRUNCATION SELECTION)
        candidates = range(0, elitism_num)
        random_index = sample(candidates, 2)
        org_1 = orgs_sorted[random_index[0]]
        org_2 = orgs_sorted[random_index[1]]

        # CROSSOVER
        wih_new = crossover_weights(org_1.wih, org_2.wih)
        who_new = crossover_weights(org_1.who, org_2.who)

        # MUTATION
        if random() <= settings['mutate']:
            wih_new = mutate_weights(settings, wih_new, 'wih')
            who_new = mutate_weights(settings, who_new, 'who')

        organisms_new.append(organism(settings, wih=wih_new, who=who_new, name=f'gen[{gen}]-org[{w}]'))

    return organisms_new, stats