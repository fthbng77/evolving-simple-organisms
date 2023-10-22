import operator
from collections import defaultdict
from math import floor
from random import randint, random, sample, uniform
from simulation import Predator, Prey
import numpy as np

def mutate_weights(settings, weights, weight_type):
    if weight_type == 'wih':
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

def evolve(settings, entities_old, gen, entity_type):
    entities_new = [] 
    elitism_num = int(floor(settings['elitism'] * settings['pop_size']))
    new_entities = settings['pop_size'] - elitism_num

    #--- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(int)
    for entity in entities_old:
        if entity.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = entity.fitness
        if entity.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = entity.fitness
        stats['SUM'] += entity.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']

    if entity_type not in ['prey', 'predator']:  # entity_type kontrolü ekle
        raise ValueError("Invalid entity_type. Expected 'prey' or 'predator'.")

    entities_sorted = sorted(entities_old, key=operator.attrgetter('fitness'), reverse=True)
    if entity_type == 'prey':
        entities_new.extend([Prey(settings, wih=entities_sorted[i].wih, who=entities_sorted[i].who, name=entities_sorted[i].name) for i in range(elitism_num)])
    elif entity_type == 'predator':
        entities_new.extend([Predator(settings, wih=entities_sorted[i].wih, who=entities_sorted[i].who, name=entities_sorted[i].name) for i in range(elitism_num)])

    #--- GENERATE NEW ENTITIES ---------------------------+
    for w in range(0, new_entities):
        # SELECTION (TRUNCATION SELECTION)
        candidates = range(0, elitism_num)
        random_index = sample(candidates, 2)
        entity_1 = entities_sorted[random_index[0]]
        entity_2 = entities_sorted[random_index[1]]

        # CROSSOVER
        wih_new = crossover_weights(entity_1.wih, entity_2.wih)
        who_new = crossover_weights(entity_1.who, entity_2.who)

        # MUTATION
        if random() <= settings['mutate']:
            wih_new = mutate_weights(settings, wih_new, 'wih')
            who_new = mutate_weights(settings, who_new, 'who')

        if entity_type == 'prey':
            entities_new.append(Prey(settings, wih=wih_new, who=who_new, name=f'gen[{gen}]-entity[{w}]'))
        elif entity_type == 'predator':
            entities_new.append(Predator(settings, wih=wih_new, who=who_new, name=f'gen[{gen}]-entity[{w}]'))

    return entities_new, stats
