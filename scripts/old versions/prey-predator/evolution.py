import operator
from collections import defaultdict
from math import floor
from random import randint, random, sample, uniform
from modules import Organism, NeuralNetwork, Prey as CustomPrey, Predator as CustomPredator

def mutate_weights(settings, neural_network, weight_type):
    if weight_type == 'wih':
        index_row = randint(0, settings['hnodes']-1)
        neural_network.hidden.weight.data[index_row] *= uniform(0.9, 1.1)
    else:  # MUTATE: WHO WEIGHTS
        index_row = randint(0, settings['onodes']-1)
        index_col = randint(0, settings['hnodes']-1)
        neural_network.output.weight.data[index_row][index_col] *= uniform(0.9, 1.1)
    
    neural_network.hidden.weight.data.clamp_(-1, 1)  # Ensures weights stay in the range [-1, 1]
    neural_network.output.weight.data.clamp_(-1, 1)  # Ensures weights stay in the range [-1, 1]
    return neural_network

def crossover_weights(org1, org2):
    crossover_weight = random()
    wih_new = (crossover_weight * org1.hidden.weight.data) + ((1 - crossover_weight) * org2.hidden.weight.data)
    who_new = (crossover_weight * org1.output.weight.data) + ((1 - crossover_weight) * org2.output.weight.data)
    return wih_new, who_new

def evolve(settings, entities_old, gen, organism_type):
    entities_new = []
    
    if organism_type == 'prey':
        elitism_num = int(floor(settings['elitism'] * settings['prey_count']))
        inodes = settings['inodes_prey']
    elif organism_type == 'predator':
        elitism_num = int(floor(settings['elitism'] * settings['predator_count']))
        inodes = settings['inodes_predator']
    else:
        raise ValueError(f"Invalid organism_type: {organism_type}. Expected 'prey' or 'predator'.")


    if organism_type == 'prey':
        new_entities = settings['prey_count'] - elitism_num
    elif organism_type == 'predator':
        new_entities = settings['predator_count'] - elitism_num

        
    # Get stats from current generation
    stats = defaultdict(int)
    for entity in entities_old:
        if entity.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = entity.fitness
        if entity.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = entity.fitness
        stats['SUM'] += entity.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']

    if organism_type not in ['prey', 'predator']:
        raise ValueError("Invalid entity_type. Expected 'prey' or 'predator'.")

    entities_sorted = sorted(entities_old, key=operator.attrgetter('fitness'), reverse=True)
    
    if organism_type == 'prey':
        entities_new.extend([CustomPrey(settings, neural_network=NeuralNetwork(inodes, settings['hnodes'], settings['onodes']), name=entities_sorted[i].name) for i in range(elitism_num)])
    elif organism_type == 'predator':
        entities_new.extend([CustomPredator(settings, neural_network=entities_sorted[i].neural_network.copy(), name=entities_sorted[i].name) for i in range(elitism_num)])

    # Generate new entities
    for w in range(0, new_entities):
        # Selection (truncation selection)
        candidates = range(0, elitism_num)
        random_index = sample(candidates, 2)
        entity_1 = entities_sorted[random_index[0]]
        entity_2 = entities_sorted[random_index[1]]

        # Crossover
        wih_new, who_new = crossover_weights(entity_1.neural_network, entity_2.neural_network)

        neural_net_new = NeuralNetwork(inodes, settings['hnodes'], settings['onodes'])
        neural_net_new.hidden.weight.data = wih_new
        neural_net_new.output.weight.data = who_new

        # Mutation
        if random() <= settings['mutate']:
            neural_net_new = mutate_weights(settings, neural_net_new, 'wih')
            neural_net_new = mutate_weights(settings, neural_net_new, 'who')

        if organism_type == 'prey':
            entities_new.append(CustomPrey(settings, neural_network=neural_net_new, name=f'gen[{gen}]-entity[{w}]'))
        elif organism_type == 'predator':
            entities_new.append(CustomPredator(settings, neural_network=neural_net_new, name=f'gen[{gen}]-entity[{w}]'))
    return entities_new, stats
