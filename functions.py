# -*- coding: utf-8 -*-
# @Author: Li Jiaxing
# @Email:li_jax@outlook.com
# @Created: 2024-03-31
# @Last Modified: 2024-04-02

# Description:
import numpy as np
from math import sin, asin, cos, radians, fabs, sqrt
from typing import Dict, Tuple, List
import random
from config import places, encode, pop_size, crossover_probability, mutation_probability, crossover_start

def cal_p2pDistance(place1: Tuple[int, int], place2: Tuple[int, int]) -> float:
    """
    Calculate the distance of 2 points, through the longitude and latitude
    :param place1: a tuple of longitude and latitude
    :param place2: a tuple of longitude and latitude
    :return: the distance, in kilometers
    
    """
    # convert latitude and longitude to radians
    lng1 = radians(place1[0])
    lng2 = radians(place2[0])
    lat1 = radians(place1[1])
    lat2 = radians(place2[1])
    # define the radius of the earth (km)
    EARTH_RADIUS = 6371

    # calculate the difference of the longitude and latitude of this 2 points
    dlng = fabs(lng1 - lng2)
    dlat = fabs(lat1 - lat2)

    # calculate the h value through Haversine formula, and then get the distance finally
    h = sin(dlat / 2)*sin(dlat / 2) + cos(lat1) * cos(lat2) * sin(dlng / 2)*sin(dlng / 2)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    
    return distance

def generate_distanceMatrix(placesDict: Dict[str, tuple] = places) -> np.array:
    """
    Generate a distance matrix of all the places
    :param places: places dictionary in config.py
    :return: a numpy array of the distance matrix, which I can use to calculate the sum of the distances
    
    """
    # form a 2D array in a shape of 15*15 which contains the longitude and latitude of each place
    distance_list = []
    for p in placesDict.keys():
        distance_list.append([cal_p2pDistance(placesDict[p], placesDict[q]) for q in placesDict.keys()])
    distance_matrix = np.array(distance_list).reshape(15, 15)

    return distance_matrix

def cal_seqDistance(encodeStr: str, matrix: np.array) -> float:
    """
    Input the string of squences and global distance matrix, get the sum of the distances
    :param encode: the string of sequences, sample: "ABCDEFGHIJKLMNOA"
    :param matrix: the distance matrix, a golbal variable, which can get from generate_distanceMatrix() at the beginning

    """

    # single character <-> index, because the matrix is a 2D array, and I have to get the distance by index
    # so I convert the character A-O to index 0-14
    charIndex = {c:i for i, c in enumerate("ABCDEFGHIJKLMNO")}

    # encodeStr -> encodeIndex 
    encodeIndex = [charIndex[c] for c in encodeStr]
    # as for the input encodeIndex , I will take them apart in pairs, and calculate and sum up the distances
    sum_distance = 0
    for i in range(len(encodeIndex)-1):
        sum_distance += matrix[encodeIndex[i], encodeIndex[i+1]]

    return sum_distance

def init_population(nums: float = pop_size) -> Dict[str, float]:
    """
    Initialize the population, and give each individual a random sequence such as "ABCDEFGHIJKLMNOA"
    :param nums: the number of individuals in the population
    :return: a dictionary of individuals, the key is the sequence, and the value is the fitness of the individual,
             the default fitness is 0.0

    """

    # generate a dict of individuals, and the sequence is random，and the fitness is 0.0
    # there I temporarily ignore the A in the start and end of the sequence, because it's the same place
    # and I will shuffle a list cotaining the numbers 1-14 for nums times, as the represent of the places B-O

    numlist = list(range(1, 15))
    charIndex = {i:c for i, c in enumerate("ABCDEFGHIJKLMNO")}
    population = {}
    for i in range(nums):
        random.shuffle(numlist)
        population["A"+"".join(charIndex[i] for i in numlist )+"A"] = 0.0

    return population

def cal_individualFitness(population: Dict[str, float]) -> Dict[str, float]:
    """
    Through the function cal_seqDistance(), I can use the population dict to calculate the sum distance,
    and the fitness is the reciprocal of the distance
    :param population: a list of individuals, got from init_population()
    :return: a dictionary of individuals, the key is the sequence, and the value is the fitness of the individual

    """

    for p in population.keys():
        population[p] = 1 / (cal_seqDistance(p, generate_distanceMatrix()) + 1e-6)

    return population

def cal_selectionProbability(population: Dict[str, float]) -> Dict[str, float]:
    """
    calculate the selection probability of each individual, which is the ratio
    of individual fitness to the sum fitness of the population.
    :param population: a list of fitness in a particular population, with fitness as the value
    :return: a dictionary of each individual, the key is the sequence, and the value is the selection probability
    
    """
    sum_fitness = sum(population.values())
    population_ps = {p:population[p] / sum_fitness for p in population.keys()}

    return population_ps

def recoverySequence(strSeq: str) -> str:
    """
    After crossover and mutation, the sequence may be abnormal, so I have to recover the right regulation
    :param strSeq: the sequence of the individual
    :return: the right sequence of the individual
    
    """

    charCount = {c:strSeq.strip("A").count(c) for c in strSeq}
    for c in "BCDEFGHIJKLMNO":
            if c not in charCount.keys():
                charCount[c] = 0
    key1 = [key for key, value in charCount.items() if key != 'A' and value > 1]
    key2 = [key for key, value in charCount.items() if key != 'A' and value < 1]
    # loop the strSeq in reverse order, and replace the 0 value with the >1 value
    for i in range(len(key1)):
        for char in strSeq[::-1]:
            key1 = [key for key, value in charCount.items() if key != 'A' and value > 1]
            key2 = [key for key, value in charCount.items() if key != 'A' and value < 1]
            if len(key1) == 0 or len(key2) == 0:
                break
            elif char == key1[0]:
                # update strSeq, replace key1 with key2
                strSeq = strSeq[:strSeq.rfind(key1[0])] + key2[0] + strSeq[strSeq.rfind(key1[0])+1:]
                # update charCount
                charCount[key1[0]] -= 1
                charCount[key2[0]] += 1
                continue
            else:
                continue

    return strSeq

def execute_selection(population: Dict[str, float], pc: float = crossover_probability, pm: float = mutation_probability, crossover_start: int = crossover_start) -> Dict[str, float]:
    """
    according to the selection probability, I will conduct the selection, crossover and mutation there
    :param population_ps: a dictionary of individuals, the key is the sequence, and the value is the fitness of the individual
    :param pc: crossover probability
    :param pm: mutation probability
    :return: a new generation of individuals, the key is the sequence, and the value is the fitness of the individual
    
    """

    """--------- Selection ---------
    I will calculate the cumulative probability of individuals, and then add some individuals 
    to the new population according to the roulette wheel selection method.
    
    """
    # calculate the selection probability of individuals
    population_ps = cal_selectionProbability(population)
    # calculate the cumulative probability of individuals
    population_cps = {}
    sum_ps = 0.0
    for i in population_ps.keys():
        sum_ps += population_ps[i]
        population_cps[i] = sum_ps

    # roulette wheel selection: select individuals according to the selection probability
    population_new = {}
    while len(population_new) < len(population):
        r = random.random()
        for i in population_cps.keys():     
            if i == 1 and r <= population_cps[i]:
                population_new[i] = population[i]
                break
            else:
                # value - r, select the smallest +value
                tmp = {k:v - r for k, v in population_cps.items()}
                tmp = {k:v for k, v in tmp.items() if v > 0}
                i = min(tmp, key=tmp.get)
                population_new[i] = population[i]
                break
                
    """--------- Crossover ---------
    Crossover the individuals in the new population, and the probability is pc.
    When a random number(0.0-1.0) is less than pc, I will conduct the crossover operation. Then there will be 2 new individuals,
    which I will replace their parents in the population_new.
    
    """
    pop_add = []
    pop_crossNew = {}
    while len(pop_add) < len(population):
        # select 2 individual randomly, not the same one
        ind1 = random.choice(list(population_new.keys()))
        keys_excluding_ind1 = [key for key in population_new.keys() if key != ind1]
        ind2 = random.choice(list(keys_excluding_ind1))  
        # whether crossover or not, depends on the probability pc
        if random.random() < pc:    # do the crossover
            # according to the crossover start point, I will crossover this 2 individuals, then get 2 new individuals
            new_ind1 = ind1[:crossover_start] + ind2[crossover_start:]
            new_ind2 = ind2[:crossover_start] + ind1[crossover_start:]
            # recover the sequence
            new_ind1 = recoverySequence(new_ind1)
            new_ind2 = recoverySequence(new_ind2)
            if ((new_ind1 and new_ind2) not in pop_add) and (new_ind1 != new_ind2):
                # add in pop_add
                pop_add.append(new_ind1)
                pop_add.append(new_ind2)
            else :
                continue
    
    # add the individuals in pop_add to the population_new
    for ind in pop_add:
        pop_crossNew[ind] = 0.0
    del population_new

    """--------- mutation ---------
    Be careful that I must make sure that the A in the start and end of the sequence.
    Because the sequence must obey the regulation, so changing a single character may cause the sequence to be abnormal.
    So I choose to swap 2 characters randomly as a mutation operation. the result is same as changing a single character and 
    then recover the sequence.
    
    """
    
    pop_add2 = []
    pop_mutNew = {}
    while len(pop_add2) < len(population):
        # select an individual randomly
        ind = random.choice(list(pop_crossNew.keys()))
        if random.random() < pm:
            # delete the A in the start and end of the sequence
            ind = ind.strip("A")
            # select 2 points randomly, and then swap them
            idx1 = random.randint(0, 13)
            idx2 = random.randint(0, 13)
            ind = list(ind)
            ind[idx1], ind[idx2] = ind[idx2], ind[idx1]
            # recover the sequence
            ind = "A" + "".join(ind) + "A"
            # update the population_new
            if ind not in pop_add2:
                pop_add2.append(ind)
            else:
                continue

    # add the individuals in pop_add2
    for ind in pop_add2:
        pop_mutNew[ind] = 0.0
            
    return pop_mutNew

if __name__ == "__main__":
    population1 = init_population()
    population1 = cal_individualFitness(population1)
    pop_new = execute_selection(population1)

    print('-----------------------')
    print(len(pop_new))

    pass