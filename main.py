# -*- coding: utf-8 -*-
# @Author: Li Jiaxing
# @Email:li_jax@outlook.com
# @Created: 2024-03-31

from config import places, encode, pop_size
from typing import List
from functions import *
import matplotlib.pyplot as plt



# iteration stop condition: the number of iterations is 100 or the fitness does not improve for 10 continuous iterations
def iterationStop(fitnessList: List, toleration: int  = 10, maxiterations: int = 500) -> bool:
    """
    This function is used to determine whether the iteration stops
    :param fitnessList: a list that stores the fitness of each generation
    :param toleration: the number of continuous iterations that the fitness does not improve
    :param maxiterations: the maximum number of iterations
    :return: True or False, False means the iteration continues
    
    """
    # make sure the length of the list is greater than toleration
    if len(fitnessList) < toleration:
        return False    # iteration continues
    
    # if the the iteration > maxiterations, the iteration stops
    if len(fitnessList) > maxiterations:
        return True # iteration stops

    last_tolerationFitness = fitnessList[-toleration:]

    # check whether the fitness has been improved during the last toleration
    for i in range(1, toleration):
        if last_tolerationFitness[i] > last_tolerationFitness[i - 1]:
            return False  # iteration continues

    




if __name__ == "__main__":
    # initialize the population
    population = init_population(nums = pop_size)

    # calculate the first generation's fitness
    fitness0 = cal_individualFitness(population)
    # calculate the sum of the fitness
    avg_fitness0 = sum(fitness0.values())/len(fitness0)


    # # this list is used to store the fitness of each generation
    fitness_list = [avg_fitness0]
    best_fitness_list = [max(fitness0.values())]

    # loop: iteration for population to evolve
    epochs = 0
    while not iterationStop(fitness_list):
        # calculate the fitness of the current generation
        population = cal_individualFitness(population)
        # select+crossover+mutation
        population = execute_selection(population)

        # calculate the fitness of the new generation
        fitness = cal_individualFitness(population)
        avg_fitness = sum(fitness.values()) / len(fitness)
        fitness_list.append(avg_fitness)

        best_fitness = max(fitness.values())
        best_fitness_list.append(best_fitness)

        print(f"Epochs: {epochs}, Average Fitness: {fitness_list[-1]}, Best Fitness: {best_fitness_list[-1]}")
        epochs += 1

    # polt the fitness curve
    plt.plot(fitness_list, label = "Average Fitness")
    plt.plot(best_fitness_list, label = "Best Fitness") 
    plt.xlabel("Epochs")
    plt.ylabel("Fitness")
    plt.title("Fitness Curve")
    plt.legend()
    plt.show()


    