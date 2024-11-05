############## 
# Many of these are the same imports as the odell_tsptools module
##############

import math # to calculate square roots when implementing distance formula
import glob # to iterate through my directory via wildcard * (e.g., *.tsp) and read all .tsp files
import time # for the .time() method so that we can time how long the TSP algorithms take
import copy # to create deep copies when defining recursive methods
import matplotlib.pyplot as plt # for plotting our lists of tuples, i.e., creating a 'geographical map'
from matplotlib.cm import ScalarMappable
import seaborn as sns # for more plotting options
import queue # for implementing FIFO and priority queue for various search algorithms
import numpy as np # used for random number array generations, especially for the genetic algorithms
import random # for random SINGLE number generations, especially for getting a quick probability check
import pandas as pd # to create DataFrame objects for genetic algorithm, allowing for later plotting and what-not
import warnings
from collections import Counter # for counting unique Hamming distances in the SolutionString class
import networkx as nx # for visualizing the solutions during wisdom of the crowds


import scipy.stats as stats     # for the Kruskal-Wallis test
from scipy.interpolate import make_interp_spline
import scikit_posthocs as sp    # for the post-hodc tests (Dunn's with Bonferoroni correction)
                                # Needs to be installed using pip:
                                # $ pip install scikit_posthocs
import os
import tkinter as tk            # for our simple GUI
from tkinter import font        # for font options in the GUI
import threading


from memory_profiler import profile # for memory profiling, since I was having memory issues with the execute_all() function
from memory_profiler import memory_usage # to check memory usage within a function, esp. for the for-loop in my GA.
import gc # for garbage collection; to force garbage collection after each run of the genetic algorithm

# Ignore specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")



# ================================================ #
#               MY CLASS DEFINITIONS    
# ================================================ #


# The user can also specify the number of strings to generate and the length of each string.
# Strings are generated in the initialization of the class and stored in a list.
# The user can also specify the alphabet and the frequency of a character in the alphabet.


class ClosestStringProblem():                                                                              
    def __init__(self, 
                name = "",                                  # Can be left empty. When loading file, takes name of file.
                num_strings = 10,                          # number of strings and length of strings determines the problem size.
                string_length = 100, 
                alphabet = set(['A', 'T', 'G', 'C']),       # The alphabet. Default is the DNA alphabet.
                weighted_chars = set(['G', 'C']),           # Characters to have different weights in the alphabet.
                character_frequency = 0.50,
                initialize_solution_population = True,
                population_size = 100,                      # The size of the population of SolutionString objects.
                load_file = None,                           # If we want to load a problem from a file, we can specify the file here.

                # The following parameters are for the genetic algorithm!   
                snapshot_interval = 10,                   # For saving additional figures for a run at the specified interval, e.g., every 1000 generations.
                mutation_rate = 1,
                matchsum_threshold = 0.50,                  # The mutation rate for the genetic algorithm.
                parents = 0.5,                              # The proportion of parents to keep in the genetic algorithm; parents get to reproduce.
                ):



        # The following parameters must be set by the user, even if load file is used.
        self._snapshot_interval = snapshot_interval
        self._mutation_rate = mutation_rate
        self._matchsum_threshold = matchsum_threshold
        self._parents_proportion = parents


        if load_file:
            # If we have a valid file to load, then we don't need to initialize the rest of the parameters.
            # Check if the file exists.
            if not os.path.exists(load_file):
                raise FileNotFoundError(f"File {load_file} not found.")
            
            self._name = load_file
            self.load_problem(load_file)

                    
            if initialize_solution_population:
                self._population_size = population_size
                self._solution_population = self.initialize_solution_population()
            return

        # The following is for manual instantiation of the class, if no file is loaded.

        if not isinstance(alphabet, set):
            raise ValueError("Alphabet must be a set") 

        self._name = name
        self._num_strings = num_strings
        self._string_length = string_length

        # Following attributes are used to generate the strings.
        self._alphabet = alphabet
        self._weighted_chars = weighted_chars
        self._character_frequency = character_frequency
        self._strings = self.generate_strings()

        # The following parameters are set by the class, and can be overridden by the user.
        if initialize_solution_population:
            self._population_size = population_size
            self._solution_population = self.initialize_solution_population()

    def __del__(self):
        # Delete the SolutionString objects in the population.
        del self._solution_population
        del self._strings
        gc.collect()


    def write_problem(self, tag):
        # Write the problem data to a file. I'll use an extension called '.csp' for Closest String Problem.
        # Sort alphabet by converting from set to list, to ensure consistent ordering of letters in filename.
        filename = 'Random' + ''.join(sorted(self.alphabet)) + f'_{self.num_strings}' + f'_{self.string_length}' + tag + '.csp'

        with open(filename, 'w') as f:
            f.write(f"Name: {filename}\n")                                  
            f.write(f"Number of Strings: {self.num_strings}\n")
            f.write(f"String Length: {self.string_length}\n")
            f.write(f"Alphabet: {'-'.join(self.alphabet)}\n")
            f.write(f"Weighted Chars: {'-'.join(self.weighted_chars)}\n")
            f.write(f"Character Frequency: {self.character_frequency}\n")
            for string in self.strings: # Each string gets its own line.
                f.write(f"{string}\n")


    def load_problem(self, filename):
        # Load the problem data from a file.
        # We'll need to parse the string.
        with open(filename, 'r') as f:
            # read all the lines, which we'll access via indexing
            lines = f.readlines()

            # Read first line, split it into substrings via ':', get second substring, and strip any whitespace.
            # Rinse and repeat...
            self._name = lines[0].split(":")[1].strip()

            self._num_strings = int(lines[1].split(":")[1].strip())
            self._string_length = int(lines[2].split(":")[1].strip())

            # For the alphabet, split by ':', and the letters are separated by '-'
            # Next, split by '-' to get the individual letters...
            self._alphabet = set(lines[3].split(":")[1].strip().split("-"))
            # Do the same thing for the weighted_chars set
            self._weighted_chars = set(lines[4].split(":")[1].strip().split('-'))

            self._character_frequency = float(lines[5].split(":")[1].strip())
            self._strings = [line.strip() for line in lines[6:]]
        

    def initialize_solution_population(self):
        # Initialize a population of SolutionString objects.
        # The population size is defined by the user.
        return [SolutionString(id = f"sol_{i}",
                            length = self.string_length,
                            alphabet = self.alphabet, 
                            random_init = True) 
                            for i in range(self._population_size)
                            ]

    def reset_population(self):
        # Reset the population of SolutionString objects.
        # This is useful when we want to run the genetic algorithm again.
        self._solution_population = self.initialize_solution_population()

    def evaluate_population_fitness(self, generation):
        # Given a generation, update the fitness of each SolutionString in the population.
        for solution in self._solution_population:
            solution.evaluate_fitness(self, generation)

    def update_population_heatmaps(self, generation):
        # Given a generation, update the heatmap of each SolutionString in the population.
        for solution in self.solution_population:
            solution.update_heatmap(self, generation)

    def select_best_solution(self, by_average = True):
        # Select the best solution from the population.
        if not by_average:
            return min(self.solution_population, key = lambda x: x._max_hamming_distance)
        else:
            return min(self.solution_population)

    def select_worst_solution(self):
        # Select the worst solution from the population.
        return max(self.solution_population)

    def select_parents(self):
        # Select parents from the population.
        # The number of parents is determined by the parents_proportion parameter.
        # The parents_proportion parameter is the proportion of parents to keep in the genetic algorithm.
        num_parents = int(self._parents_proportion * self._population_size)

        return copy.deepcopy(sorted(self.solution_population)[:num_parents]) # These can be added to the next generation.

    def reproduce(self, parents, generation = 0):
        # Reproduce the parents to create offspring.
        # The offspring are created by crossing over the parents.
        # The offspring are then mutated based on the mutation rate.
        # Let's keep the parents and create offspring until we reach the population size.
        # Also, we'll keep it simple. The best parent gets to reproduce randomly with any other parent.
        # NOTE: Mutation is handled in the SolutionString class, and it is employed in the reproduce() method.

        self._solution_population = parents

    

        # Make a copy before loop to avoid random type conversions of SolutionString to string
        parents_copy = copy.deepcopy(parents)

        while len(self._solution_population) < self._population_size:

            # Get the first parent randomly
            first_parent_index = random.sample(range(len(parents_copy)), 1)[0]
            first_parent = parents_copy[first_parent_index]

            # Returns list, so we just need the first element.
            other_parent_index = random.sample(range(len(parents_copy)), 1)[0]
            # Don't grab the same parent...
            while other_parent_index == first_parent_index:
                other_parent_index = random.sample(range(len(parents_copy)), 1)[0]

            other_parent = parents_copy[other_parent_index]

            # Crossover produces 8 possible offspring, but we only return 2 at random.

            offspring1, offspring2 = self.crossover(first_parent, other_parent)

            del first_parent
            del other_parent
  
            self._solution_population.append(offspring1)
            self._solution_population.append(offspring2)


        # Update heatmap data, which updates the match sums as well.
        self.update_population_heatmaps(generation)

        # Now, I mutate the population, except the parents
        self.mutate_population(start_index = len(parents))

        del parents_copy


    def mutate_population(self, start_index = 0):
        '''
        Mutations focused on the lowest match sums in the heatmap.
        '''
        for solution in self._solution_population[start_index:]:
            solution.mutate(self._mutation_rate, self._matchsum_threshold)


    def crossover(self, parent1, parent2):
        '''
        Unlike the TSP problem, since we can have repeated letters in our parent strings, we can simply cut the parents in half (to get 'gametes').
        Next, we cross over each half.
        '''
        # Get the gametes
        parent1_gamete1 = parent1.string[:parent1.length // 2]
        parent1_gamete2 = parent1.string[parent1.length // 2:]


        parent2_gamete1 = parent2.string[:parent2.length // 2]
        parent2_gamete2 = parent2.string[parent2.length // 2:]

        # Cross over the gametes and get all possible offspring
        # I arranged these so that the offspring are in order of the parents, so its easy to conceptualize.
        offspring_string1 = parent1_gamete1 + parent2_gamete2
        offspring_string3 = parent1_gamete2 + parent2_gamete1
        offspring_string5 = parent1_gamete1 + parent2_gamete1
        offspring_string7 = parent1_gamete2 + parent2_gamete2

        offspring_string8 = parent2_gamete2 + parent1_gamete2
        offspring_string6 = parent2_gamete1 + parent1_gamete1
        offspring_string4 = parent2_gamete1 + parent1_gamete2
        offspring_string2 = parent2_gamete2 + parent1_gamete1

        # Create the offspring SolutionString objects
        offspring = []
        for offspring_string in [offspring_string1, offspring_string2, offspring_string3, offspring_string4, offspring_string5, offspring_string6, offspring_string7, offspring_string8]:
            offspring.append(SolutionString(string = offspring_string, length = parent1.length, alphabet = parent1.alphabet))

        # Sort offspring by fitness
        offspring = sorted(offspring)

        # Return two random offspring
        offspring1, offspring2 = random.sample(offspring, 2)

        # Delete the offspring list
        for off in offspring:
            del off
        del offspring
    

        return offspring1, offspring2


    @property
    def solution_population(self):
        return self._solution_population

    @property
    def name(self):
        return self._name
    
    @property
    def num_strings(self):
        return self._num_strings
    
    @property
    def string_length(self):
        return self._string_length
    
    @property
    def alphabet(self):
        return self._alphabet   

    @property
    def weighted_chars(self):
        return self._weighted_chars

    @property
    def character_frequency(self):
        return self._character_frequency

    @property
    def strings(self):
        return self._strings

    @property
    def string_list(self):
        yield from self.strings

    def generate_strings(self):
        # Generates a list of strings based on the parameters of the class.
        # Calls a private method to generate a single string.
        strings = []
        for _ in range(self.num_strings):
            string = self._generate_single_string()
            strings.append(string)
        return strings
    
    def _generate_single_string(self):
        # Define the alphabet (our choices for string characters) and the weights for each character in the alphabet.
        alphabet = list(self.alphabet)

        # Weights for character selection is defined here.
        # A weight for G and C in alphabet ATGC, for example, of 0.5, after normalization, will be 0.25, which is uniform.
        weights = [1 - self._character_frequency if char not in self._weighted_chars else self._character_frequency for char in alphabet]

        total_weight = sum(weights)

        # Normalize the weights so that they all sum to 1.
        normalized_weights = [weight / total_weight for weight in weights]

        # Generate a string of length self.string_length by randomly selecting characters from the alphabet.
        return ''.join(random.choices(alphabet, weights=normalized_weights, k=self.string_length))


###################
# HittingStringProblem, which I MAY not use in this project...
###################


class HittingStringProblem(ClosestStringProblem):
    '''
    A class to represent a hitting string problem.
    The problem is defined by a list of strings of a certain length.
    The strings are generated randomly based on the alphabet and the wildcard ('*' character) frequency.
    The wildcard coverage can be reduced using the reduce_wildcard_coverage method.
    Note: Solutions for N strings of length L is trivial if N = L, since a solution can be obtained by the diagonals in a matrix of the strings.
        Example: If N = L = 3, then the strings ['100', '010', '001'] will have a wildcard coverage of 0%.
        But a solution string of L = 3 that has a positional match for each string in the list is ['111'].

    Note: Solutions for a population with 100% wildcard coverage is also trivial, since ANY string of length L will be a solution.
        Example: If N = 3, L = 3, then the strings ['*01', '1*1', '10*'] will have a wildcard coverage of 100%, with coverage string ***.
    '''

    # Initialize the parent class with the same parameters.
    def __init__(self, name = "", 
                 num_strings = 100000, 
                 string_length = 10, 
                 alphabet = set(['0', '1', '*']),
                 wild_card_freq = 0.0, 
                 initialize_solution_population = True):

        super().__init__(name, num_strings, string_length, alphabet, initialize_solution_population)
        self._wild_card_freq = wild_card_freq
        self._strings = self.generate_strings()

    @property
    def wild_card_freq(self):
        return self._wild_card_freq
    
    @property
    def wildcard_coverage_string(self):
        return self.calculate_coverage(self.strings)
    
    @property
    def wildcard_coverage_percent(self):
        return self.calculate_coverage_percentage()


    def calculate_coverage_percentage(self):
        coverage_string = self.calculate_coverage(self.strings)
        if not coverage_string:
            return 0.0
        
        total_positions = len(coverage_string)

        wildcard_positions = coverage_string.count('*')
        
        return (wildcard_positions / total_positions) * 100


    def calculate_coverage(self, strings):
        '''
        Given a list of strings, calculate the coverage string, which is a string of the same length as the strings.
        If a string in the list has an asterick at the kth position, then the coverage string is assigned '*' at the kth position.
        Otherwise, it's assigned '0'.
        '''

        if not strings:
            return ""
        
        string_length = len(strings[0])

        # Initialize the coverage string with all 0's! We will iterate through every character in our list of strings...
        # ... and assign the ith character of the string to the ith char of the coverage string IF it is a wildcard '*'.   
        coverage = ['0'] * string_length
        
        for string in strings:
            for i, char in enumerate(string):
                if char == '*':
                    coverage[i] = '*'
        
        return ''.join(coverage)


    def reduce_wildcard_coverage(self, target_percentage):
            '''
            Uses the coverage string to target wildcard positions for replacement.
            The aim is to reduce the wildcard coverage to a certain percentage.
            '''
            # For constraining the wildcard coverage to a certain percentage, after the problem strings are generated.
            # The coverage string is used as a reference for targeting the wildcard positions.
            current_coverage = self.calculate_coverage(self.strings)

            # If the current coverage is already less than the target percentage, then we don't need to do anything.
            current_percentage = self.calculate_coverage_percentage()
            
            if current_percentage <= target_percentage:
                return
            
            total_positions = len(current_coverage)

            # Define the upper bound of wildcard characters that we will allow in the coverage string.
            target_wildcards = int((target_percentage / 100) * total_positions)

            # Get the current number of wildcard positions.
            current_wildcards = current_coverage.count('*')
            
            # For as long as we don't have our required limit on coverage...
            while current_wildcards > target_wildcards:

                # For each ith position in the coverage string...
                for i in range(total_positions):
                    # If the ith position is a wildcard...
                    if current_coverage[i] == '*':
                        # For each string in the list of strings...
                        for string in self._strings:
                            # If the character at the ith position is a wildcard...
                            if string[i] == '*':
                                # Replace the character at the ith position with a random choice of 0 or 1.
                                # First convert string to a list so that we can change the ith position inplace.
                                editable_string = list(string)
                                editable_string[i] = str(random.choice([0, 1]))

                                # Join the list back into a string and replace the original string in the list of strings.
                                new_string = ''.join(editable_string)

                                # Re assign the new string to the list of strings.
                                self._strings[self._strings.index(string)] = new_string

                                break
                        # Recalculate the coverage and the number of wildcard positions.
                        current_coverage = self.calculate_coverage(self.strings)
                        current_wildcards = current_coverage.count('*')

                        # If we meet the goal, then break out of the loop.
                        # Note that the condition to reach the break will also break our while-loop
                        if current_wildcards <= target_wildcards:
                            break



    ####################
    # Helper Methods
    ####################
    # The following methods are 'helpers' insofar as they are not meant to be called by the user.
    # These set-up the problem or prvovide additional functionality that is reused in the algorithm.
    # The initial focus on algorithms will be on developing a GA and GA + WoC, so the initial solution will be a list of strings.

    def generate_strings(self):
        # Generates a list of strings based on the parameters of the class.
        # Calls a private method to generate a single string.
        strings = []
        for _ in range(self.num_strings):
            string = self._generate_single_string()
            strings.append(string)
        return strings
    

    
    def _generate_single_string(self):
        # Define the alphabet (our choices for string characters) and the weights for each character in the alphabet.
        alphabet = list(self.alphabet)

        # Weights for character selection is defined here.
        weights = [1 - self.wild_card_freq if char != '*' else self.wild_card_freq for char in alphabet]

        total_weight = sum(weights)

        # Normalize the weights so that they all sum to 1.
        normalized_weights = [weight / total_weight for weight in weights]

        # Generate a string of length self.string_length by randomly selecting characters from the alphabet.
        return ''.join(random.choices(alphabet, weights=normalized_weights, k=self.string_length))






############################################################################################################
# The SolutionString class is a class that represents a single solution string in the population (in context of genetic algorithm).
# A particular SolutionString object may be updated by passing a set of strings that represents the HittingStringProblem.
# Data attributes in this object include:
# - id: A unique identifier for the string.
# - string: The binary string that represents the solution.
# - length: The length of the binary string.
# - alphabet: The set of characters that can be used in the string.
# - random_init: A boolean flag to determine if the string should be randomly initialized.
# - total_hamming_distance: The Hamming distance between the solution string and the target string.
# - hit_percent: The percentage of strings in the population that the solution string hits at any position.
#
############################################################################################################


class SolutionString:
    '''
    This class represents a single solution string in the population (in context of genetic algorithm).
    A particular SolutionString object may be updated by passing a set of strings that represents the HittingStringProblem.
    Data attributes in this object include:
    - id: A unique identifier for the string.
    - string: The binary string that represents the solution.
    - length: The length of the binary string.
    - alphabet: The set of characters that can be used in the string.
    - random_init: A boolean flag to determine if the string should be randomly initialized.
    - total_hamming_distance: The Hamming distance between the solution string and the target string.
    - hit_percent: The percentage of strings in the population that the solution string hits at any position.

    NOTE: The SolutionString object should have its data saved to a DataFrame before updating the solution string.
    NOTE: Either the Hamming distance of Percent hits can be used as a fitness function for the genetic algorithm.
    '''


    def __init__(self,
                # The following are default values for the SolutionString constructor. 
                id = "na", 
                string = "", 
                length = 0, 
                alphabet = set(['A', 'T', 'G', 'C']), 
                random_init = False,
                total_hamming_distance = 0
                ):

        self._id = id
        self._string = list(string)
        self._length = length
        self._alphabet = alphabet
        self._random_init = random_init

        self._generation = 0

        # Initialize the fitness attributes to 0. These will be updated when the solution string is evaluated.
        self._hamming_dictionary = {}
        self._lambda = 0
        self._total_hamming_distance = total_hamming_distance
        self._max_hamming_distance = length
        self._min_hamming_distance = 0
        self._hit_percent = 0
    
        if random_init:
            self._string = self.generate_random_string()

        # Initialize the heatmap and match sums attributes.
        self._heatmap = None        # Deprecated! Use the match sums instead. I decided this data is not needed... And this is a lot of data that's unused.
        self._match_sums = None     # The match sums are used to determine the positions with the lowest match sums for mutation.

    def __del__(self):
        del self._string
        del self._alphabet
        del self._hamming_dictionary
        del self._heatmap
        del self._match_sums
        gc.collect()

    def generate_random_string(self):
        alphabet = self._alphabet
        return random.choices(list(alphabet), k = self._length)

    def evaluate_fitness(self, problem: ClosestStringProblem, generation = 0):
        '''
        Given an HSP problem, update the solution string's data with respect to the target strings.
        This should be done for each SolutionString at every generation.
        NOTE: I evaluate fitness using the lambda parameter. Hence, comparators will be based on the lambda value.
        '''
        self._generation = generation
        target_strings = problem.strings

        # Calculate the total Hamming distance between the solution string and the target strings.
        self._hamming_dictionary = {target_string: self.hamming_distance(target_string) for target_string in target_strings}

        # Count unique Hamming distances
        self._hamming_counts = Counter(self._hamming_dictionary.values())

        # Calculate the weighted sum of the Hamming distances
        weighted_sum = sum(hamming_distance * count for hamming_distance, count in self._hamming_counts.items())

        # Calculate the total count of occurrences
        total_count = sum(self._hamming_counts.values())

        # Compute the lambda
        self._lambda = weighted_sum / total_count

        # Just in case, we'll also store the total hamming distance.
        # Compute total Hamming distance
        self._total_hamming_distance = sum(self._hamming_dictionary.values())

        # Compute the maximum Hamming distance
        self._max_hamming_distance = max(self._hamming_dictionary.values())
        # And compute the min
        self._min_hamming_distance = min(self._hamming_dictionary.values())

        del target_strings



    def plot_smoothed_hamming_distribution(self, generation, lower_x = None, upper_x = None):
        # Extract Hamming distances and their counts from self._hamming_dictionary
        hamming_distances = np.array(list(self._hamming_counts.keys()))
        counts = np.array(list(self._hamming_counts.values()))

        # Sort the data for smooth plotting
        sorted_indices = np.argsort(hamming_distances)
        hamming_distances = hamming_distances[sorted_indices]
        counts = counts[sorted_indices]

        # Create a smooth curve
        x_new = np.linspace(hamming_distances.min(), hamming_distances.max(), 300)
        spl = make_interp_spline(hamming_distances, counts, k=3)  # k=3 for cubic spline
        counts_smooth = spl(x_new)

        # Plot the smooth curve
        plt.figure(figsize=(10, 6))
        plt.plot(x_new, counts_smooth, color='blue', label='Smoothed Line')

        # In case we want to fix the x-axis limits
        if lower_x and upper_x:
            plt.xlim(lower_x, upper_x)

        # Add labels and title
        plt.xlabel('Hamming Distance')
        plt.ylabel('Count')
        plt.title(f'Smoothed Hamming Distance Distribution, Generation {generation}')
        plt.legend()

    def plot_hamming_distribution(self, generation, lower_x = None, upper_x = None):
        # Extract Hamming distances and their counts from self._hamming_dictionary
        hamming_distances = np.array(list(self._hamming_counts.keys()))
        counts = np.array(list(self._hamming_counts.values()))

        # Sort the data for plotting
        sorted_indices = np.argsort(hamming_distances)
        hamming_distances = hamming_distances[sorted_indices]
        counts = counts[sorted_indices]

        # Plot the distribution
        plt.figure(figsize=(10, 6))
        plt.plot(hamming_distances, counts, color='blue', marker='o', linestyle='-', label='Hamming Distance Distribution')

        # In case we want to fix the x-axis limits
        if lower_x and upper_x:
            plt.xlim(lower_x, upper_x)

        # Add labels and title
        plt.xlabel('Hamming Distance')
        plt.ylabel('Count')
        plt.title(f'Hamming Distance Distribution, Generation {generation}')
        plt.legend()


    def evaluate_hit_percent(self, problem: ClosestStringProblem, generation = 0):
        self._generation = generation
        
        # Calculate the percentage of strings in the population that the solution string hits at any position.
        self._hit_percent = float(sum([1 for target_string in target_strings if self.hit_check(target_string)]) / len(target_strings)) * 100


    def update_heatmap(self, problem: ClosestStringProblem, generation = 0):
        self._generation = generation

        if not self._match_sums:
            self._match_sums = self.calculate_match_sums(problem.strings)
  
    
    def calculate_match_sums(self, csp_instance_strings):
        # Counts the number of matches per kth position in the solution string among the problem strings.
        match_sums = [0] * len(self._string)


        for string in csp_instance_strings:
            for i in range(len(self._string)):
                if self._string[i] == string[i]:
                    match_sums[i] += 1

        return match_sums


    def mutate(self, mutation_rate, match_sum_threshold = 0.1):
        '''
        Mutate the solution string based on the mutation rate, targeting areas with low match-sums.
        '''

        # Get the bottom 10% of the match sums
        bottom_threshold = int(match_sum_threshold * len(self._string))

        # Get the indices of the bottom 10% of the match sums
        bottom_indices = np.argsort(self._match_sums)[:bottom_threshold]

        # Mutate the solution string based on the mutation rate
        # Select a random index from bottom_indices: this is the position to mutate
        # Mutate the position with a random character from the alphabet

        index = random.choice(bottom_indices)

        # Print what we're mutating and the match sum associated with that position
        print(f"Mutating position {index} with match sum {self._match_sums[index]}")

        if random.random() < mutation_rate:
            # Get a possible mutation letter...
            possible_mutation = random.choice(list(self._alphabet))
            # ... and make sure it's not the same as the current letter, so we guarantee a mutation happens if the probability condition is met.
            while possible_mutation == self._string[index]:
                possible_mutation = random.choice(list(self._alphabet))
            # Now we can mutate the string at the index.
            self[index] = random.choice(list(self._alphabet))



    def heatmap_to_dataframe(self):
        if not self._heatmap:
            raise ValueError("Heatmap has not been generated yet.")

        # Create a list of dictionaries to hold the data for the DataFrame
        data = []
        for i, row in enumerate(self._heatmap):
            row_data = {'index': i, 'generation': self._generation}
            row_data.update({f'pos_{j}': value for j, value in enumerate(row)})
            data.append(row_data)

        # Create the DataFrame
        df = pd.DataFrame(data)
        return df


    #################
    # Data Saving Methods
    #################
    # The append_heatmap_data method does not take generation as argument since the update

    def append_heatmap_data(self, tag, experiment_name = 'vanilla_experiment', run_number = 0):
        df = self.heatmap_to_dataframe()

        filename = f"results/{experiment_name}/run_{run_number}/{tag}_heatmap_data.csv"

        # Make the experiment name directory, if it doesn't exist
        if not os.path.exists(f'results/{experiment_name}/run_{run_number}'):
            os.mkdir(f'results/{experiment_name}/run_{run_number}')

        file_exists = os.path.exists(filename)

        if not file_exists:
            df.to_csv(filename, mode='w', header=not file_exists, index=False)
        else:
            df.to_csv(filename, mode='a', header=not file_exists, index=False)
        del df

    def append_fitness_data(self, generation, tag, experiment_name = 'vanilla_experiment', run_number = 0):
        data = {
            'generation': [generation],
            'total_hamming_distance': [self._total_hamming_distance],
            'lambda(average)': [self._lambda],
            'max_hamming_distance': [self._max_hamming_distance],
            'min_hamming_distance': [self._min_hamming_distance],
            'solution_string': [''.join(self._string)],
        }

        df = pd.DataFrame(data)

        filename = f"results/{experiment_name}/run_{run_number}/{tag}_fitness_data.csv"

        # Make the experiment name directory, if it doesn't exist
        if not os.path.exists(f'results/{experiment_name}/run_{run_number}'):
            os.mkdir(f'results/{experiment_name}/run_{run_number}')
 

        # Append the fitness data to the file, if it exists. Otherwise, create the file.
        file_exists = os.path.exists(filename)

        if not file_exists:
            df.to_csv(filename, mode='w', header=not file_exists, index=False)
        else:
            df.to_csv(filename, mode='a', header=not file_exists, index=False)
        del df


    def append_run_data(self, tag = 'run_solutions', experiment_name = 'vanilla_experiment', run_number = 0, run_time = 0):
        data = {
            'run_number': [run_number],
            'total_hamming_distance': [self._total_hamming_distance],
            'lambda(average)': [self._lambda],
            'max_hamming_distance': [self._max_hamming_distance],
            'min_hamming_distance': [self._min_hamming_distance],
            'runtime': [run_time],
            'solution_string': [''.join(self._string)],
        }

        df = pd.DataFrame(data)

        filename = f"results/{experiment_name}/{tag}_fitness_data.csv"

        # Append the fitness data to the file, if it exists. Otherwise, create the file.
        file_exists = os.path.exists(filename)

        if not file_exists:
            df.to_csv(filename, mode='w', header=not file_exists, index=False)
        else:
            df.to_csv(filename, mode='a', header=not file_exists, index=False)

    #################
    # Helper methods for the update_string_data method.
    #################


    def hamming_distance(self, target_string):
        '''
        Given a target string, calculates the Hamming distance between the solution string and the target string.
        '''
        if len(target_string) != self._length:
            raise ValueError("Target string must be of the same length as the solution string when computing Hamming distance.")
        
        return sum([1 for i in range(self._length) if self._string[i] != target_string[i]])
    
    def hit_percent(self, target_strings):
        '''
        Given a list of target strings, calculates the percentage of strings in the population that the solution string hits at any position.
        '''
        
    def hit_check(self, target_string):
        # This is just to check if the solution string hits any of the target strings at any kth position.
        for i in range(self._length):
            if self._string[i] == target_string[i]:
                return True

        return False

    #################
    # Properties
    #################
        
    @property
    def id(self):
        return self._id

    @property
    def string(self):
        return ''.join(self._string)

    @property
    def length(self):
        return self._length


    ################
    # All things Hamming
    ################
    @property
    def hamming_dictionary(self):
        return self._hamming_dictionary

    @property
    def hamming_counts(self):
        return self._hamming_counts

    @property
    def total_hamming_distance(self):
        return self._total_hamming_distance

    @property                   # Our main fitness metric!
    def lambda_value(self):
        return self._lambda

    @property
    def max_hamming_distance(self):
        return self._max_hamming_distance

    @property
    def min_hamming_distance(self):
        return self._min_hamming_distance

    @property
    def fitness(self):
        return self._lambda, self._total_hamming_distance, self._min_hamming_distance, self._max_hamming_distance

    #####################   
    @property
    def hit_percent(self):
        return self._hit_percent

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def random_init(self):
        return self._random_init

    @property
    def heatmap(self):
        return self._heatmap

    @property
    def match_sums(self):
        return self._match_sums

    @property
    def generation(self):
        return self._generation

    
    def __setitem__(self, index, value):
        # This is to allow for setting of the mutable character list that represents the 'string'.
        # This will facilitate using mutation operators in the genetic algorithm.
        self.string[index] = value

    def __getitem__(self, index):
        print("Getting item")
        return self._string[index]

    def __str__(self):
        return self.string

    def __repr__(self):

        return self.string

    #################
    # COMPARISON OPERATORS
    #################
    def __eq__(self, other):
        # For comparing two SolutionString objects.
        if isinstance(other, SolutionString):
            return self._lambda == other._lambda
        return False

    def __ne__(self, other):
        # For comparing two SolutionString objects.
        return not self.__eq__(other)  

    def __lt__(self, other):
        # For comparing two SolutionString objects via fitness, for less-than
        if isinstance(other, SolutionString):
            return self._lambda < other._lambda
        return False

    def __le__(self, other):
        # For comparing two SolutionString objects via fitness, for less-than or equal
        if isinstance(other, SolutionString):
            return self._lambda <= other._lambda
        return False

    def __gt__(self, other):
        # For comparing two SolutionString objects via fitness. Greater than...
        if isinstance(other, SolutionString):
            return self._lambda > other._lambda
        return False

    def __ge__(self, other):
        # For comparing two SolutionString objects via fitness. Greater than or equal
        if isinstance(other, SolutionString):
            return self._lambda >= other._lambda
        return False

    #################
    # Plotting Functions
    #################

    def plot_match_sums(self, generation = 0):

        data = self._match_sums

        if not data or not isinstance(data, list):
            raise ValueError("Generated match sums data is not a list")

        # Create the bar plot
        plt.figure(figsize=(20, 10))
        sns.barplot(x=list(range(len(data))), y=data, color='blue')

        # Add labels and title
        plt.xlabel('Position in String')
        plt.ylabel('Sum of Matches')
        plt.title(f'Sum of Matches at Each Position, Generation {generation}')
        del data

    def plot_heatmap(self):
        """
        Plots a heatmap from a list of lists of binary values (0 and 1).

        Parameters:
        data (list of lists): The binary data to plot as a heatmap.
        """
        # Create the heatmap
        sns.heatmap(self._heatmap, cmap="YlGnBu", cbar=False, linewidths=.1, linecolor='black')

        # Add labels and title
        plt.xlabel('Position in String')
        plt.ylabel('String Index')
        plt.title('String Heatmap')

        # Display the heatmap

##############################
# GUI for running GA
##############################


def ga_gui():


    def execute_all():
        num_runs = int(num_runs_entry.get())
        num_generations = int(num_generations_entry.get())
        snapshot_interval = int(snapshot_interval_entry.get())
        mutation_rate = float(mutation_rate_entry.get())
        matchsum_threshold = float(matchsum_threshold_entry.get())
        parents = float(parents_entry.get())

        print(f"num_runs: {num_runs}")
        print(f"num_generations: {num_generations}")
        print(f"snapshot_interval: {snapshot_interval}")
        print(f"mutation_rate: {mutation_rate}")
        print(f"matchsum_threshold: {matchsum_threshold}")
        print(f"parents: {parents}")

        # make results directory
        if not os.path.exists("results"):
            os.mkdir("results")


        # iterate through all .csp instances in the data directory
        # Run experiment series for each .csp instance
        for filename in os.listdir('data'):
            if filename.endswith('.csp'):
                print(f"Running experiment series for {filename}...")
    
                # And for each .csp instance, run the experiment series for the number of runs specified.
                for run_number in range(num_runs):
                    print(f"Run number: {run_number}")

                    _ = time.perf_counter()
                    solution = genetic_algorithm(filename, 
                                num_generations, 
                                snapshot_interval, 
                                mutation_rate, 
                                matchsum_threshold,
                                parents, 
                                run_number)
                    runtime = time.perf_counter() - _

                    print(f"Run {run_number} completed in {runtime} seconds.")
                    # Save the data of the best solution for this run.
                    # Note that this is a separate method that appends the data to a CSV file in the experiment directory.
                    solution.append_run_data(experiment_name = filename.removesuffix('.csp'), run_number = run_number, run_time = runtime)
                    
                    # See if we can free up memory by deleting the solution and forcing garbage collection...
                    del solution
                    gc.collect()


    def execute_single():

        # Get all the same params as execute_all...
        num_runs = int(num_runs_entry.get())
        num_generations = int(num_generations_entry.get())
        snapshot_interval = int(snapshot_interval_entry.get())
        mutation_rate = float(mutation_rate_entry.get())
        matchsum_threshold = float(matchsum_threshold_entry.get())
        parents = float(parents_entry.get())

        # ... but also get the filename from the user.
        filename = filename_entry.get()
        # graph x-axis limits
        lower_x = int(lower_x_entry.get())
        upper_x = int(upper_x_entry.get())

        print(f"num_runs: {num_runs}")
        print(f"num_generations: {num_generations}")
        print(f"snapshot_interval: {snapshot_interval}")
        print(f"mutation_rate: {mutation_rate}")
        print(f"matchsum_threshold: {matchsum_threshold}")
        print(f"parents %: {parents}")
        print(f"filename: {filename}")

        # make results directory
        if not os.path.exists("results"):
            os.mkdir("results")

        # For the given filename, run a single experiment series for the number of runs specified.
        for run_number in range(num_runs):

            _ = time.perf_counter()
            best_solution = genetic_algorithm(filename, 
                                num_generations, 
                                snapshot_interval, 
                                mutation_rate, 
                                matchsum_threshold, 
                                parents, 
                                run_number,
                                x_lower = lower_x,
                                x_upper = upper_x
                                )
            runtime = time.perf_counter() - _

            print(f"Run {run_number} completed in {runtime} seconds.")
            # Save the data of the best solution for this run

    #@profile # FOR MEMORY PROFILING LINE-BY-LINE
    def genetic_algorithm(filename,
                        num_generations, 
                        snapshot_interval, 
                        mutation_rate, 
                        matchsum_threshold, 
                        parents, 
                        run_number,              # For labelling runs in the experiment directory
                        x_lower = None,
                        x_upper = None
                        ):
        # Initially, I had some memory accumulation that was an issue.
        # It's been resolved by NOT storing the heatmap data and avoiding plots in the generation for-loop
        # Thus, I probably have an excessive inclusion of gc.collect() and del statements.
        # I'm keeping them in for now, but I may remove them later if they're not necessary.

        # We initialize the problem with the filename, and the parameters for the genetic algorithm.
        csp = ClosestStringProblem(initialize_solution_population = True,
                                    load_file = 'data/' + filename,
                                    snapshot_interval = snapshot_interval,
                                    mutation_rate = mutation_rate,
                                    matchsum_threshold = matchsum_threshold,
                                    parents = parents
                                    )
     
        experiment_name = filename.removesuffix('.csp')
        # Make experiment directory using experiment_name
        if not os.path.exists(f"results/{experiment_name}"):
            os.mkdir(f"results/{experiment_name}")

        # The problem comes with a population of solution strings, which we can access via the solution_population attribute.
        # But we have two methods: 1) for evaluating the population fitness and 2) for updating the heatmap of matches, from which we get match-sums.
        csp.evaluate_population_fitness(0)
        csp.update_population_heatmaps(0)

        ############################
        # We're keeping track of the best solution by minimizing the average Hamming distance.
        # One idea is to have as many strings with as low a Hamming distance as possible.
        # We admit that max Hamming distances may not be optimal if selection is based upon averages.
        ############################

        # At generation 0, best solution is the first one we see. 
        # Get the initial data for the best solution.
        # The default comparators are by average Hamming distance.

        best_by_average = csp.select_best_solution(by_average = True)

        # The heatmap data is again commented out for now. We don't need it for the main data.
        #
        #
        #best_by_average.append_heatmap_data("best_by_average",
        #                                    experiment_name = filename.removesuffix(".csp"))

        ############################
        # The following are our generations.
        ############################
        for i in range(num_generations + 1):
            print(f"Generation {i}")

            parents = copy.deepcopy(csp.select_parents())             # Parents selected after evaluation of fitness 

            csp.reproduce(parents, i)                                 # Parents reproduce up to population size. Handles mutation, too. Mutation targets bottom 10% match-sum locations, via point mutations.

            csp.evaluate_population_fitness(i)                        # After reproduction and mutation of offspring, evaluate fitness.

            # Find the best solution, if there is one, in this generation. Update overall best_by_average if necessary.
            this_gen_best_by_average = csp.select_best_solution(by_average = True)
            if this_gen_best_by_average < best_by_average:
                best_by_average = this_gen_best_by_average
    
            #################
            # Record Data
            #################
            # For now, I commented out the snapshots for the heatmap data and the plots!
            # The main data I need is the fitness data...
            #
            #
            # Next, record the heatmap data for the best solutions...
            #if (i % snapshot_interval == 0):   
             
                #best_by_average.append_heatmap_data("best_by_average",
                #                        experiment_name = filename.removesuffix(".csp"), 
                #                        run_number = run_number
                #                        )


                # These are some of our plots for the best solution by average Hamming distance.
                # One is the distribution of problem strings with a given Hamming distance to our solution.
                #best_by_average.plot_hamming_distribution(i, lower_x = x_lower, upper_x = x_upper)

                #plt.savefig(f"results/{experiment_name}/run_{run_number}/best_by_average_hamming_dist_gen_{i}.png")
                #plt.clf()
                #plt.close('all')
                #gc.collect()

                # The other is the heatmap of matches between our solution and the problem strings.
                #best_by_average.plot_match_sums(i)
                #plt.savefig(f"results/{experiment_name}/run_{run_number}/best_by_average_matchsum_gen_{i}.png")
                #plt.clf()
                #plt.close('all')
                #gc.collect()

            # Free up memory by deleting the parents deepcopy and forcing garbage collection.
            del parents
            gc.collect()

            best_by_average.append_fitness_data(i, "best_by_average", experiment_name = filename.removesuffix(".csp"), run_number = run_number)

        # delete the csp instance since we're done with it...
        del csp

        # return best solution (by average Hamming distance) for this run.
        return best_by_average
    # end of genetic_algorithm function


    #################################################
    # Main GUI
    #################################################

    # Create the main window
    root = tk.Tk()
    root.title("Genetic Algorithm Parameters")

    # Define a larger font! In case it's too small
    large_font = font.Font(size=24)

    # Labels and Entry fields for each parameter
    tk.Label(root, text="Number of Runs", font=large_font).grid(row=0, column=0, padx=10, pady=5)
    num_runs_entry = tk.Entry(root, font=large_font)
    num_runs_entry.insert(0, "50") # default is 100 runs
    num_runs_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text="Number of Generations", font=large_font).grid(row=1, column=0, padx=10, pady=5)
    num_generations_entry = tk.Entry(root, font=large_font)
    num_generations_entry.insert(0, "10") # default is 100 generations
    num_generations_entry.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text="Snapshot Interval", font=large_font).grid(row=2, column=0, padx=10, pady=5)
    snapshot_interval_entry = tk.Entry(root, font=large_font)
    snapshot_interval_entry.insert(0, "1") # default is 10 generations
    snapshot_interval_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text="Mutation Rate", font=large_font).grid(row=3, column=0, padx=10, pady=5)
    mutation_rate_entry = tk.Entry(root, font=large_font)
    mutation_rate_entry.insert(0, "1") # default is 1, which is 100%
    mutation_rate_entry.grid(row=3, column=1, padx=10, pady=5)

    tk.Label(root, text="Matchsum Threshold", font=large_font).grid(row=4, column=0, padx=10, pady=5)
    matchsum_threshold_entry = tk.Entry(root, font=large_font)
    matchsum_threshold_entry.insert(0, "0.1") # default is 0.1, which is 10% of the bottom matchsums
    matchsum_threshold_entry.grid(row=4, column=1, padx=10, pady=5)

    tk.Label(root, text="Percent to be Parents", font=large_font).grid(row=5, column=0, padx=10, pady=5)
    parents_entry = tk.Entry(root, font=large_font)
    parents_entry.insert(0, "0.5") # default is 0.5, so we take 50% of population to be parents
    parents_entry.grid(row=5, column=1, padx=10, pady=5)

    # Population size!
    tk.Label(root, text="Population Size", font=large_font).grid(row=5, column=3, padx=10, pady=5)
    population_entry = tk.Entry(root, font=large_font)
    population_entry.insert(0, "100") # default is 100 strings
    population_entry.grid(row=5, column=4, padx=10, pady=5)

    # Button to execute the GA, which will call the execute_all function; that iterates through all .csp instances in the data directory
    execute_button = tk.Button(root, text="Execute All", font=large_font, command=execute_all)
    execute_button.grid(row=6, column=0, columnspan=2, pady=10)

    # Get filename for single runs
    tk.Label(root, text="Filename", font=large_font).grid(row=7, column=0, padx=10, pady=5)
    filename_entry = tk.Entry(root, font=large_font)
    filename_entry.grid(row=7, column=1, padx=10, pady=5)

    # Get lower_x for graph x_axis
    tk.Label(root, text="Lower X", font=large_font).grid(row=8, column=0, padx=10, pady=5)
    lower_x_entry = tk.Entry(root, font=large_font)
    lower_x_entry.insert(0, "0") # default is 0
    lower_x_entry.grid(row=8, column=1, padx=10, pady=5)

    # Get upper_x for graph x_axis
    tk.Label(root, text="Upper X", font=large_font).grid(row=9, column=0, padx=10, pady=5)
    upper_x_entry = tk.Entry(root, font=large_font)
    upper_x_entry.insert(0, "500") # default is 500
    upper_x_entry.grid(row=9, column=1, padx=10, pady=5)

    # Button to execute the GA, which will call the execute_single function; that runs the GA for a single .csp instance
    execute_single_button = tk.Button(root, text="Execute Single", font=large_font, command=execute_single)
    execute_single_button.grid(row=10, column=0, columnspan=2, pady=10)


    # Start the Tkinter event loop
    root.mainloop()