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
import threading


# Ignore specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")



# ================================================ #
#               MY CLASS DEFINITIONS    
# ================================================ #




# The HittingStringProblem class is a class that generates a list of strings of a certain length.
# Strings are composed of the alphabet, which is a set of characters. The user can specify the frequency that the wildcard character '*' appears in the strings.
# The user can also specify the number of strings to generate and the length of each string.
# Strings are generated in the initialization of the class and stored in a list.
class ClosestStringProblem():                                                                              
    def __init__(self, 
                name = "", 
                num_strings = 100, 
                string_length = 10, 
                alphabet = set(['A', 'T', 'G', 'C']),
                weighted_chars = set(['G', 'C']),
                character_frequency = 0.50,
                initialize_solution_population = True,
                population_size = 100):

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

        if initialize_solution_population:
            self._population_size = population_size
            self._solution_population = self.initialize_solution_population()
            
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
        for solution in self.solution_population:
            solution.evaluate_fitness(self, generation)

    def update_population_heatmaps(self, generation):
        # Given a generation, update the heatmap of each SolutionString in the population.
        for solution in self.solution_population:
            solution.update_heatmap(self, generation)

    def select_best_solution(self):
        # Select the best solution from the population.
        return min(self.solution_population)


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
        self._hit_percent = 0
    
        if random_init:
            self._string = self.generate_random_string()

        # Initialize the heatmap and match sums attributes.
        self._heatmap = None
        self._match_sums = None


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



    def plot_smoothed_hamming_distribution(self, generation):
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

        # Add labels and title
        plt.xlabel('Hamming Distance')
        plt.ylabel('Count')
        plt.title(f'Smoothed Hamming Distance Distribution, Generation {generation}')
        plt.legend()




    def evaluate_hit_percent(self, problem: ClosestStringProblem, generation = 0):
        self._generation = generation
        
        # Calculate the percentage of strings in the population that the solution string hits at any position.
        self._hit_percent = float(sum([1 for target_string in target_strings if self.hit_check(target_string)]) / len(target_strings)) * 100






    def update_heatmap(self, problem: ClosestStringProblem, generation = 0):
        self._generation = generation
        if not self._heatmap:
            self._heatmap = self.generate_heatmap(problem.strings)
        if not self._match_sums:
            self._match_sums = self.calculate_match_sums(problem.strings)
        
    def generate_heatmap(self, csp_instance_strings):
        heatmap = []

        for string in csp_instance_strings:
            # For each string, we use a list comprehension to generate a list of 1's and 0's,...
            #, ... and then we append this list to our matrix.
            heatmap.append(
                [1 if self._string[i] == string[i] else 0 
                for i in range(len(self._string))
                ]
                )

        return heatmap
    
    def calculate_match_sums(self, csp_instance_strings):
        # Counts the number of matches per kth position in the solution string among the problem strings.
        match_sums = [0] * len(self._string)


        for string in csp_instance_strings:
            for i in range(len(self._string)):
                if self._string[i] == string[i]:
                    match_sums[i] += 1

        return match_sums

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

    def append_heatmap_data(self, filename):
        df = self.heatmap_to_dataframe()

        file_exists = os.path.exists(filename)

        if not file_exists:
            df.to_csv(filename, mode='w', header=not file_exists, index=False)
        else:
            df.to_csv(filename, mode='a', header=not file_exists, index=False)

    def append_fitness_data(self, generation, filename):
        data = {
            'generation': [generation],
            'total_hamming_distance': [self._total_hamming_distance],
            'lambda': [self._lambda],
            'solution_string': [''.join(self._string)],
        }

        df = pd.DataFrame(data)

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
    def fitness(self):
        return self._lambda, self._total_hamming_distance

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


    def __getitem__(self, index):
        # This is to allow for indexing of the mutable character list that represents the 'string'.
        return self.string[index]
    
    def __setitem__(self, index, value):
        # This is to allow for setting of the mutable character list that represents the 'string'.
        # This will facilitate using mutation operators in the genetic algorithm.
        self.string[index] = value

    def __str__(self):
        # For easy printing of our string...
        return "".join(self.string)

    def __repr__(self):
        # For easy printing of our string...
        return "".join(self.string)


    #################
    # COMPARISON OPERATORS
    #################
    def __eq__(self, other):
        # For comparing two SolutionString objects.
        if isinstance(other, SolutionString):
            return self.string == other.string
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

    def plot_match_sums(self):

        data = self._match_sums

        if not data or not isinstance(data, list):
            raise ValueError("Generated match sums data is not a list")

        # Create the bar plot
        plt.figure(figsize=(20, 10))
        sns.barplot(x=list(range(len(data))), y=data, color='blue')

        # Add labels and title
        plt.xlabel('Position in String')
        plt.ylabel('Sum of Matches')
        plt.title('Sum of Matches at Each Position')


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



