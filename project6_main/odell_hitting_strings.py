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

import networkx as nx # for visualizing the solutions during wisdom of the crowds


import scipy.stats as stats     # for the Kruskal-Wallis test
import scikit_posthocs as sp    # for the post-hodc tests (Dunn's with Bonferoroni correction)
                                # Needs to be installed using pip:
                                # $ pip install scikit_posthocs
import os
import tkinter as tk            # for our simple GUI
import threading

# ================================================ #
#               MY CLASS DEFINITIONS    
# ================================================ #

# The HittingStringProblem class is a class that generates a list of strings of a certain length.
# Strings are composed of the alphabet, which is a set of characters. The user can specify the frequency that the wildcard character '*' appears in the strings.
# The user can also specify the number of strings to generate and the length of each string.
# Strings are generated in the initialization of the class and stored in a list.

class SolutionString():
    pass

class StringCluster():
    # A class to represents each string in our population as an n-dimensional sample with binary attributes
    # Each kth attribute is assigned 1 if it matches the solution string at the kth position, and 0 otherwise.
    # The string cluster object should be updated every time the solution string is updated.
    # Thus, be sure to record any data into a DataFrame BEFORE updating the solution string. 
    pass




class HittingStringProblem:
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
                                                                                               
    def __init__(self, name = "", 
                 num_strings = 100, 
                 string_length = 100, 
                 alphabet = set(['0', '1', '*']), 
                 wild_card_freq = 0.1, 
                 initialize_solution_population = True):
        

        if not isinstance(alphabet, set):
            raise ValueError("Alphabet must be a set")
    
        self._name = name
        self._num_strings = num_strings
        self._string_length = string_length
        self._alphabet = alphabet
        self._wild_card_freq = wild_card_freq

        self._strings = self.generate_strings()

        if initialize_solution_population:
            pass


    
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
    def wild_card_freq(self):
        return self._wild_card_freq
    
    @property
    def strings(self):
        return self._strings

    @property
    def string_list(self):
        yield from self.strings

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
                                # First convert string to a list so that we can change the ith position.
                                string_list = list(string)
                                string_list[i] = str(random.choice([0, 1]))

                                # Join the list back into a string and replace the original string in the list of strings.
                                new_string = ''.join(string_list)

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
