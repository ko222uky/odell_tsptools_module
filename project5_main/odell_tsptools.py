# ================================================ #
#               IMPORTED MODULES
# ================================================ #

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
import tkinter as tk        # for our simple GUI
import threading

# Python 3.12.4
# Matplotlib version 3.8.4

# ================================================ #
#               MY CLASS DEFINITIONS    
# ================================================ #

# ================================================ #
#              VERTEX CLASS DEFINITION
# ================================================ #
    
    # The Vertex class will store the label and coords of each point represented in the TSP problem
    # Methods for comparison (==, !=) and addition (binary and chaining) are implemented.

class Vertex:
    def __init__(self, label = "", x = 0, y = 0):
        # Check whether we have numeric values for our x and y values
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("x and y must be numeric values")
        
        # In case I want to have subclasses of Vertex in the future, I will use name mangling here.
        self.__label = label # stores the vertex's label; used when building paths
        self.__x = x
        self.__y = y

    # The appropriate getters for Vertex
    @property # using properties to have attribute privacy!
    def coords(self): # returns a tuple containing (x, y), which is an intuitive format
        return (self.__x, self.__y)
    
    @property
    def x(self): # returns the x component directly
        return self.__x
    
    @property
    def y(self): # returns the y component directly
        return self.__y

    @property
    def label(self): # returns the label directly
        return self.__label

    # magic method for '=='
    def __eq__(self, other):
        return self.__coords == other.__coords
    # likewise for '!='
    def __ne__(self, other):
        return self.__coords != other.__coords

    # It would be nice to calculate distances between vertices by simply using '+'
    # magic method for '+' so we can easily add two vertices, but to allow us to chain '+' operations, we need another object
    # I'll use a PathDistance object that stores the current vertex and distance travelled so far!
    # Then, I will define '+' for the PathDistance object to allow for full chaining when the first two operands are Vertex objects
    
    def __add__(self, other):
        # This is where we can implement the distance formula!
        if not isinstance(other, Vertex):
            raise TypeError("Operand needs to be a Vertex object!")

        # define the current path_str
        path_str = str(self.__label) + '-' + str(other.__label)
        
        # Return PathDistance object initialized via the following:
        return PathDistance(path_str,   # the current path we've taken, which is simply our first two Vertex objects
                            other,      # the right-operand will be the path's latest vertex
                            math.sqrt(((other.__x - self.__x)**2) + ((other.__y - self.__y)**2)) # the distance formula
                            )

    # For cases when we have other + self and 'other' is NOT a compatible operand.
    # This occurs when we use the sum() function with a list.
    # In cases where other methods use '0' as the first element in a sequence of additions, we simply return self
    def __radd__(self, other):
        return self

    # same as __radd__ except now for cases when we have self + other and 'other' is NOT compatiable operand.
    def __ladd__(self, other):
        return self

    # Since it'd be nice to print the Vertex object, I will define __str__ using the same format as the .tsp file nodes
    def __str__(self):
        return str(self.__label) + ' ' + str(self.__x) + ' ' + str(self.__y)

    # For echoing variables in the interactive interpreter, if I so choose to use it...
    def __repr__(self):
        return 'Vertex("' + self.__label + ', ' + self.__x + ', ' + self.__y + '")'


# ================================================ #
#              PATHDISTANCE CLASS DEFINITION
# ================================================ #


    # The PathDistance object conceptualizes a path that's currently being travelled
    # implemented mainly to allow for the chaining of '+' operations, starting with vertex objects
    # But it also stores the current vertex path as a string.
    # The PathDistance's string can be passed to the TSPMap instance's str_to_coords() method,
    # which returns a list of (x, y) tuples

class PathDistance():

    def __init__(self, path_str = "", vertex = Vertex(), distance = 0.0):
        self.__current_path = path_str
        self.__current_vertex = vertex
        self.__current_distance = distance
        self.__edges = None # Leave edges uncomputed for now... we'll use lazy computation

    # The appropriate getters for PathDistance
    @property
    def current_path(self):
        return self.__current_path

    @property
    def reversed_path(self):
        return self.__current_path[::-1]

    @property
    def current_vertex(self):
        return self.__current_vertex

    @property
    def current_distance(self):
        return self.__current_distance

    @property
    def length(self):
        
        return len(self.__current_path.split('-'))


    @property
    def edges(self):
        if self.__edges is None:
            # Now we can compute it, since we need it.
            self.__edges = self._calculate_edge()
        return self.__edges
    
    @property
    def normalized_edges(self):
        '''
        Returns a list of edges in the format [(1, 2), (2, 3), ...] for easy plotting or counting.
        '''
        if self.__edges is None:
            # Now we can compute it, since we need it.
            self.__edges = self._calculate_edge()
        normalized_edges = []    
        for v1, v2 in self.__edges.items():
            # treat v1 and v2 as integers, since they are vertex labels
            # Then make a tuple using v1 and v2,
            # Then  sort the tuple so that the edge (1, 2) is the same as (2, 1)
            normalized_edge = tuple(sorted((int(v1), int(v2))))
            normalized_edges.append(normalized_edge)

        return normalized_edges


    def split(self, split_point = 0.50):
        '''
        Takes a PathDistance objects and 'splits' it by return two lists of vertex labels.
        first_chunk: list[str] containing vertex labels of the first half
        second_chunk: list[str] containing vertex labels of the second half
        split_point: integer that defines the division point, default 0.50  is a 50/50 division.
        '''
        first_chunk, second_chunk = [], []

        # Convert split_point to an index
        split_point = int(np.ceil(self.length * split_point))

        for i in range(split_point):
            first_chunk.append(self[i])

        for j in range(split_point, self.length):
            second_chunk.append(self[j])

        return first_chunk, second_chunk

    def _calculate_edge(self):
        vertex_labels = self.__current_path.split('-') # get a list of our vertex labels
        edge_dict = {} # edges will be stored as dictionary. Key will be i and value is j for edge(i, j)
        for i in range(len(vertex_labels) - 1):
            j = i + 1 # i and j to iterate through subsequent pairs of vertex labels
            if j > len(vertex_labels):
                break
            # the key is the starting vertex, value is the ending vertex
            edge_dict[vertex_labels[i]] = vertex_labels[j]
        return edge_dict
        
    def _edge_insert(self, edge: str, vertex: Vertex):
        '''
        Inserts an edge between two vertices in the current str path.
        Returns the updated edge dictionary.

        Parameters:
        edge (str): the edge to insert the vertex into, in format '1-3' for, e.g., vertices 1 and 3
        vertex (Vertex): the vertex to insert into the edge

        NOTE: This method does not update the current distance nor path string.
        NOTE: Not intended for direct use by the user. Requires TSPMap object to rebuild path.
        NOTE: Called by the TSPMap object's _edge_insert() method.
        '''
        if self.__edges is None:
            # Now we can compute it, since we need it.
            self.__edges = self._calculate_edge()

        edge_points = edge.split('-') # parse input edge string

        if edge_points[0] not in self.__edges.keys():
            raise ValueError("Edge does not exist in PathDistance object!")
        else:
            # edge parameter expected in format '1-3' for vertices 1 and 3
      
            # We can easily build our new edge. If it was '1-3' and we want to insert '2',
            # ... we can build '1-2' and '2-3' and then delete '1-3'
            # thus, '1' keys to vertex
            # and vertex keys to '3'

            self.__edges[edge_points[0]] = str(vertex.label)        # insert the new edge
            self.__edges[str(vertex.label)] = edge_points[1]        # insert the new edge
                      
            # no need to delete the old edge, since we overwrote value at key edge_points[0]      
                                                                         
        return self.__edges # edge distances, and path strings, will be rebuilt via TSP object

    # Addition defined between PathDistance obj and Vertex obj, which updates self's instance attributes and returns self
    def __add__(self, other):
        if not isinstance(other, Vertex):
            raise TypeError("Operand needs to be a Vertex object!")
        # update the current path string
        self.__current_path += ('-' + str(other.label))
    
        # calculate the next increment of distance from current vertex to the next (other) vertex
        more_distance = math.sqrt(((other.x - self.__current_vertex.x)**2) + ((other.y - self.__current_vertex.y)**2))
        
        # current vertex set to the next ('other') vertex
        self.__current_vertex = other
        
        # update our current distance by adding the additional distance
        self.__current_distance += more_distance
    
        # return the same object
        return self

    # To compare two PathDistance objects, we simply compare the distances!
    def __eq__(self, other):
        return self.__current_distance == other.__current_distance

    def __ne__(self, other):
         return self.__current_distance != other.__current_distance

    def __getitem__(self, index):
        '''
        Simple indexing of vertex labels in the PathDistance object.
        '''
        return self.__current_path.split('-')[index]

    def __iter__(self):
        for label in self.__current_path.split('-'):
            yield label

    # Likewise, comparators are defined so we can compare different PathDistance objects.
    # Helpful for finding min or max paths!
    # less than
    def __lt__(self, other):
        return self.__current_distance < other.__current_distance
    
    # greater than
    def __gt__(self, other):
        return self.__current_distance > other.__current_distance
    
    # less than or equal to
    def __le__(self, other):
        return self.__current_distance <= other.__current_distance
    
    # greater than or equal to
    def __ge__(self, other):
        return self.__current_distance >= other.__current_distance

    # magic method for printing our PathDistance object, for usage in print() and str() functions...
    # I'll format it as "path-string current-distance"
    def __str__(self):
        return str(self.__current_path) + ' ' + str( self.__current_distance)

    # In case we use the interactive interpreter, as in Jupyter, for echoing, I will define the following:
    def __repr__(self):
        return 'PathDistance("' + self.__current_path + ', ' + str(self.__current_vertex) + ', ' + str(self.__current_distance) + '")'

    # Returns the length of the path stored in the PathDistance
    def __len__(self):
        return len(self.__current_path.split('-'))

###################################################
#               GetItem metaclass
###################################################

# implemented in order to allow the TSPMap class (as opposed to the object) to be subscriptable
# useful for accessing specific TSPMap objects via an implementation of __getitem__ for the whole class...
class GetItem(type):
    def __getitem__(cls, key):
        return cls._batch.get(key, "TSPMap object not found.")

    def keys(cls):
        return cls.__batch.keys()

# ================================================ #
#              TSPMap CLASS DEFINITION
# ================================================ #

# TSPMap class will store all of the vertices for a given TSP problem
# Thus, each TSP problem is conceptualized as a unique map (think geographical map) with a set of verticies, i.e., destinations

class TSPMap(metaclass=GetItem):
    # class attribute for storing TSPMap objects when computing over a directory of .tsp files
    # the class methods that run the algorithms for every .tsp file in the directory utilize TSP_batch
    _batch = {}

    def __init__(self, tsp_file: str, skippable_lines=7): # pass it the TSP filename
       
        # TSPMap object will be named after the .tsp file
        self.__name = tsp_file 

        # initialize an empty dictionary which will store our key-values, that is, vertex_label : vertex_object
        self.__nodes = {} 
        
        # The key-value pairs of our dictionary will be defined by data in the .tsp files
        with open(tsp_file, 'r') as file:
            # the .tsp files have coordinates begin on the 8th line, so skip number of lines (default is 7 for .tsp)
            for _ in range(skippable_lines):
                next(file)
            # Now we read the lines corresponding to the coordinate section
            for line in file:
                # Each node in the coord section has a unique label and x,y coordinate separated by whitespace ' '
                # Split the line by whitespace after we've stripped it of trailing characters
                node = line.strip().split(' ')

                # I expect each of the node lines to have 3 substrings in the order of label, x, and y.
                if len(node) != 3:
                    print("Error in line read! Node line does not contain exactly 3 elements!")

                # each instance stores a dictionary of the nodes, keyed according to the vertex name
                self.__nodes[node[0]] = Vertex(node[0], float(node[1]), float(node[2]))
       
    @property
    def nodes(self):
        return self.__nodes

    @property
    def name(self):
        return self.__name

    # Here, the dimension is the number of nodes in the .tsp problem file; so we return length of vertex list
    @property
    def dimension(self):
        '''
        Property
        Returns the number of vertices represented by the TSPMap instance
        '''
        return len(self.__nodes.values())

    # in case we wanted to plot the vertices, we return a list of (x, y) tuples that can be unzipped
    # this list represents all available verticies in our map (think geographical map)
    @property 
    def map_coords(self):
        '''
        Property
        Returns a list of (x, y) tuples representing the vertices in the TSPMap instance
        '''
        x_y_list = []
        for vertex in self.__nodes.values():
            # iterate through vertex values in our instance's dictionary and append (x, y) tuples
            x_y_list.append(vertex.coords)
        return x_y_list
    
    @property
    def centroid(self):
        '''
        Property
        Returns the centroid of the vertices in the TSPMap instance
        '''
        return self.find_centroid(self.map_coords)

    # Define a magic method for getting a particular vertex from our TSP map.
    # We can directly using indexing with our TSPMap object with the following:
    def __getitem__(self, dict_key: str):
        return self.__nodes[dict_key]
    
    # to print a TSPMap object, I convert the values of the nodes dict to a list using .values() method
    # it's a list of vertices, so I can use list(map(str, nodes.values())) to convert it to a list of strings
    # the vertices have __str__ defined, so this works and will give the string value for a Vertex object...
    # then, I join those list of string values, separated by '\n\', into a single string
    def __str__(self):
        return self.__name + '\n' + '\n'.join(list(map(str, self.__nodes.values())))

    ######################
    #   Other methods... #
    ######################

    # What if we have a particular path in string form derived from vertices in this instance?
    # We can use a string to get either a list of (x, y) tuples OR a list of vertices...

    def str_to_coords(self, path):
        '''
        Converts a path string to a list of (x, y) tuples.
        The path string is expected to be in the format "1-2-3-4-5" where each number is a vertex label.
        The method returns a list of (x, y) tuples representing the vertices in the path.
        '''
        # We'll be flexible enough to take either the string directly OR the PathDistance obj
        if isinstance(path, PathDistance):
            path_string = path.current_path
        elif isinstance(path, str):
            path_string = path
        else:
            raise TypeError("Argument 'path' must be either string or PathDistance object!")
        x_y_list = []
        # split path string via '-' (hyphen) separators,
        # since that's what I used to build the string in the PathDistance class
        vertex_keys = path_string.split('-') 
        # the split string can be used as keys to access our vertices in the instance
        for key in vertex_keys:
            x_y_list.append(self.__nodes[key].coords)
        return x_y_list


    def str_to_vertices(self, path):
        if isinstance(path, PathDistance):
            path_string = path.current_path
        elif isinstance(path, str):
            path_string = path
        else:
            raise TypeError("Argument 'path' must be either string or PathDistance object!")
        
        vertex_list = []
        # parse our path by splitting by '-'
        vertex_keys = path_string.split('-')
        
        for key in vertex_keys:
            vertex_list.append(self.__nodes[key])
        return vertex_list
    
    def area_of_triangle_path(self, path):
        '''
        Calculates the area of the triangle formed by the vertices in the path.
        The path is expected to be in the format "1-2-3" where each number is a vertex label.
        The method returns the area of the triangle.
        '''
        # We'll be flexible enough to take either the string directly OR the PathDistance obj
        if isinstance(path, PathDistance):
            path_string = path.current_path
        elif isinstance(path, str):
            path_string = path
        else:
            raise TypeError("Argument 'path' must be either string or PathDistance object!")
        
        # get the vertices in the path
        path_coords = self.str_to_coords(path_string)

        # calculate the area of the triangle
        return triangle_area(path_coords)
    
    def find_centroid(self, points: list[tuple[float, float]]):
        '''
        Takes a list of (x, y) coordinates, i.e., tuples of floats
        '''

        if not points:
            return None  # Return None if the list is empty to avoid division by zero
        
        # Sum the x and y coordinates separately
        x_sum = sum(point[0] for point in points)
        y_sum = sum(point[1] for point in points)
        
        # Calculate the mean for x and y
        n = len(points)
        centroid_x = x_sum / n
        centroid_y = y_sum / n
        
        return (centroid_x, centroid_y)

    ##########################################################################
    # GENETIC ALGORITHM (Project_4 and Project_5)
    #########################################################################

    ########################
    # The following methods were added during Project 4, Genetic Algorithms
    ########################
    
    def trim_path(self, path: PathDistance, trim_size: int):
        '''
        Trims N verticies from the end of the path. Returns a new PathDistance object of length L - N.
        For example, trim_path('1-2', 2) returns Path with '1-2'
        '''

        if path.length <= trim_size:
            raise ValueError("Cannot trim more vertices than the path has!")
        if isinstance(path, str):
            path = self.build_path(path)
        
        # iterate through vertices and build a new path string
        vertex_labels_list = []

        for i in range(path.length - trim_size):
            vertex_labels_list.append(path[i])

        return self.build_path(vertex_labels_list)

    def loop_path(self, path: PathDistance):
        '''
        Connects last vertex with the first vertex of a PathDistance object.
        Returns a new PathDistance object with the looped path.
        '''
        if isinstance(path, str):
            path = self.build_path(path)


        # new path
        if path[0] == path[-1]:
            return path

        tour = path + self.__nodes[path[0]]

        return tour




    def build_path(self, L: list[str]):
        '''
        Builds a PathDistance object given a list of vertex labels.
        The vertex labels are used to key the problem map's nodes dictionary.
        NOTE: Returned paths will be in reverse order compared to the passed list of labels.
        This is due to using pop() to build the path. Of course, a path is equivalent to its inverse.
        So, this is not a problem.
        '''
        if isinstance(L, str):
            # We assume it's a PathDistance path string '1-2-3-4-', for example
            L = L.split('-')

        # reversed list of vertex labels; to build the path in the correct order
        L = list(reversed(L))

        # Initialize the PathDistance object by adding two Vertex objects
        path = self.__nodes[str(L.pop())] + self.__nodes[str(L.pop())]

        # builds the rest of our PathDistance
        # We continue to work backwards, so we reverse our list.
        for vertex_label in reversed(L):
            path += self.__nodes[str(L.pop())]

        return path
    # end build_path()

    def generate_random_paths(self, population_size: int):
        '''
        Random PathDistance objects using the problem map's vertices, with no replacement.
        population size: int, specify the number of random paths to generate

        Returns list of the PathDistance objects.
        '''
        
        population = []

        for i in range(population_size):
            # Generate permutation of numbers from [1, self.dimension], where self.dimension is the # of vertices in the map
            random_permutation = list(np.random.permutation(np.arange(1, self.dimension + 1)))
            # Build the PathDistance and append
            population.append(self.build_path(random_permutation))
        
        return population
   
    def rank_population(self, population: list[PathDistance]):
        '''
        Sorts a list of PathDistance objects by their distance.
        Returns dictionary with key = rank, value = PathDistance
        '''
        sorted_population = sorted(population)

        ranked_population = {}

        #in-place sort in ascending order. First element is the 'best' in terms of shorest distance
        for i, path in enumerate(sorted_population):
            rank = i + 1 # so our rank is from [1, population_size]
            ranked_population[rank] = path
        return ranked_population
    # end rank_population()

    def select_subpopulation(self, ranked_population: dict[int, PathDistance], sub_size):
        '''
        Takes a dictionary of rank, PathDistance objects and returns a subpopulation of size M
        The returned subpopulation contains the top M individuals from the ranked population.
        '''
        
        ranked_subpopulation = {}

        for i in range(sub_size):
            rank = i + 1
            ranked_subpopulation[rank] = ranked_population[rank]

        return ranked_subpopulation
    # end select_subpopulation()

    def count_edges(self, edge_count_dict: dict[tuple, int], path_population: list[PathDistance]):
        '''
        Takes a list of PathDistance objects, iterating through the list, and counts the number of times an edge appears
        Returns a dictionary of edge counts.
        '''
        for path in path_population:
            for edge in path.normalized_edges:
                edge_count_dict[edge] = edge_count_dict.get(edge, 0) + 1
        return edge_count_dict




    def chunk_and_sweep(self, first_parent: PathDistance, second_parent: PathDistance, split_num: float):
        '''
        One of the cross-over functions for intended use within the reproduce() function.

        Takes two parental PathDistance objects and returns four offspring lists with vertex labels.
        We return a list because this is easier to mutate. The final offspring is built after the mutation step.

        split_num: float, is passed to the PathDistance .split() method to return two chunks
        '''
        ################
        # Chunk
        ################
        # Find our 'chunks' of our paths. Each parent contributes two 'gametes', so to speak
        first_chunk1, first_chunk2 = first_parent.split(split_num)
        
        second_chunk1, second_chunk2 = second_parent.split(split_num)

        ################
        # Sweep
        ################
        # grab a chunk, sweep through the other parent and build the offspring!
        # build first parent's gametes using second parent's body

        for vertex in second_parent:
            if vertex not in first_chunk1:
                first_chunk1.append(vertex)

            if vertex not in first_chunk2:
                first_chunk2.append(vertex)
        
        # build second parent's gametes using first parent's body
        for vertex in first_parent:
            if vertex not in second_chunk1:
                second_chunk1.append(vertex)
            
            if vertex not in second_chunk2:
                second_chunk2.append(vertex)
        # Build the PathDistance objects using our lists.
        
        offspring_lists = [first_chunk1, first_chunk2, second_chunk1, second_chunk2]

        offspring = list(self.build_path(o) for o in offspring_lists)

        return offspring
    # end chunk_and_sweep()


    def simple_mutate(self, path: PathDistance, mutation_probability = 1):
        '''
        Simple mutation that picks two indices in a path and swaps the vertices.
        We assume that the mutation rate is handled outside of the function call.
        '''
        if random.uniform(0, 1) > mutation_probability:
            # Probability results in the mutation NOT occuring; return back path
            return path

        # Get a list of the path, so that we can mutate it.
        protopath = path.current_path.split('-')
        
        # Generate a random number between lower (0) and upper (path.length - 1)
        # Do this twice, one for each position that we will swap
        first_position = random.randint(0, path.length - 1)
        second_position = random.randint(0, path.length - 1)
        # if the second number is the same, keep generating until we get a different number
        while (second_position == first_position):
            second_position = random.randint(0, path.length - 1)
        protopath[first_position], protopath[second_position] = protopath[second_position], protopath[first_position]

        return self.build_path(protopath)

    def reproduce(self,
                  parent_rank_df: pd.DataFrame,  # data frame for saving the distribution of parent reproductive events
                  rank_min: int,                 # the best rank number
                  rank_max: int,                 # the worst rank number
                  ranked_subpopulation: dict[int, PathDistance],    # our ranked subpop, i.e., our parents
                  population_size: int,                             # Number we reproduce up to
                  parent_subpop_size: int,                          # Number of parents
                  lambda_param: float,          # Poisson distribution parameter for reproductive events
                  crossover_method: str,        # Method for creating new offspring
                  split_lower: float,           # lower parameter bound for crossover_method splitting
                  split_upper: float,           # upper parameter bound for crossover_method splitting
                  mutation_prob: float          # Mutation rate
                  ):

        #print("our ranked subpopulation") # for testing
        #for key, value in ranked_subpopulation.items(): # for testing
            #print(f"rank {key} and  {value}") # for testing

        ##############################
        # Poisson Distribution for Parental Reproduction
        #############################
       
        offspring_size = int(population_size - parent_subpop_size) # so parent subpop + offspring subpop should reconstitute population size
        
        # First, determine distribution of parent reproductive events
        offspring_counts = np.random.poisson(lambda_param, offspring_size)
    
        # Clamp the Poisson values to the rank range [rank_min, rank_max]
        # Basically, if we get a random value below rank_min, we force it to be value rank_min;
        # ... and if we get a value > rank_max, we force it -- clip it -- to the same value as rank_max
        offspring_counts_clamped = np.clip(offspring_counts, rank_min, rank_max)
    
        # Count the distribution of clamped offspring counts; return unique values--which are our ranks (x-axis in the histogram)
        # ... also return the counts, i.e., how many times a rank appeared. This will be the y-axis.
        # This count is also the number of times that given rank gets to reproduce. So, it's like drawing raffle tickets to reproduce!
        parent_rank, parent_offspring_number  = np.unique(offspring_counts_clamped, return_counts=True)
    
        # Zip the two arrays into a dictionary, for easy iteration below
        rank_offspring_dict = dict(zip(parent_rank, parent_offspring_number))
    
        # Convert the dictionary to a DataFrame and concatenate it to the main DataFrame
        # This represents the distribution of reproductive events by rank for a single generation
        df_single_generation = pd.DataFrame([rank_offspring_dict])
        
        # Replace NaN values with 0
        df_single_generation.fillna(0, inplace=True)


        # FUTUREWARNING for the following line of code. I'm going to ignore the warning for now.
        # Update the DataFrame for our reproductive events, since we want to get an average and total distribution at the end to plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            new_parent_rank_df = pd.concat([parent_rank_df, df_single_generation], ignore_index=True)
        
        #print("Distribution of reproductive events to respective ranks") # for testing
        #print(rank_offspring_dict) # we use this to portion out the reproductive events, for testing

        # Initialize list for storing our new population; note that this includes parents
        new_population = [] 
        
        # Note that num_reproductive_events (the inner for-loop) will total to the number of offspring for this generation
        # Thus, our new population size will be |set of parents| + |set of new offspring| = population_size

        # Begin allotting the reproductive events to their respective ranks
        for rank, num_reproductive_events in rank_offspring_dict.items():
            #print('\n\n') # for testing
            #print(f"Parent Rank {rank} gets to mate {num_reproductive_events} times!") # for testing

            for _ in range(num_reproductive_events):
                
                # Generate random number to pick a random mate; keep doing so until we get a non-self number
                mate_rank = random.randint(rank_min, rank_max)
                while mate_rank == rank:
                    mate_rank = random.randint(rank_min, rank_max)
                #print(f"Mate rank = {mate_rank}") # for testing
                
                # call our crossover method with random split point within interval [split_lower, split_upper]
                random_split_point = random.uniform(split_lower, split_upper)
                #print(f"split point = {random_split_point}") # for testing
           

                # Select appropriate method given by keyword argument
                if crossover_method == 'chunk_and_sweep':
                    offspring = self.chunk_and_sweep(ranked_subpopulation[rank],        # get the main parent
                                                     ranked_subpopulation[mate_rank],   # get the randomly chosen mate
                                                     random_split_point                 # pass our random split point
                                                    )         
                elif crossover_method == 'sweep_and_chunk':
                    print("Sweep and chunk!") # for testing
                elif crossover_method == 'alternating_sweep':
                    print("Alternating sweep!") # for testing
                else:
                    # if we get non-sense keyword, just default to chunk_and_sweep
                    offspring = self.chunk_and_sweep(ranked_subpopulation[rank],        # get the main parent
                                                     ranked_subpopulation[mate_rank],   # get the randomly chosen mate
                                                     random_split_point                 # pass our random split point
                                                    )       
                # We now have our offspring. Pass them through mutation function
                # mutate our offspring (remember--not 100% guaranteed to mutate unless probability = 1)

                #print(f"mutation probability: {mutation_prob}") # for testing

                # Do a simple list comprehension to get our list of mutated offspring!
                mutated_offspring = [self.simple_mutate(o, mutation_prob) for o in offspring]

                #print(f"main parent: {ranked_subpopulation[rank]}")         # for testing
                #print(f"best offspring: {sorted(mutated_offspring)[0]}")    # for testing
                # Here, we simply select the best of the four offspring... sort in ascending order and get index 0
                # Add offspring to new population
                new_population.append(sorted(mutated_offspring)[0])
        
        # Add the parent to the new population, too.
            new_population.append(ranked_subpopulation[rank])
        
        # Return the new population (list of PathDistance objects) and the df for this generation's reproductive event distributions    
        return new_population, new_parent_rank_df
     # end reproduce()
            

        


    def genetic_algorithm(self, 
                          generations = 100,                # the number of iterations, i.e., population generations.
                          population_size = 100,            # the max size of our population each generation 
                          subpop_proportion = 0.1,          # the portion of population chosen to repopulate the next generation
                          lambda_param = 1,                 # Poisson distribution param of reproduction opportunities within parental subpop
                          crossover_method = 'chunk_and_sweep', # define the crossover method
                          split_lower = 0.1,
                          split_upper = 0.9,                     # define the 'split point' for crossovers. 
                          mutation_prob = 0.1,                   # mutation probability. Random floats between [0, 1] below this -> mutate
                          abominate_threshold = float('inf'),
                          abomination_percentage = 0.1,
                          save_data = True,
                          file_path = '.',
                          run_number = 0,
                          tours = 'false',
                          verbose = 4,                       # 0: no print, 1: print G num by mod 1000, 2: print G num by mod 100, 3: print G num by mod 10
                          ):
        
        '''
        Genetic algorithm that initializes with a random population of size population_size.
        Returns the best and worst PathDistance objects, which can be plotted.
        '''

        #######################
        # other parameters
        #######################
        parent_subpop_size = int(np.ceil(population_size * subpop_proportion))



        rank_min = 1            # the 'best' parental rank, i.e., #1 is best!
        rank_max = parent_subpop_size  # the 'worst' parental rank

        abomination_proportion = int(np.ceil(parent_subpop_size * abomination_percentage))

        ##################################
        # Initialize DataFrame objects here
        ##################################

        #######################
        # parent-rank offspring-number DataFrame
        ########################
        # column names
        all_possible_parent_ranks = np.arange(1, parent_subpop_size + 1)
        
        # empty DataFrame initialized with column names
        parent_rank_offspring_number_df = pd.DataFrame(columns=all_possible_parent_ranks)

        #######################
        # GA run DataFrame initialized here (all solution distances from each generation)
        #######################
        # This DataFrame will store the results of every generation and thus the entire run.
        # These results will be saved as a .csv with which we can create pretty figures
        
        all_possible_ranks = np.arange(1, population_size + 1)
        
        ga_run_df = pd.DataFrame(columns=all_possible_ranks)
        


        #######################
        # Best and worst path
        ######################

        best_path = PathDistance(path_str = '', distance = float('inf'))    # inf distance is the worst, so we initialize best to this
        worst_path = PathDistance(path_str = '', distance = float('-inf'))  # -inf is the best, so we initialize the worst to this

        edge_counts = {} # dictionary to store edge counts; edge counts are akin to allele frequency, or gene frequency (loosely)

        abominate_count = 0

        ############################################
        # Initialize the first random population here!
        ############################################
        # Generate the first population randomly
        new_population = self.generate_random_paths(population_size)
        edge_counts = self.count_edges(edge_counts, new_population)
        if tours == 'true':
            # make our population into tours
            new_tours = [self.loop_path(path) for path in new_population]
            # rank population as they are tours...
            ranked_tours = self.rank_population(new_tours)

            # convert the ranked tours back to a dict of paths
            ranked_population = {rank: self.trim_path(path, 1) for rank, path in ranked_tours.items()}

        else:
            # Rank the population, after which we record data
            ranked_population = self.rank_population(new_population)
        
        ####################
        # generational data recorded here
        ####################
        
        # Dictionary comprehension to convert rank: PathDistance to rank: PathDistance.current_distance
        ranked_distances = {rank: path.current_distance for rank, path in ranked_population.items()}
        
        # Convert the dictionary to a DataFrame and concatenate it to the main DataFrame
        this_generation = pd.DataFrame([ranked_distances])
        

        # IGNORE FUTURE WARNING HERE; issue seems to come from the conversion of a dict to a pandas df
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # append it to our ga_run_df
            ga_run_df = pd.concat([ga_run_df, this_generation], ignore_index=True)

        # key 'min_rank' will have the best of this generation... compare with the global best path
        if ranked_population[rank_min] < best_path:
            best_path = ranked_population[rank_min]

        # index -1 (the last element) is the worst of this generation; compare with global worst
        if ranked_population[rank_max] > worst_path:
            worst_path = ranked_population[rank_max]

        # Select the parents; the rest of the population is culled in this step
        parent_subpopulation = self.select_subpopulation(ranked_population, parent_subpop_size)

        # Now we simulate the reproductive cycle through G generations.
        for _ in range(generations):

            if verbose == 1 and _ % 1000 == 0:
                print(f"Generation {_} of {generations} for run (or crowd member): {run_number}")
            if verbose == 2 and _% 100 == 0:
                print(f"Generation {_} of {generations} for run (or crowd member): {run_number}")
            if verbose == 3 and _% 10 == 0:
                print(f"Generation {_} of {generations} for run (or crowd member): {run_number}")
            if verbose == 4: 
                print(f"Generation {_} of {generations} for run (or crowd member): {run_number}")


            # Get the new population produced by our parents, and get the distribution data for reproductive events
            new_population, parent_rank_offspring_number_df = self.reproduce(
                                                        parent_rank_offspring_number_df,    # parent reproduction data
                                                        rank_min,           # best rank
                                                        rank_max,           # worst rank
                                                        parent_subpopulation,   # our ranked subpopulation of parents
                                                        population_size,        # maximum population size up to which we reproduce
                                                        parent_subpop_size,     # size of our parent subpopulation
                                                        lambda_param,           # pass lambda param for parent reproduction
                                                        crossover_method,       # define crossover_method
                                                        split_upper,            # split range interval lower bound
                                                        split_lower,            # split range interval upper bound
                                                        mutation_prob           # mutation rate
                                                        )
                                                 
            edge_counts = self.count_edges(edge_counts, new_population)  

            if tours == 'true':
                # make our population into tours
                new_tours = [self.loop_path(path) for path in new_population]
                #print("ranking newly generated population as tours...") # for testing
                # rank population as they are tours...
                ranked_tours = self.rank_population(new_tours)
                # convert the ranked tours back to a dict of paths
                ranked_population = {rank: self.trim_path(path, 1) for rank, path in ranked_tours.items()}
            else:
                # Rank the population, after which we record data
                ranked_population = self.rank_population(new_population)
             
            # rank the new population
            ranked_population = self.rank_population(new_population)
                
            ####################
            #Record  generational data after ranking population!
            ####################i
    
            # Dictionary comprehension to convert rank: PathDistance to rank: PathDistance.current_distance
            ranked_distances = {rank: path.current_distance for rank, path in ranked_population.items()}
        
            # Convert the dictionary to a DataFrame and concatenate it to the main DataFrame
            this_generation = pd.DataFrame([ranked_distances])

            # append it to our ga_run_df
            ga_run_df = pd.concat([ga_run_df, this_generation], ignore_index=True)

            # rank_min is the key for this generation's best!
            if ranked_population[rank_min] < best_path:
                best_path = ranked_population[rank_min]

            else:
                #print(f"Best path did not improve in generation {_}") # for testing
                # If we didn't improve, increment the abominate count
                abominate_count += 1

            # rank_max is the key for this generation's worst!
            if ranked_population[rank_max] > worst_path:
                worst_path = ranked_population[rank_max]
            

            parent_subpopulation = self.select_subpopulation(ranked_population, parent_subpop_size)

            #print(abominate_count) # for testing
            if abominate_count > abominate_threshold:
                # reset the count
                abominate_count = 0
                abominations = self.generate_random_paths(abomination_proportion)

                # Assign random path to the worst parents, working towards the best
                for abomination_index in range(abomination_proportion):

                    #print(f"Abomination {abomination_index}! Replacing {rank_max - abomination_index}") # for testing

                    parent_subpopulation[rank_max - abomination_index] = abominations[abomination_index]

                    
        # Final formatting for the DataFrames to be written to .csv 
        # Modify the column names by placing 'r' in the front, to denote rankings
        if save_data:

            print("Writing data to .csv files...")
            
            # Process the reproductive event distribution data
            parent_rank_offspring_number_df.columns  = ['r' + str(col) for col in parent_rank_offspring_number_df.columns]
            parent_rank_offspring_number_df.index.name = 'generation'
            parent_rank_offspring_number_df.to_csv(f'{file_path}/events/run_{run_number}_parent_rank_offspring_number_df.csv')
            

            ########################
            # process the GA run data
            ########################
            # Select only the numerical columns (excluding the first column, or index, to be named 'generation')
            numerical_cols = ga_run_df.iloc[:, 1:]

            # Calculate min, max, average, and SEM for each row ignoring the 'generation' column
            ga_run_df['min'] = numerical_cols.min(axis=1)
            ga_run_df['max'] = numerical_cols.max(axis=1)
            ga_run_df['average'] = numerical_cols.mean(axis=1)
            ga_run_df['SEM'] = numerical_cols.std(axis=1) / np.sqrt(numerical_cols.shape[1])

            ga_run_df.columns = ['rank_' + str(col) for col in ga_run_df.columns]
            ga_run_df.index.name = 'generation'
            ga_run_df.to_csv(f'{file_path}/ga/run_{run_number}_ga_run_df.csv')
            print("Data saved.")

        #print("Genetic algorithm run complete. The best and worst of this run are: ") # for testing
        #print(f"Best: {best_path}") # for testing
        #print(f"Worst: {worst_path}") # for testing


        return best_path, worst_path, ranked_population, edge_counts
    
    def _majority_vote_aggregation(self, crowd_edge_counts: dict[tuple[str, str], int]):
        '''
        This function aggregates the individual solutions of the crowd members.
        The aggregation is done using a majority vote by iterating through edges, sorted in descending order by count.
        The function returns the aggregated path as a PathDistance object.
        '''
        # Boolean to track if we needed to resort to greedy edge insertion
        greedy_insertion = False

        # Keep track of used vertices
        used_vertices = set()

        # Sort the edges by their count in descending order; gives a list of tuple[tuple, int]
        sorted_edges = sorted(crowd_edge_counts.items(), key=lambda x: x[1], reverse=True)

        # Initialize the aggregated path using the first edge; also remove it from our list
        first_edge, count = sorted_edges.pop(0)
        used_vertices = {first_edge[0], first_edge[1]}

        aggregated_path = self.build_path(str(first_edge[0]) + '-' + str(first_edge[1]))
        print(f"\nAggregating solution ... starting with edge {first_edge[0]}-{first_edge[1]} with highest edge count of {count}")
        # loop until we have no more edges to add
        while True:
            # Boolean check to see if we have no more edges to add
            found_edge = False

            # Iterate through the sorted edges
            for edge, count in sorted_edges:
                
                # Get the vertices of the edge and our current vertex in the aggregated path
                v1, v2 = edge

                # If both vertices are already in our path, skip this edge since our path already has it.
                if v1 in used_vertices and v2 in used_vertices:
                    print(f"Skipping edge {v1}-{v2} since both vertices are already in the growing aggregation path...")
                    continue
                
                # Remember: We work with the Vertex object's label property, not the actual Vertex object itself
                current_vertex = int(aggregated_path.current_vertex.label)
                print(f"Aggregating solution ... finding vertex for current vertex: {current_vertex}")

                if current_vertex in edge:
                    # if current_vertex == v1, and if v2 is not already in our paththen we add v2 to the path
                    if current_vertex == v1 and v2 not in used_vertices:
                    
                        aggregated_path += self[str(v2)]
                        used_vertices.add(v2)
                        print(f"Adding vertex {v2} to the path tp yield edge {current_vertex}-{v2}")
                        found_edge = True
                        break

                    # if current_vertex == v2, then we add v1 to the path
                    if current_vertex == v2 and v1 not in used_vertices:

                        aggregated_path += self[str(v1)]
                        used_vertices.add(v1)
                        print(f"Adding vertex {v1} to the path tp yield edge {current_vertex}-{v1}")
                        found_edge = True
                        break

            if not found_edge:
                break

        ########    
        # Keep track of used vertices! We need to add the unused vertices to the path if they exist
        ########
        all_vertices = set(self.__nodes.keys())

        included = set(aggregated_path.current_path.split('-'))

        unused_vertices = set()

        for vertex in all_vertices:
            if vertex not in included:
                unused_vertices.add(vertex)

        print(f"\nInitial aggregated path p_0 length is {aggregated_path.length}. Unused vertices: {unused_vertices}")

        print("Now adding unused vertices to the path...")


        ###############################
        # Algorithm for inserting the unused vertices
        ###############################

        path_edges = set(aggregated_path.normalized_edges)

        for unused_vertex in unused_vertices:
            print(f"\nWorking on unused vertex {unused_vertex}...")

            # Convert path to a list of vertices, for easy manipulation
            aggregated_path_vertex_list = aggregated_path.current_path.split('-')

            # Get the normalized edges of the path, for finding edge locations
            path_edges = set(aggregated_path.normalized_edges)

            # Initialize a set to store the possible edges for the unused vertex
            unused_vertex_partners = set()
            possible_edges = set()

            # For a given unused vertex, find all of its edge partners!
            for edge, count in sorted_edges:
                print(f"Checking edge {edge} for our unused vertex's partner (neighbor)...")
                v1, v2 = edge

                if v1 == int(unused_vertex):
                    print(f"Found partner for unused vertex {unused_vertex}: {v2}")
                    unused_vertex_partners.add(v2)

                if v2 == int(unused_vertex):
                    print(f"Found partner for unused vertex {unused_vertex}: {v1}")
                    unused_vertex_partners.add(v1)

            print(f"\nOur unused vertex {unused_vertex} has these partners: {unused_vertex_partners}")
            print(f"Now finding all possible edges for the unused vertex partners...")
            # For all of the unused vertex's partners, find all possible edges!
            for partner1 in unused_vertex_partners:
                for partner2 in unused_vertex_partners:
                    if partner1 != partner2:
                        # add the normalized, i.e. sorted, edge to our set of possible edges
                        possible_edges.add(tuple(sorted((partner1, partner2))))



            print(f"\nSorting possible edges by edge count, if applicable...")
            # Create a list of tuples (edge, count)
            edges_with_counts = [(edge, crowd_edge_counts.get(edge, 0)) for edge in possible_edges]

            # Sort the list of tuples based on the count in descending order
            sorted_edges_with_counts = sorted(edges_with_counts, key=lambda x: x[1], reverse=True)

            print(f"\nSorted possible edges for unused vertex {unused_vertex}:")

            matched_vertex1 = None
            matched_vertex2 = None

            print(f"\nNow finding if one of the possible edges is in our path...")

            # Now we find if one of the possible edges is in our path! Just print "Found it!" for now, for testing
            for edge, count in sorted_edges_with_counts:
                #print(f"Checking edge {edge} with count {count}...")
                if edge in path_edges:
                    print(f"Edge insertion for unused vertex {unused_vertex} found! Edge: {edge}")
                    matched_vertex1, matched_vertex2 = edge

                    # Break since we found our match
                    break

            # Iterate through our path string,
            # and find the index of the first matched vertex,
            # whether it's matched_vertex1 or matched_vertex2:
            if matched_vertex1 is not None and matched_vertex2 is not None:
                for i in range(aggregated_path.length):
                    if aggregated_path[i] == str(matched_vertex1) or aggregated_path[i] == str(matched_vertex2):
                        # Insert the unused vertex after the matched vertex
                        print(f"Inserting unused vertex {unused_vertex} after vertex {aggregated_path[i]} in our aggregated path...")

                        # This modifies in-place!
                        aggregated_path_vertex_list.insert(i + 1, str(unused_vertex))

                        # rebuild our path from the modified list with the newly inserted vertex label
                        aggregated_path = self.build_path(aggregated_path_vertex_list)

                        # break since we inserted our vertex
                        break

            
        ################
        # Check which vertices are still unused
        ################
        included = set(aggregated_path.current_path.split('-'))

        unused_vertices = set()
        for vertex in all_vertices:
            if vertex not in included:
                unused_vertices.add(vertex)

        print(f"\nUnused vertices after edge count majority voted aggregation: {unused_vertices}")

        if len(unused_vertices) == 0:
            print("\nAll vertices have been added to the path!")
            return self.loop_path(aggregated_path), greedy_insertion

        # check if the unused vertices are empty...
        ##########################
        # resorting to closest edge insertion!
        ##########################
        if len(unused_vertices) != 0:
            print("\nAdding the remaining unused vertices to the path by means of closest edge insertion...")
            # Greedy edge insertion for the remaining unused vertices
            greedy_insertion = True
            aggregated_tour = self.loop_path(aggregated_path)
            # Get the tour vertices as the 'initial vertices'
            aggregated_tour_vertices = aggregated_tour.current_path.split('-')
            # Call the closest edge insertion method with our current aggregated tour.
            aggregated_tour, _ = self.closest_edge_insertion(aggregated_tour_vertices, plot_steps=False, line_segment=True)
    
            # Check if we have any unused vertices after the closest edge insertion...
            unused_vertices = set()
            included = set(aggregated_tour.current_path.split('-'))
            for vertex in all_vertices:
                if vertex not in included:
                    unused_vertices.add(vertex)

            print(f"\nFinal unused vertices after closest edge insertion (should be empty): {unused_vertices}")


        return aggregated_tour, greedy_insertion
# end _majority_vote_aggregation() function




    ######################
    # END OF GENETIC ALGORITHM
    #####################

    ######################
    # Other
    # Algorithms for TSP #
    ######################

    # The following algorithms will return a PathDistance object (or list of objects) as a solution.
    # The brute force algorithm will be O(N!) since I am going through every possible path permutation
    # No thinking about optimization. I'm just going to sequentially work through all possibilities, exhaustively.
    # This does not use a Tree class, only a recursive function.
    
    # user can call this function
    def brute_force(self):

        # all solutions (i.e. paths) will be appended onto here.        
        all_paths = []       

        # iterate through our nodes and choose our starting point
        for vertex in self.__nodes.values():

            # create a deep copy of our full nodes and
            next_nodes = copy.deepcopy(self.__nodes)
            # remove the current node, which I do by popping it off and using the returned value to build current path
            current_path = copy.deepcopy(PathDistance(vertex.label, next_nodes.pop(vertex.label), 0.0))
            
            # I assume we will not have a .tsp file with only one node, so I won't check list size here...
            
            self._recursive_brute_force(current_path, next_nodes, all_paths)
            
        return all_paths

    # the following is the recursive function; only intended to be called by brute_force_all() method    
    # type hint to show what we really expect for the parameters
    
    # user does not call the following function
    def _recursive_brute_force(self, cur_path: PathDistance, nodes: dict[str, Vertex], solutions: list[PathDistance]):
        # iterate through our unvisited nodes...
        for vertex in nodes.values():
           
            # re-assign our next set of unvisited nodes via making a deep copy
            next_nodes = copy.deepcopy(nodes)
           
            # add on to our current path and remove that vertex from the next set of nodes
            # but again, we do so with deep copies
            current_path = copy.deepcopy(cur_path)
            current_path += next_nodes.pop(vertex.label)

            # if our next set of nodes is NOT empty after removing current node, continue the recursion; else append solution
            if len(next_nodes.values()) != 0:
                self._recursive_brute_force(current_path, next_nodes, solutions)
            elif len(next_nodes.values()) == 0:
                # print(current_path) # for testing
                solutions.append(current_path)


    def brute_force_tree():
        # <<Possible future feature>>
        # brute force algorithm for constructing a tree wherein leaf nodes contain paths
        # would require a tree class     
        pass

    ######################
    # Greedy Edge Insertion for TSP 
    ######################

    def closest_edge_insertion(self, initial_vertices: list[str], plot_steps=False, line_segment=False, solution_directory_prefix=""):  
        
        '''
        Greedy algorithm for the TSP problem.
        The closest edge insertion is a heuristic used in a greedy algorithm that builds a solution by iteratively adding the vertex that minimizes the total distance.
        The vertex is added to the edge that is closest to the current path.
        Distance to edge is approximated as a point projected to a line (whether infinite or finite)
        The algorithm starts with a circular path from the initial vertices, usually 3 vertices.

        Parameters:
        initial_vertices (list): list of vertex labels to start the path
        plot_steps (bool): whether to plot the steps of the algorithm
        line_segment (bool): whether to calculate the distance to a finite line segment or an infinite line
        
        '''
        print("Starting closest edge insertion algorithm...")
        solution_steps = [] # list of PathDistance objects representing the solution as it builds
        step = 0 # counter for the number of steps taken    
        unvisited_nodes = copy.deepcopy(self.__nodes)
        initial_tour = None

        # Construct circular path from initial vertices.
        # Assumed to be triangular (3 vertices).
        if len(initial_vertices) == 3:
            initial_tour = PathDistance(self[str(initial_vertices[0])].label, self[str(initial_vertices[0])], 0.0)

            del unvisited_nodes[str(initial_vertices[0])]

            for vertex_label in initial_vertices[1:]:

                initial_tour += self.__nodes[str(vertex_label)]

                # remove the node from our unvisited nodes
                del unvisited_nodes[str(vertex_label)]

            initial_tour += self[str(initial_vertices[0])] # close the loop
            # print(initial_tour) # for testing
            # print(unvisited_nodes.keys()) # for testing

        else:
            print(f"Building tour with {len(initial_vertices)} initial vertices...")

            initial_tour = self.build_path(initial_vertices)

            # remove the nodes from our unvisited nodes
            for vertex_label in initial_vertices:
                if vertex_label in unvisited_nodes:
                    del unvisited_nodes[vertex_label]


        # add the initial tour to our list of solution steps
        solution_steps.append(initial_tour)


        if plot_steps:
            initial_vertices_str = [str(x) for x in initial_vertices]

            solution_directory_prefix = solution_directory_prefix + "_".join(initial_vertices_str) + "/"

            self.plot_path(initial_tour, solution_directory_prefix + "Step " + str(step), save=True)
            
            # close plot
            plt.close()

        next_tour = copy.deepcopy(initial_tour)

        print(f"Next tour, sent by driver function: {next_tour}")

        # print(step, next_tour) # for testing
        
        return self._recursive_closest_edge_insertion(next_tour,     # initial tour
                                                      unvisited_nodes,  # unvisited nodes
                                                      step,             # counter for the number of steps taken, for naming the plots
                                                      solution_steps,   # list of PathDistance objects representing the solution as it builds
                                                      plot_steps,       # specify whether we need to plot the steps, individually
                                                      line_segment,      # specify whether we need to calc distance to a finite line (=True) or infinite line (=False)
                                                      solution_directory_prefix,            # prefix for the solution directory
                                                      )
        # Now we can start the greedy edge insertion recursive call...
        # Pass the initial tour and the unvisited nodes, and the solution steps list

        #return self._recursive_closest_edge_insertion(self, initial_tour, unvisited_nodes, step, solution_steps)

    def _recursive_closest_edge_insertion(self, 
                                          tour: PathDistance,                   # current (best) tour, greedy first choice
                                          unvisited_nodes: dict[str, Vertex],   # unvisited nodes
                                          step: int,                            # counter for the number of steps taken, for naming the plots
                                          solution_steps: list[PathDistance],   # list of PathDistance objects representing the solution as it builds
                                          plot_steps=False, 
                                          line_segment=False,
                                          solution_directory_prefix=""):        # specify whether we need to plot the steps, individually
        '''
        Recursive function for the closest edge insertion greedy algorithm.
        This function is intended to be called by the closest_edge_insertion() method.
        '''
        # base case: if we have visited all nodes, we return the tour
        if len(unvisited_nodes) == 0:
            return tour, solution_steps

        # for each vertex in our unvisited nodes, we find the closest edge to our current tour
        # and insert the vertex into that edge. We do this until we find the added vertex that minimizes the total distance
        # we then add the new tour to our list of solution steps

        #############################
        # initialize our min search;
        # find the vertex that minimizes the total distance
        # when added to the current tour
        #############################

        # make a deep copy of the current tour, to prevent weird stuff from happening
        tour = copy.deepcopy(tour)

        unvisited_nodes_keys = list(unvisited_nodes.keys())     # keys of unvisited nodes
                                                                # we iterate through this key list to delete the vertex from the unvisited nodes

        # dummy value for testing; we set the minimum to a HIGH value...
        min_tour = PathDistance(path_str="", vertex=Vertex(), distance=99999) # initialize min_tour

        # iterate through the rest of the unvisited nodes
        # insert the vertex into the edge, one by one.
        for vertex_label in unvisited_nodes_keys:

            # get the vertex object from the label
            vertex = self[vertex_label]

            # make a deep copy of the original tour, to prevent headaches
            # we will insert the vertex into the edge of this deep copy
            temp_tour = copy.deepcopy(tour) # make a deep copy of the original tour

            # find the closest edge to our current tour
            closest_edge = self._find_closest_edge(temp_tour, vertex, line_segment)

            # insert the vertex into the edge
            # print(f"inserting {vertex_label} into edge {closest_edge} of {temp_tour}") # for testing

            temp_tour = self._edge_insert(temp_tour, closest_edge, vertex)
            # print(temp_tour) # for testing

            if temp_tour < min_tour:
                min_tour = temp_tour
                min_vertex_label = vertex_label

        # add the new tour to our list of solution steps
        solution_steps.append(min_tour)

        # remove the vertex from the unvisited nodes
        del unvisited_nodes[min_vertex_label]

        # print solution step
        if plot_steps:
            step += 1
            self.plot_path(min_tour, solution_directory_prefix + "Step " + str(step), save=True)
            # close plot
            plt.close()

        # commented-out print-statements that were used when testing...
        # print("##############################################")
        # print("Step " + str(step), ":", min_tour)
        # print("##############################################")
        # recursive call
        return self._recursive_closest_edge_insertion(min_tour,         # new tour
                                                      unvisited_nodes,  # unvisited nodes
                                                      step,             # counter for the number of steps taken, for naming the plots
                                                      solution_steps,   # list of PathDistance objects representing the solution as it builds
                                                      plot_steps,       # specify whether we need to plot the steps, individually
                                                      line_segment,      # specify whether we need to calc distance to a finite line (=True) or infinite line (=False)
                                                      solution_directory_prefix                                                      
                                                      )
    # end of recursive_closest_edge_insertion() method  

    def _find_closest_edge(self, tour: PathDistance, vertex: Vertex, line_segment=False):
        '''
        Finds the closest edge to a given vertex in a given PathDistance object.
        Returns the edge in format '1-2' for vertices 1 and 2.

        Parameters:
        tour (PathDistance): the current tour, often a cyclical path
        vertex (Vertex): the vertex for which we want to find the closest edge
        line_segment (bool): whether we want to calculate the distance to a line segment or an infinite line

        NOTE: This method is intended to be called by the _recursive_closest_edge_insertion() method.
        '''
        # Actually, we want to find the closest edge to a vertex in a given PathDistance object.
        minimum_distance = float('inf') # set to infinity, a more pythonic way to do this

        # initialize the closest edge to None
        closest_edge = None

        # iterate through all edges to find the closest edge
        for edge_key, edge_points in self.calculate_edges(tour).items():

            # find the distance from the vertex to the edge
            distance = self._point_to_edge_distance(vertex, edge_points, line_segment)


            if distance < minimum_distance:
                minimum_distance = distance
                closest_edge = edge_key

        return closest_edge
    # end of _find_closest_edge() method


    def _point_to_edge_distance(self, vertex: Vertex,
                                edge_points: tuple[tuple[float, float], tuple[float,float]], 
                                line_segment=False):
        '''
        Calculates the distance from a point to an edge.
        The edge points are a tuple of two tuples of (x, y) coordinates.
        These are obtained via .calculate_edges() method for TSPMap object.
        Returns the distance.

        Parameters:
        vertex (Vertex): the vertex for which we want to find the distance to the edge
        edge_points (tuple): a tuple of two tuples of (x, y) coordinates
        line_segment (bool): whether we want to calculate the distance to a line segment or an infinite line

        NOTE: Caveat is whether we mean an infinite line or a line segment.
        NOTE: Uses of 'infinite line' assumptions for an edge may give odd results.
        NOTE: These odd results in TSP show up as cross-overs.
        '''
        # point is a tuple of (x, y) coordinates
        # edge_points is a tuple of two tuples of (x, y) coordinates
        # we calculate the distance from the point to the line segment defined by the edge points
        # we return the distance; source: https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html

        # for comment purposes, let A = (x1, y1), B = (x2, y2), and P = (x0, y0)

        # unpack the edge points
        x1, y1 = edge_points[0]
        x2, y2 = edge_points[1]

        # unpack the point
        x0, y0 = vertex.coords

        # together, these two differences define the vector AB.
        # AB points from A to B
        diff_x = x2 - x1
        diff_y = y2 - y1

        # distance between the two edge points
        numerator = abs(diff_x * (y1 - y0) - (x1 - x0) * diff_y)
        edge_length = math.sqrt(diff_x**2 + diff_y**2)

        if not line_segment:
            # The edge line is treated as infinite (i.e., not a line segment)

            return numerator / edge_length
        else:


            ######################
            # PROJECTION SCALAR
            ######################
            # Explanation of what the projection scalar is.
            # The edge is treated as a line segment.
            # Compute the projection scalar. This tells us how 'aligned' the vect
            # The projection scalar involves the dot product between AP and AB
            # The dot product between AP and AB tells us how aligned AP is with AB, i.e.,
            # ... how much of AP is in the direction of AB.
            # Since the trigonometric definition of the dot product gives us the angle between AP and AB, we know this:
            # if the <AP, AB> = 0, then AP is orthogonal to AB, and the projection point is at A
            # if the dot product <AP, AB> is positive, then the projection point is beyond A (pointing in same direction as AB)
            # if the dot product <AP, AB> is negative, then the projection point is beyond B (pointing in opposite direction as AB)
            # Next, we divide the projection scalar by the magnitude of the edge.
            # This normalizes how 'well-aligned' AP is with AB in terms of the length of AB.
            # If the projection scalar is in the interval [0, 1], we can find the point on the edge.
            # Thus, to find the projection point, we multiply the projection scalar by AB.
            # Otherwise, if the projection scalar is less than 0, the closest point would be A.
            # And if the projection scalar is greater than 1, the closest point would be B.


            # print all the variable values: FOR TESTING
            #print(f"Point: {vertex.label}, Edge: {edge_points}")
            #print(f"x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, diff_x: {diff_x}, diff_y: {diff_y}, edge_length: {edge_length}")

            projection_scalar = ((x0 - x1) * diff_x + (y0 - y1) * diff_y) / (edge_length**2)

            #######################
            #   Test if the point P is closer to one of the end points A or B
            #######################

            if projection_scalar < 0:
                # The projection point is "behind" A,
                # so we compute the distance from the point to the first edge point
                # NOTE: This is a more approximate measure, and differs a lot from an infinite line
                distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

                return distance

            elif projection_scalar > 1:
                # The projection point is beyond the second edge point, B.
                # so we compute the distance from the point to the second edge point
                distance = math.sqrt((x2 - x0)**2 + (y2 - y0)**2)

                return distance
            
            elif 0 <= projection_scalar <= 1:
                # the point is within the line segment, so we compute the distance from the point to the projection point
                # that is, the point on the line segment to which we projected the point
                # Find the projection points on the segment
                x_proj = x1 + projection_scalar * diff_x
                y_proj = y1 + projection_scalar * diff_y
        
                # Compute the distance from the point to the projection point on the segment
                distance = math.sqrt((x_proj - x0)**2 + (y_proj - y0)**2)

                return distance
    # end of _point_to_edge_distance() method


    # The following method calculates the coordinates of the edges of a given PathDistance object.
    def calculate_edges(self, tour: PathDistance):
        '''
        Calculates the edges of a given PathDistance object.
        Returns a dictionary of edges, where the key is in format '1-2' for vertices 1 and 2.
        The values are pairs of vertex coordinates, which are tuples of (x, y) coordinates.
        The coordinate pairs allows us to calculate point-to-edge distances.

        Parameters:
        tour (PathDistance): the current tour, often a cyclical path

        NOTE: distinct from the PathDistance's private _calculate_edge() method.
        NOTE: The purpose of the _calculate_edge() method for paths is to enable easy instantiation of a new PathDistance object.
        '''
        # Dictionary of edges, where the key is in format '1-2' for vertices 1 and 2
        # values are pairs of vertex coordinates
        edge_dict_coords = {}

        # get our coordinates from the path string
        coords = self.str_to_coords(tour)

        # vertices list
        vertex_labels = tour.current_path.split('-')


        # iterate through our coordinates and build our edge dictionary
        for i in range(len(coords) - 1):
            # edge key is in format '1-2' for vertices 1 and 2
            edge = vertex_labels[i] + '-' + vertex_labels[i+1]

            # edge values are pairs of vertex coordinates
            edge_dict_coords[edge] = (coords[i], coords[i+1])

        return edge_dict_coords
    # end of calculate_edges() method



    def _edge_insert(self, tour: PathDistance, edge: str, vertex: Vertex):
        '''
        Inserts vertex into edge of a given PathDistance object.
        Reconstructs object and returns new object with updated path and distance.
        Uses the PathDistance object's _edge_insert() method to return a dictionary of edges,...
        with key = starting vertex label, and value = ending vertex label.
        This diction of edges defines the new PathDistance object after the insertion.
        The TSP object will be responsible for rebuilding the edge distances, hence its own _edge_insert() function.
        
        Parameters:
        tour (PathDistance): the current tour, often a cyclical path
        edge (str): the edge to insert the vertex into, in format '1-3' for, e.g., vertices 1 and 3
        vertex (Vertex): the vertex to insert into the edge

        Returns:
        PathDistance object with the updated path and distance

        NOTE: This reconstruction assumes no repeated vertices EXCEPT for the starting and ending vertices.
        '''
        # tour is a PathDistance object, which is a path that's currently being travelled
        # by 'tour', we mean a circular path... so we can insert a vertex between two vertices in the path
        # edge is a tuple of two vertex labels, which represent the edge we want to insert
        # vertex is the Vertex object we want to insert between the two vertices in the edge
        # we will return a new PathDistance object with the updated path and distance
        # We use the PathDistance object's _edge_insert() method to do this,
        # but the TSP object will be responsible for rebuilding the edge distances...

        edge_dict = tour._edge_insert(edge, vertex)

        edge_dict_keys = list(edge_dict.keys())

        # key to starting vertex
        starting_vertex_key = edge_dict_keys[0]
        # our new tour will start with the first key (which is the starting vertex label)

        # get starting vertex from the nodes dict and make new PathDistance object
        new_tour = PathDistance(starting_vertex_key, self[starting_vertex_key], distance = 0.0)

        # iterate for as many times as we have edges (i.e., same as number of keys)
        for _ in range(len(edge_dict_keys)):

            # The next edge point label to add is found using current vertex of our tour.
            # and we use this point label as a key to access the next vertex in the self.__nodes
            # I name my variables explicitly to show this.
            next_vertex_key = edge_dict[new_tour.current_vertex.label]
            next_vertex = self[next_vertex_key]        
            new_tour += next_vertex # add the next vertex to our tour
   
        return copy.deepcopy(new_tour) # return the new tour, which is a PathDistance object  
    # end of _edge_insert() method
    





    ################################
    # BASIC PLOTTING FOR TSPMAP
    ################################

    # object method to plot a particular path over the problem map
    def plot_path(self, path: PathDistance, plot_title: str, save=False): 
        plot = _plot_coords(self.map_coords,       # TSPMap coords
                    self.str_to_coords(path),          # path coords
                    self.nodes.keys(),             # vertex labels
                    title = plot_title,
                    write=save,
                    plot_type="path")
        return plot


    def plot_map(self, save=False): 
        plot = _plot_coords(self.map_coords,       # TSPMap coords
                    None,                          # path coords
                    self.nodes.keys(),             # vertex labels
                    title = self.name,
                    write=save,
                    plot_type="map")
        return plot



    def plot_path_heatmap(self, paths: list[PathDistance], number_of_paths):
        '''
        For use with the genetic algorithm, this method will plot a heatmap of the paths.
        Also utilized in the wisdom of crowds.
        
        Arguments:
        paths (list): list of PathDistance objects
        number_of_paths (int): number of paths in the list
        '''

        ##############################
        # Initialize a graph object
        ##############################
        G = nx.Graph()

        
        # Iterate through all paths and plot the paths
        for path in paths:
            
            # coordinate list
            coords = self.str_to_coords(path)
            
            # Add edges and track node frequencies
            for i in range(len(coords) - 1):
        
                # since we have a list of tuples as coords, we can easily create our nodes
                node1 = coords[i]
                node2 = coords[i+1]
        
                # Add edges to graph
                if G.has_edge(node1, node2):
                    G[node1][node2]['weight'] += 1
                else:
                    G.add_edge(node1, node2, weight=(0.01*number_of_paths)) # adjusted the default alpha value via trial and error
                    
        # Prepare node positions
        pos = {n: n for n in G.nodes()}

        # Draw nodes with transparency (alpha) based on their frequency

        nx.draw_networkx_nodes(G, pos, node_size=10, node_color='blue', alpha=1)
        

        # Get the weights of the edges
        edge_weights = np.array([G[u][v]['weight'] for u, v in G.edges()])

        # Normalize edge weights for transparency; we divide by the max weight, which is really the number of paths
        # (since an edge only appears once per path
        max_weight = number_of_paths
        edge_alphas = edge_weights / max_weight  # Normalize for alpha

        # Adding a color scale (viridis) to the edges in addition to alpha
        cmap = plt.cm.viridis_r
        norm = plt.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
        
        # Draw edges with both alpha and colormap
        for (u, v), alpha, weight in zip(G.edges(), edge_alphas, edge_weights):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=2, alpha=min(1, max(0, alpha)), edge_color=[weight], edge_cmap=cmap, edge_vmin=edge_weights.min(), edge_vmax=edge_weights.max()
            )
        
        # Add a colorbar using the edges
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(edge_weights)
        plt.colorbar(sm, ax=plt.gca(), label="Edge Weight")
        
        # Show the beautiful plot!
        plt.title(f"Path Heatmap for {self.dimension}-size TSP")


 


    ######################
    #   CLASS METHODS    #
    ######################
        
    # in case you looked here for some class methods, to make a class subscriptable, I implemented a metaclass

    # The following methods utilize the TSP algorithms defined above in order to iterate through the entire directory of .tsp files
    # Thus, it will call the algorithms, find the minimum (or maximum) solutions, and return data for plotting.
    # Specifically, what I intend to have are:
    # 1) for each solution, a list of tuples of the (x,y) coordinates, which may then allow plotting of the solution.
    # 2) for a list of tuples of (dimensionality, time) corresponding to the .tsp problem files.
    #
    #
    #
    # The returned data in 1) and 2) can be used to visualize the solution, as well as the time complexity vs dimensionality of the problem

    @classmethod
    def brute_force_all(cls, file_ext: str, write_all_paths=False, write_all_mins=False, write_dim_time=False):
        '''
        Performs the brute_force() algorithm on all TSPMaps, instantiated by files matched by passed parameter.
        
        Parameters:
        file_ext (string)
        
        Keywords:
        write_all_paths (default =False) saves the returned all path values to a *.txt file (bfs = brute force)
        write_all_mins  (default =False) saves all min paths to .txt file
        write_dim_time  (default =False) saves all (dimension, time) tuples to a .txt file


        Returns:
        1) Dictionary of lists of all possible PathDistance objects, where each list is keyed to TSPMap name
        2) Dictionary of the minimum path solution keyed to the corresponding TSPMap name      
        3) Dictionary of tuples containing (dimensionality, time) for each respective TSPMap object

        '''
        all_paths_dict = {} # for all permutations for each TSPMap
        all_min_paths_dict = {} # for the minimum paths for each TSPMap
        all_dim_time_dict = {} # for all (dimensionality, time) tuples for each TSPMap

        # file_ext used by glob() to match file names via Unix shell rules, which is simpler
        # pass *.tsp to match all .tsp files!
         
        # iterate through all .tsp, create TSPMap objects, and call brute_force() for each object...
        # before doing anything with the TSPMaps, let's save them first.
        files = glob.glob(file_ext)        
        for file in files:
            cls._batch[file] = TSPMap(file)
        
        # after reading in the files and making our TSPMap objects, we can now find our solutions.             
        # IMPORTANT: I start timing on the line before brute_force() call and end after assignment of minimum path.
        for tsp in cls._batch.values():
            print(f"Brute forcing solution(s) for {tsp.name}")
            _ = time.perf_counter() # start timer using performance counter from time module
            all_paths_dict[tsp.name] = tsp.brute_force() # brute_force() call on TSPMap instance; key all permutations to dict
            all_min_paths_dict[tsp.name] = min(all_paths_dict[tsp.name]) # find the minimum path and key it to dict
            elapsed_time = time.perf_counter() - _ # stop timer and find the elapsed time
            
            all_dim_time_dict[tsp.name] = (tsp.dimension, elapsed_time) # for plotting (dimension, time); keyed to dict
        
        # Write to a file, if we set the various keywords to =True
        if write_all_paths:
            with open("brute_force_all_paths.txt", 'w') as file:
                for key in cls._batch.keys():
                    file.write(f"{key} all paths\n")
                    for path in all_paths_dict[key]:
                        file.write(str(path) + '\n')
        if write_all_mins:
            with open("brute_force_minimum_paths.txt", 'w') as file:
                for key, value in all_min_paths_dict.items():
                        file.write(f"{key} {value}\n")
        if write_dim_time:
            with open("brute_force_dimension_vs_time.txt", 'w') as file:
                for key, value in all_dim_time_dict.items():
                    file.write(f"{key} {value}\n")

        return all_paths_dict, all_min_paths_dict, all_dim_time_dict
    # END brute_force_all()

# ================================================ #
#               Inherited Class                 
# ================================================ #

# Whereas the TSPMap did not define edges, and thus simulated a complete undirected graph,
# here, we have a class that inherits TSPMap but now uses a dictionary to store a hard-coded dictionary of edges
# The dictionary of edges use a starting vertex's label as a key.
# The values will be a list of PathDistance objects indicating a connection point between the start vertex and another vertex
# The current_vertex attribute of the PathDistance object can be used to get the ending vertex of our edge

# Algorithms implemented with this class include e.g., bread-first and depth-first search


class TSPMapWithEdges(TSPMap):

    def __init__(self, tsp_file: str, edges: dict[str, list[PathDistance]], skippable_lines=7):
        # call parent constructor
        # this should build the self.__nodes containing all relevant Vertex objects
        super().__init__(tsp_file, skippable_lines)
        
        # Create dictonary with PathDistance objects
        self.__edges = {}

        for vertex, tuples_list in edges.items():
            edge_list = []                  # initialize our empty list
            self.__edges[vertex] = edge_list
            for edge_tuple in tuples_list:
                # append an new PathDistance object to our list, keyed by the starting vertex
                # we are actually adding two Vertex objects, which gives us a new PathDistance object
                # recall that self[] uses TSPMap object magic method for __getitem__ to access the __nodes dict
                self.__edges[vertex].append(self[vertex] + self[ str(edge_tuple[1]) ]) 

    @property
    def edges(self):
        return self.__edges
    

    # This private function is utilize in the uninformed searches.
    # I realize that the 'edges' parameter is unnecessary, since I could've used self.__edges instead.
    # So, I'll leave it for now.
    def _expand(self, current_path: PathDistance, edges: dict[str, list[PathDistance]]):
        '''
        Private function utilized by various graph search algorithms.
        This essentially 'expands' the frontier of next possible moves. It yields possible PathDistance objects
        Parameters:
        current_path = PathDistance object whose current_vertex is being expanded to generate possible next paths
        edges = dictionary with vertex labels as keys and lists of PathDistance objects whose current_vertex is a possible next vertex
         '''


        # key our dictionary to access the hard-coded edges!
        for edge in edges[current_path.current_vertex.label]:
            # possible paths will be an extension, or addition to, our current path travelled.
            possible_path = copy.deepcopy(current_path) # copy our path
            possible_path += edge.current_vertex        # add our next vertex
            yield possible_path


    
    def uninformed_search(self, start: str, goal: str, method="breadth-first"):
        '''
        Performs an uniformed search, the type of which may be defined
        Returns a PathDistance object representing n
        
        Parameters:
        start: label of the starting vertex
        goal: label of the goal vertex

        Keywords:
        method: specifies the type of uninformed search
                e.g., "breadth-first", "depth-first", "uniform-cost"
        '''
        if start == goal:
            return self[start] # if the start IS the goal, then return that vertex
       
       # instantiate a starting PathDistance object using our 'start' label
        initial_path = PathDistance(path_str = start, vertex = self[start], distance = 0.0)

        reached = {}
        # for this algorthm, the goal state is defined by the vertex labels.
        # Thus, we use a dictionary for the 'reached' states, where the key is the vertex label and the value is the PathDistance obj
        reached[start] = initial_path
        # The method-type largely influences the data structure for the frontier

        ###################################
        #       BREADTH-FIRST SEARCH
        ###################################
        if method == "breadth-first":   
            frontier = queue.Queue()   # breadth-first utilizies a FIFO queue for defining our frontier
            frontier.put(initial_path)

            while not frontier.empty():             # while our frontier still has options...is not empty...
                parent_path = frontier.get()        # dequeue our frontier queue to get a parent
                #print(f"Expanding {parent_path}")
                for child_path in self._expand(parent_path, self.__edges):    # expand parent to see next possible options (i.e., children)
                   state = child_path.current_vertex.label              # get child's state, which is a next possible path's current vertex
                   if state == goal:                                    # if the child reaches our goal, return our child PathDistance!
                       return child_path
                   if state not in reached.keys():      # otherwise, if the state is one we haven't been to...
                       reached[state] = child_path      # add new dictionary entry keyed by state
                       frontier.put(child_path)         # add the child path to the frontier
            raise SearchFailed("The breadth-first search failed to find the goal state")

        ###################################
        #       DEPTH-FIRST SEARCH
        ###################################
        elif method == "depth-first":
            frontier = []                               # we'll use a vanilla Python list as a stack
            frontier.append(initial_path)
            while frontier:                             # while our frontier still has options...is not empty...
                parent_path = frontier.pop()            # pop our frontier stack to get a parent
                #print(f"Expanding {parent_path}")
                for child_path in self._expand(parent_path, self.__edges):  # expand parent to see next possible options (i.e., children)
                    state = child_path.current_vertex.label                  # get child's state, which is a next possible path's current vertex
                    if state == goal:                                        # if the child reaches our goal, return our child PathDistance!
                       return child_path                                    # we don't need to use {reached} dictionary for depth-first search
                    frontier.append(child_path)                             # push the child path to the frontier stack
            raise SearchFailed("The depth-first search failed to find the goal state")

        ###################################
        #       UNIFORM-COST SEARCH
        ###################################
        elif method == "uniform-cost": # i.e., Dijkstra's algorithm
            frontier = queue.PriorityQueue()   # utilizies a priority queue for defining our frontier; priority is shortest distance
            frontier.put((initial_path.current_distance, initial_path))
            # note that priority queue is basically a list of tuples
            while not frontier.empty():             
                parent_path = frontier.get()[1] # using _, caused funky errors since PathDistance isn't iterable.. even though we get a tuple
                                                # So, I defined __getitem__ for PathDistance to return self and ignore key. This works. Somehow.
                                                # Weird...
                if parent_path.current_vertex.label == goal: # We know, since we pop the highest priority, that if it's at the goal, we return
                    return parent_path
                
                for child_path in self._expand(parent_path, self.__edges):  # expand parent to see next possible options (i.e., children)
                    state = child_path.current_vertex.label      # get child's state, which is a next possible path's current vertex
                    # The next if-statement basically says: 
                    # "If we haven't been to there, let's make note of it and add it to the frontier"
                    # OR "If we HAVE been there yet it's a shorter distance, then let's also make note of it and add it to the frontier
                    # "But if we've been there and it's a longer way, then ignore it."
                    if state not in reached.keys() or child_path.current_distance < reached[state].current_distance:
                        reached[state] = child_path      # add new dictionary entry keyed by state
                        frontier.put((child_path.current_distance, child_path))         # add the child path to the frontier
            raise SearchFailed("The uniform-cost search failed to find the goal state")
    # end uninformed_search()

    ##################################
    # TSPMapWithEdges Plotting Methods:
    ##################################

    ##################################
    #   plot_path() object method 
    ##################################
    # Plot a path for the problem, overlayed on top of the problem map
    def plot_path(self, path: PathDistance, plot_title: str, save=False): 
        ax = _plot_coords(self.map_coords,          # TSPMap coords
                    self.str_to_coords(path),       # path coords
                    self.nodes.keys(),              # vertex labels
                    title = plot_title,
                    write=False, # set to False to avoid double saving since we write below;
                    plot_type="path",
                    path_color='red'
                    )
        # iterate through our values, which are lists of PathDistance objects
        for edges in self.__edges.values():
            # iterate through our list to get our individual PathDistance objects
            for edge in edges:
                # for each PathDistance object, convert it to return a list of coordinates...            
                path_coords = self.str_to_coords(edge)
                # I know that my edges consist of two vertices... index 0 is starting vertex, index 1 is ending vertex
                start_coords = path_coords[0]
                end_coords = path_coords[1]
                # 'xy' is the arrowhead location, xytext is the start of the arrow
                # set zorder to -1 so that it is drawn first, so that it is behind the other elements
                ax.annotate('', xy=end_coords, xytext=start_coords, zorder = -1,
                            arrowprops=dict(lw=0.5, shrink = 0.05,
                            facecolor='lightgrey'))
       
        if save:
            # if we save the plot, we can use the title. But remove any suffixes first! (like .tsp) 
            plt.savefig(plot_title + '_map.png', format='png')
        else:
            plt.show()
    # end def plot_path()    

    ##################################
    #   plot_map() object method 
    ##################################
    # Plot the map, only. This shows the available vertices and their corresponding edges
    def plot_map(self, save=False): 
        ax = _plot_coords(self.map_coords,       # TSPMap coords
                    None,                          # path coords
                    self.nodes.keys(),             # vertex labels
                    title = self.name,
                    write=False,
                    plot_type="map")

        # iterate through our values, which are lists of PathDistance objects
        for edges in self.__edges.values():
            # iterate through our list to get our individual PathDistance objects
            for edge in edges:
                # for each PathDistance object, convert it to return a list of coordinates...            
                path_coords = self.str_to_coords(edge)
                # I know that my edges consist of two vertices... index 0 is starting vertex, index 1 is ending vertex
                start_coords = path_coords[0]
                end_coords = path_coords[1]
                # 'xy' is the arrowhead location, xytext is the start of the arrow
                # set zorder to -1 so that it is drawn first, so that it is behind the other elements
                ax.annotate('', xy=end_coords, xytext=start_coords, zorder = -1,
                            arrowprops=dict(lw=0.5, shrink = 0.05,
                            facecolor='lightgrey'))
        if save:
            # if we save the plot, we can use the title. But remove any suffixes first! (like .tsp) 
            plt.savefig(self.name + '_map.png', format='png')
        else:
            plt.show()
    # end plot_map() definition



#end TSPMapWithEdges class definition

# Class for raising exceptions when a search algorithm fails to return a solution
class SearchFailed(Exception):
    pass

    
        

# ================================================ #
#                Plotting functions 
# ================================================ #

# the following is a driver function to reduce code and clutter...
# ... when passing results from, e.g., brute_force_all() to plotting functions
def plot_results(all_paths_dict: dict[str, list[PathDistance]],
                 all_min_paths: dict[str, list[PathDistance]],
                 all_dim_time_dict: dict[str, list[tuple]],
                 write_runtime=False,
                 write_min_paths=False,
                 print_min_paths=False):
    '''
    Receives input from any of the TSPMap class method algorithms (e.g., brute_force_all()),
    and plots using matplotlib, with the option of saving plots as .png.

    Parameters:
    all_paths_dict: Dictonary of lists containing all PathDistance objects obtained via brute force
                    pass None, if no all_paths_dict exists
    all_min_paths: Dictionary of the minimum PathDistance object (implies symmetric path), keyed to TSPMap name
    all_dim_time_dict: Dictionary of tuples containing (dimension: int, time: float) for plotting runtimes
    
    Keywords:
    write_runtime: (default =False) saves the runtime plot to .png
    write_min_paths: (default =False) saves the min path plot to .png
    print_min_paths: (default =False) prints min path solutions to terminal
    '''
    if all_paths_dict:
        # for now, we don't do anything with all the paths, since that's specific to brute_force
        pass

    # Let's plot the runtime tuples first
    _plot_tuples(all_dim_time_dict, write_runtime)
    
    # Next, handle the min paths and maps
    for key, min_path in all_min_paths.items():
        if print_min_paths:
            print(f"The minimum path for {key} is ({min_path})")

        # call plotting function
        _plot_coords(TSPMap[key].map_coords,                   # pass map coords for TSPMap object. Lays out the territory.
                        TSPMap[key].str_to_coords(min_path),   # pass coordinates for min path, obtained from PathDistance obj
                        TSPMap[key].nodes.keys(),   # pass list of Vertex obj names, which we get from TSPMap's nodes property
                        title = key,                # each figure's title is TSPMap name, which is also the dictionary key
                        write=write_min_paths,      # specifies whether we want to save the plots as .png
                        plot_type='path',           # this specifies that we want to overlay path on TSPMap's nodes; uses plt.plot()
                        path_label='Minimum Path'   # provides a legend label to name the line drawn on the plt.plot()
                    )
# end plot_results()

#############################################
# Private functions called by the basic plotting functions
#############################################
def _plot_tuples(tuples_list, write=False):
    ''' 
    Generates plots given a list of tuples.
    Used for lists of tuples containing (dimension, time) for the runtime of each TSP problem
    
    Parameter:
    tuples_list: the list of (dimension, time) tuples, where (dimension: float, time: int)
    tsp_name: string containing the name of the TSPMap object, which is usually the key for relevant dicts
    
    Keywords:
    write (default =False): If set to True, writes plot to .png file in relative directory
    '''
    # check if we have a dict or list; if dict, get values; if still not list, raise type error
    if isinstance(tuples_list, dict):
        # we need to make sure our tuples are sorted in ascending order using the x-value for each tuple item
        # so, we can sort the list in place, where we specify the key to be the first element of each tuple
        # to do that, we can use a lambda function to access first element of each tuple by applying subscripting to each tuple
        # use sorted() since we are dealing with a view of the dictionary values
        tuples_list = sorted(tuples_list.values(), key = lambda x: x[0])
        
    elif isinstance(tuples_list, list):
        # if it's a list, use the list's sort() method
        tuples_list.sort(key = lambda x: x[0])
    
    else:
        # for everything else, raise a type error
        raise TypeError("Input parameter for plot_tuples() must be list or dictionary!")
   

    # unzip our (x, y) tuples into x and y lists, respectively
    dimension, time = zip(*tuples_list)
    
    #Create new figure object since we will be iterating with this function and making multiple Plot objects
    plt.figure() 

    # simple line graph
    plt.plot(dimension, time)

    # Add helpful aesthetics, like labels, title and legend
    plt.xlabel("N (number of vertices)")
    plt.ylabel("Time (seconds)")
    plt.title("Runtime Growth with Increasing N")
    
    if write:
        # if we save the plot, we can use the title. But remove any suffixes first! (like .tsp) 
        plt.savefig('TSP_runtime.png', format='png')
# end _plot_tuples() function


# The following function may be used later by a driver function to simplifying things
# Especially since I think it's too complicated, and I could hide more details outside of this module
# <<Future implementation>> call plot_coords with plot_results() using the returned results from brute_force_all()

def _plot_coords(tuples_list_map,
                tuples_list_path,
                vertex_labels,
                x_lab="x-axis",
                y_lab="y-axis",
                title="XY Map",
                write=False,
                suffix='.tsp',
                plot_type='map',
                path_label='path',
                path_color='blue',
                prefix=''
                ):
    '''
    Plots a list of tuples using matplotlib.pyplot
    Useful for plotting paths or the run times of the TSP problems

    Parameters:
    tuples_list_map (list[tuple]): List of tuples as (x, y) for the map
    tuples_list_path (list[tuple]): List of tuples as (x, y) for the paths

    Keywords:
    x_lab (str): Provides x_axis label
    y_lab (str): Provides y_axis label
    title (str): Gives title of the plot
    write (default =False): Saves plot as PNG file
    plot_type (default 'scatter'): can be map or path 
    '''
    ptype = plot_type.lower()

    # We must unzip our list of tuples using zip(*list) into two lists
    # The following are values for our TSP map
    x_list, y_list = zip(*tuples_list_map) 

    # create a new figure
    plt.figure()

    if ptype == 'map' or ptype != 'path':
        plt.scatter(x_list, y_list)
    
    elif ptype == 'path':    
        # Unzip values for our path
        x_path, y_path = zip(*tuples_list_path)
        
        plt.scatter(x_list, y_list)   # plot scatter
        plt.plot(x_path, y_path, label = path_label, color=path_color)      # overlay line

    # Label each point
    for i, label in enumerate(vertex_labels):
        plt.annotate(label, # annotation text
                     (x_list[i], y_list[i]), # coordinates at which we add the annotations
                     textcoords="offset points", # annotations will be offset
                     xytext=(0,7), # the offset distance of the annotation specified here by 13-points on y
                     ha='center',   # horizontal alignment is centered over point
                     fontweight='bold',
                     fontsize='12')   
    plt.draw()
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.legend()
    
    if write:
        # if we save the plot, we can use the title. But remove any suffixes first! (like .tsp)
        path = prefix + title.removesuffix(suffix) + '_map.png'  
        plt.savefig(path, format='png')

    return plt.gca()
# end _plot_coords() definition


def triangle_area(tuples: list[tuple[float, float]]):
    # Unpack the points into x and y coordinates
    x1, y1 = tuples[0]
    x2, y2 = tuples[1]
    x3, y3 = tuples[2]
    
    # Apply the Shoelace formula
    area = abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2)
    
    return area

def distance(tuple1: tuple[float, float], tuple2: tuple[float, float]):
    '''
    Returns the Euclidean distance between two points in 2D space.
    '''
    return math.sqrt((tuple1[0] - tuple2[0])**2 + (tuple1[1] - tuple2[1])**2)




# ================================================ #
#              ITEM CLASS DEFINITION
# ================================================ #




# ================================================ #
#              BAG CLASS DEFINITION
# ================================================ #






# ================================================ #
#              TKINTER GUI FUNCTIONS
# ================================================ #
# ==================================================================== #
# ====================================================================
# BEGIN ----- TKINTER FUNCTION FOR GENETIC ALGORITHM AND WISDOM OF THE CROWDS
# ====================================================================


def _run_genetic_algorithm():
    #############################################
    # Function for running the genetic algorithm
    #############################################
    
    # directories for results
    os.makedirs('results', exist_ok=True)

    # filename
    filename = entry_filename.get()
    run_title = entry_run_title.get()

    run_directory = f'results/{run_title}_{filename.rstrip(".tsp")}'    # unique directory for a given run, specified by run_title and filename
    
    # directories for runs
    os.makedirs(run_directory, exist_ok=True)
    os.makedirs(f'{run_directory}/csv', exist_ok=True)          # for all .csv data and parameter notes output 
    os.makedirs(f'{run_directory}/csv/ga', exist_ok=True)       # fitness trend data for individual runs
    os.makedirs(f'{run_directory}/csv/events', exist_ok=True)   # parent reproductive event distribution data for individual runs
    os.makedirs(f'{run_directory}/csv/final', exist_ok=True)    # This is where we place the aggregated data among all runs

    ##########################################################################
    # Parameters:
    ############

    G = int(entry_G.get())
    N = int(entry_N.get())
    percent_preserved = float(entry_percent_preserved.get())
    lambda_value = float(entry_lambda_value.get())
    xover_method = entry_xover_method.get()
    lower = float(entry_lower.get())
    upper = float(entry_upper.get())
    mutation_rate = float(entry_mutation_rate.get())
    abominate_threshold_value = int(entry_abominate_threshold_value.get())
    abomination_percentage_value = float(entry_abominate_percentage.get())
    runs = int(entry_runs.get())

    use_tours = entry_use_tours.get().lower()

    ###########################################################################

    # save the parameters for this run as .csv
    with open(f'{run_directory}/csv/parameters_{filename}.csv', 'w') as file:
        file.write("G,N,percent_preserved,lambda_value,xover_method,lower,upper,mutation_rate,abomination_threshold_value,runs\n")
        file.write(f"{G},{N},{percent_preserved},{lambda_value},{xover_method},{lower},{upper},{mutation_rate},{abominate_threshold_value},{runs}")
    
    # create our TSPMap object
    problem = TSPMap(filename)

    ####################
    # Best paths of run series
    ####################

    best_path = PathDistance('', distance = float('inf'))

    worst_path = PathDistance('', distance = float('-inf'))
    
    # BEGIN GA RUN SERIES
    
    latest_run = 0

    with open(f'{run_directory}/csv/run_series_best_worst_and_runtime.csv', 'w') as file:
        file.write('run_number,best_path,best_distance,worst_path,worst_distance,run_time\n')

        run_start = time.perf_counter()
        for run in range(runs):
            print(f"Run {run} of {runs}")
            latest_run = run # to reference specific .csv to plot. We plot the late run's individual data for demonstration purposes
            
            _ = time.perf_counter()    
            best, worst, last_pop, edge_counts = problem.genetic_algorithm(generations = G,
                                population_size = N,
                                subpop_proportion = percent_preserved,
                                lambda_param = lambda_value,
                                crossover_method = xover_method,
                                split_lower = lower,
                                split_upper = upper,
                                mutation_prob = mutation_rate,
                                abominate_threshold = abominate_threshold_value,
                                save_data=True,
                                file_path = run_directory +'/csv',
                                run_number = latest_run,
                                abomination_percentage = abomination_percentage_value,
                                tours = use_tours
                                )
            elapsed_t = time.perf_counter() - _
            
            # record data on best and worst per run
            file.write(f'{latest_run},{best.current_path},{best.current_distance},{worst.current_path},{worst.current_distance},{elapsed_t}\n')
            # Same the best and worst path of this run
            if best < best_path:
                best_path = best
            if worst > worst_path:
                worst_path = worst

            #print(f"Single Runtime: {elapsed_t} seconds")# for testing
        run_elapsed = time.perf_counter() - run_start

        print(f"Total runtime for the series of runs: {run_elapsed}")

        print("Plotting the best and worst path from the run series...")

        problem.plot_path(best, 'Example of Best: Distance = ' + str(best.current_distance) + '\nN = ' + str(problem.dimension), save=False)
        plt.savefig(f'{run_directory}/best_{str(best.current_distance)}.png')
        plt.show()

        problem.plot_path(worst, 'Example of Worst: Distance = ' + str(worst.current_distance) + '\nN = ' + str(problem.dimension), save=False)
        plt.savefig(f'{run_directory}/worst_{str(worst.current_distance)}.png')
        plt.show()

        # END GA RUN SERIES

        # Plot the heatmap of the last population
        print("Plotting the heatmap of the last population...")
        last_pop_list = list(last_pop.values())
        problem.plot_path_heatmap(last_pop_list, N)
        plt.savefig(f'{run_directory}/last_population_heatmap.png')
        plt.show()

        print("The ranked population:")
        for rank, path in last_pop.items():
            print(f"rank: {rank}, path: {path}")

        print("Run edge counts saved to CSV...")
        # Save the edge counts to a CSV file
        # Convert the dictionary to a DataFrame
        edge_counts_df = pd.DataFrame(list(edge_counts.items()), columns=['Edge', 'Count'])     
        edge_counts_df.to_csv(f'{run_directory}/csv/run_{latest_run}_edge_counts.csv')
    ##################################################################
    # PLOT THE MAIN DATA HERE
    ##################################################################
    # read in the .csv that our GA wrote
    complete_rank_offspring_df = pd.read_csv(f'{run_directory}/csv/events/run_{latest_run}_parent_rank_offspring_number_df.csv', index_col=0)

    # also a warning called for some function called from within the seaborn module. It doesn't affect the output, so we suppress it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        # Reset the DataFrame index to ensure proper plotting in Seaborn
        df_reset = complete_rank_offspring_df.reset_index(drop=True)


        # Melt the DataFrame, allowing seaborn to handle the data 
        df_melted = pd.melt(df_reset, var_name='Column', value_name='Events')


        # Plot the line graph with Seaborn (now using errorbar='sd' for shaded error bars)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_melted, x='Column', y='Events', errorbar='sd')

        # Add labels and title
        plt.xlabel('Parent Ranks')
        plt.ylabel('Reproductive Events')
        plt.title(f'Example Distribution of Reproductive Events by Parent Ranks\nLambda = {lambda_value}' )
        plt.savefig(f'{run_directory}/example_reproduction_distribution_figure')
        plt.show()


    
    ############################################
    # GA Run Data Plot
    ############################################

    df = pd.read_csv(f'{run_directory}/csv/ga/run_{latest_run}_ga_run_df.csv')

    # Plotting
    x = df['generation']  # X-axis (row identifiers)
    y = df['rank_average']  # Average
    sem = df['rank_SEM']  # Standard Error of the Mean

    plt.plot(x, df['rank_min'], label='Min', color='blue', linestyle='--')
    plt.plot(x, df['rank_max'], label='Max', color='red', linestyle='--')
    plt.plot(x, y, label='Average', color='green')

    # Plot the SEM as shaded region
    plt.fill_between(x, y - sem, y + sem, color='green', alpha=0.2, label='±SEM')

    # Add labels and legend
    plt.xlabel('Generation #')
    plt.ylabel('Distance')
    plt.title('Population Fitness with Increasing Generations')
    plt.legend()
    plt.savefig(f'{run_directory}/example_population_fitness_trend_figure')
    plt.show()

    ################################################
    # Process final data for GA run series fitness trend
    ################################################
    # Aggregation of fitness trend across each GA run
    # The process is simple. We concatenate all the GA trend data and group by the generation index.
    # The aggregation will be a simple mean
    # Appropriate SEMs will also be added for the rank_min and rank_max

    # Read all the CSV files in the csv/ga directory
    file_list = glob.glob(f'{run_directory}/csv/ga/*.csv')

    # Initialize an empty DataFrame for storing aggregated data
    df_list = []

    # Loop through each CSV file, reading the data
    for file in file_list:
        df = pd.read_csv(file, usecols=['generation', 'rank_min', 'rank_max', 'rank_average', 'rank_SEM'])
        # Append to the list of DataFrames
        df_list.append(df)

    # Concatenate all DataFrames together
    all_data = pd.concat(df_list)

    # Group by 'generation' and calculate the mean of rank_min, rank_max, rank_average
    aggregated_data = all_data.groupby('generation').agg({
        'rank_min': 'mean',
        'rank_max': 'mean',
        'rank_average': 'mean'
        }).reset_index()

    # Calculate the new SEM based on the averages
    # SEM = Standard Error of the Mean, calculated as: stddev / sqrt(n)
    # We'll need to calculate this for rank_average, rank_min and rank_max
    
    # SEM of average
    aggregated_data['rank_SEM'] = all_data.groupby('generation')['rank_average'].apply(
        lambda x: np.std(x) / np.sqrt(len(x))
        ).values

    # SEM of rank_min
    aggregated_data['rank_min_SEM'] = all_data.groupby('generation')['rank_min'].apply(
        lambda x: np.std(x) / np.sqrt(len(x))
        ).values
    
    # SEM of rank_max
    aggregated_data['rank_max_SEM'] = all_data.groupby('generation')['rank_max'].apply(
        lambda x: np.std(x) / np.sqrt(len(x))
        ).values
    # Step 6: Save the aggregated DataFrame to a CSV file
    aggregated_data.to_csv(f'{run_directory}/csv/final/ga_final_df.csv', index=False)

    print("Aggregation complete! Run series fitness trend data saved as 'ga_final.csv'.")


    ############################################
    # GA Run Series FINAL fitness trend Data Plot
    ############################################


    df = pd.read_csv(f'{run_directory}/csv/final/ga_final_df.csv')

    # define our data
    x = df['generation']  # X-axis (row identifiers are the generation index)
    y = df['rank_average']  # Average
    y_min = df['rank_min']
    y_max = df['rank_max']

    sem = df['rank_SEM']              # Standard Error of the Mean for the average
    min_rank_sem = df['rank_min_SEM'] # SEM for the min rank of our aggregated run data
    max_rank_sem = df['rank_max_SEM'] # SEM for the max rank

    plt.plot(x, y_min, label='Min', color='blue', linestyle='--')
    plt.plot(x, y_max, label='Max', color='red', linestyle='--')
    plt.plot(x, y, label='Average', color='green')

    # Plot the SEM as shaded region
    plt.fill_between(x, y - sem, y + sem, color='green', alpha=0.2, label='±SEM')
    plt.fill_between(x, y_max - max_rank_sem, y_max + max_rank_sem, color='red', alpha=0.2, label='± max SEM')
    plt.fill_between(x, y_min - min_rank_sem, y_min + min_rank_sem, color='blue', alpha=0.2, label='± min SEM')

    
    # Add labels and legend
    plt.xlabel('Generation #')
    plt.ylabel('Distance')
    plt.title('Population Fitness with Increasing Generations')
    plt.legend()
    plt.savefig(f'{run_directory}/final_population_fitness_trend_figure')
    plt.show()


    ############################################
    # Parent Reproductive Events Run Series FINAL Data Plot
    ############################################

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # Load and aggregate CSV files
        path = f'{run_directory}/csv/events'  # Adjust to your folder path
        all_files = glob.glob(f"{path}/*.csv")

        dataframes = []
        for file in all_files:
            df = pd.read_csv(file, index_col="generation")
            dataframes.append(df)

        # Concatenate all DataFrames!
        combined_df = pd.concat(dataframes, axis=1)

        # Calculate averages and SEMs, ignoring NaNs
        avg_df = combined_df.groupby('generation').mean()

    # Final aggregation - collapse all rows for a single average histogram
    # Suppress FutureWarnings as needed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        # Reset the DataFrame index for plotting
        df_reset = avg_df.reset_index(drop=True)

        # Melt the DataFrame for Seaborn
        df_melted = pd.melt(df_reset, var_name='Column', value_name='Events')

        # Step 4: Plotting with Seaborn
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_melted, x='Column', y='Events', errorbar='sd')

        # Add labels and title
        plt.xlabel('Parent Ranks')
        plt.ylabel('Reproductive Events')
        plt.title('Average Distribution of Reproductive Events by Parent Ranks')
        plt.savefig(f'{run_directory}/run_series_average_reproduction_distribution_figure')
        plt.show()

# End run_genetic_algorithm()

def _plot_final():

    # filename
    run_directory = run_directory_entry.get()
    df = pd.read_csv(f'{run_directory}/csv/final/ga_final_df.csv')

    # define our data
    x = df['generation']  # X-axis (row identifiers are the generation index)
    y = df['rank_average']  # Average
    y_min = df['rank_min']
    y_max = df['rank_max']

    sem = df['rank_SEM']              # Standard Error of the Mean for the average
    min_rank_sem = df['rank_min_SEM'] # SEM for the min rank of our aggregated run data
    max_rank_sem = df['rank_max_SEM'] # SEM for the max rank

    plt.plot(x, y_min, label='Min', color='blue', linestyle='--')
    plt.plot(x, y_max, label='Max', color='red', linestyle='--')
    plt.plot(x, y, label='Average', color='green')

    # Plot the SEM as shaded region
    plt.fill_between(x, y - sem, y + sem, color='green', alpha=0.2, label='±SEM')
    plt.fill_between(x, y_max - max_rank_sem, y_max + max_rank_sem, color='red', alpha=0.2, label='± max SEM')
    plt.fill_between(x, y_min - min_rank_sem, y_min + min_rank_sem, color='blue', alpha=0.2, label='± min SEM')

    
    # Add labels and legend
    plt.xlabel('Generation #')
    plt.ylabel('Distance')
    plt.title('Population Fitness with Increasing Generations')
    plt.legend()
    plt.show()

###########################################
# WISDOM OF THE CROWDS
###########################################


def _run_crowd_member(
            crowd_df_data,      # for crowd data to be stored in CSV
            crowd_path_list,    # for storing the PathDistance objects   
            crowd_member_id,    # thread id
            problem_map,        # TSPMap object
            G,                  # generations
            N,                  # population size
            percent_preserved,  # proportion of population preserved
            lambda_value,       # lambda parameter
            xover_method,       # crossover method
            lower,              # split lower
            upper,              # split upper
            mutation_rate,      # mutation rate
            abominate_threshold_value,          # abominate threshold
            abomination_percentage_value,       # abomination percentage
            use_tours,      # use tours
            verbose,        # verbose
            ):
    '''
    This function represents one member in a crowd.
    The crowd member will run the genetic algorithm with the parameters specified in the GUI.
    '''

    ###########################################################################

    # create our TSPMap object
    problem = problem_map

    ####################
    # The following represents one of the individuals in the crowd
    ####################
    # Store this thread's result here, which will be appended to the crowd_df_data list
    # the crowd_df_data list will be used to create a DataFrame for the entire crowd and saved as CSV
    local_df_data = []

    # For manipulations with the actual PathDistance objects obtained from each member of the crowd,....
    # We store to the local_path list
    local_path = []


    run_start = time.perf_counter()    

    local_best_path, _, _, _ = problem.genetic_algorithm(generations = G,
                        population_size = N,
                        subpop_proportion = percent_preserved,
                        lambda_param = lambda_value,
                        crossover_method = xover_method,
                        split_lower = lower,
                        split_upper = upper,
                        mutation_prob = mutation_rate,
                        abominate_threshold = abominate_threshold_value,
                        save_data=False,    # We don't want to save data from this individual crowd members... causes problems with file writing
                        run_number = crowd_member_id,
                        abomination_percentage = abomination_percentage_value,
                        tours = use_tours,
                        verbose = verbose
                        )
    
    elapsed_t = time.perf_counter() - run_start

    # We record the best path's data and the run time for this individual thread!
    local_df_data.append((crowd_member_id,                  # thread id
                        local_best_path.current_path,     # best PathDistance object's path string
                        local_best_path.current_distance, # best PathDistance object's distance
                        elapsed_t                         # Run time for our individual thread
                        )) 
    crowd_df_data.extend(local_df_data)


    # Here, we append the best PathDistance object to the local_path list
    local_path.append(local_best_path)
    crowd_path_list.extend(local_path)




def woc_start_threads():
    '''
    This function starts the threads for the Wisdom of the Crowds (WoC) genetic algorithm.
    Each thread represents a member of the crowd.
    '''
    
    #############################################
    # Create directories for storing results
    #############################################

    # directories for results
    os.makedirs('results_woc', exist_ok=True)

    # filename
    filename = entry_filename.get()
    run_title = entry_run_title.get()

    run_directory = f'results_woc/{run_title}_{filename.rstrip(".tsp")}'    # unique directory for a given run, specified by run_title and filename
    os.makedirs(run_directory, exist_ok=True)



    # number of threads, i.e., crowd members
    crowd_size = int(crowd_size_entry.get())
    
    # instantiate the problem here, so we can pass it to the crowd members
    problem = TSPMap(entry_filename.get())

    G = int(entry_G.get())
    N = int(entry_N.get())
    percent_preserved = float(entry_percent_preserved.get())
    lambda_value = float(entry_lambda_value.get())
    xover_method = entry_xover_method.get()
    lower = float(entry_lower.get())
    upper = float(entry_upper.get())
    mutation_rate = float(entry_mutation_rate.get())
    abominate_threshold_value = int(entry_abominate_threshold_value.get())
    abomination_percentage_value = float(entry_abominate_percentage.get())
    use_tours = entry_use_tours.get().lower()

    num_runs = int(entry_runs.get())
    verbose = int(verbose_entry.get())

    # lambda noise parameter
    lambda_noise = float(lambda_noise_entry.get())

    # show the plot at the end of each run? This can stall the total experiment if the plot is not closed
    show_plots = int(show_plots_entry.get())


    for run_number in range(num_runs):
        #############################################
        # CROWD RESULTS PER RUN STORED HERE 
        #############################################
        crowd_df_data = []
        crowd_path_list = []
        crowd_edge_counts = {}
        #############################################

        # list of threads, i.e., the individual members of our crowd
        workers = []

        #################################
        # Get all the parameters here...
        # These are from the same field
        #################################
        print(f"WoC run number {run_number} of {num_runs} WoC runs...")

        # start the threads! Each thread is a member in our crowd!
        _ = time.perf_counter()
        for i in range(crowd_size):
            
            noisy_lambda = lambda_value + np.random.uniform(-lambda_noise, lambda_noise, 1)[0]
            noisy_lambda = max(noisy_lambda, 1)

            print(f"Starting crowd member {i} with lambda = {noisy_lambda}...")
            worker = threading.Thread(target=_run_crowd_member, # our worker function
                                    args=(    # FROM ENTRY FIELDS:
                                                crowd_df_data,      # crowd datas to be stored as CSV
                                                crowd_path_list,    # list of PathDistance objects obtained from each thread
                                                i,                  # thread id         
                                                problem,            # TSPMap object
                                                G,                  # generations
                                                N,                  # population size
                                                percent_preserved,  # proportion of population preserved
                                                noisy_lambda,       # lambda parameter; NOISE CAN BE ADDED TO THIS PARAMETER
                                                xover_method,       # crossover method
                                                lower,              # split lower
                                                upper,              # split upper
                                                mutation_rate,      # mutation rate
                                                abominate_threshold_value,      # abominate threshold     
                                                abomination_percentage_value,   # abomination percentage
                                                use_tours,                  # determines if we work with cyclical paths or not
                                                verbose,                    # verbose, to control the number of prints to the console
                                            ) # end args for worker thread
                                        )
            workers.append(worker)
            worker.start()

        for worker in workers:
            print(f"Joining worker {worker}...")
            worker.join()
        # End the crowd work here! Next is aggregation...
        # the crowd_edge_counts dict is later used for conversion to CSV for user-friendly manipulation
        crowd_edge_counts = problem.count_edges(
                                                crowd_edge_counts, # pass the empty dictionary
                                                crowd_path_list    # pass the list of PathDistance objects
                                                )
        # AGGREGATION FUNCTION CALLED HERE
        aggregated_solution, resorted_to_greedy_insertion = problem._majority_vote_aggregation(crowd_edge_counts)

        # Capture the runtime after we finished our aggregation.
        runtime = time.perf_counter() - _

        if use_tours == 'false':
            # aggregation function returns a tour; if we don't want a tour, trim it and make it open.
            aggregated_solution = problem.trim_path(aggregated_solution, 1)

        # print path solution, for testing and verifying...
        print(aggregated_solution)
        print(f"path length: {aggregated_solution.length}")

        
        # Save the crowd data to a CSV file. This is for the individual solutions of crowd members
        df = pd.DataFrame(crowd_df_data, columns=['crowd_member', 'local_best_path', 'distance', 'ga_elapsed_t'])
        os.makedirs(f'{run_directory}/run_{run_number}/csv/', exist_ok=True)
        df.to_csv(f'{run_directory}/run_{run_number}/csv/crowd_results.csv', index=False)


        #############################################
        # May seem redundant, but I'll keep a separate CSV for the WoC runtime...
        #######################
        # The run's runtime to be saved to a CSV file
        runtime_data = {'woc_run_number': run_number, 'runtime': runtime}
        runtime_df = pd.DataFrame(runtime_data, index=[0])

        # Save the runtime data to a CSV file
        if os.path.exists(f'{run_directory}/woc_runtime.csv'):
            runtime_df.to_csv(f'{run_directory}/woc_runtime.csv', mode='a', header=False, index=False)
        else:
            runtime_df.to_csv(f'{run_directory}/woc_runtime.csv', index=False)
        #############################################



        # GA best and WoC best per run, to get AVERAGE data on solutions
        # Sort the DataFrame by the 'distance' column
        df_sorted = df.sort_values(by='distance')

        # Pull the row with the lowest value of 'local_best_path'
        lowest_local_best_path_row = df_sorted.iloc[0]
        print(f"ADDING ROW: {lowest_local_best_path_row}")
        # Convert the row to a DataFrame and add the run_number column
        lowest_local_best_path_df = pd.DataFrame([lowest_local_best_path_row])




        lowest_local_best_path_df['run_number'] = run_number

        ######################
        # Append the GA+WoC aggregated solution and its distance to the DataFrame
        ######################
        # Add the GA+WoC aggregated solution and its distance to the DataFrame! There's no best-of-the-best for WoC, since all solutions aggregate into one.
        lowest_local_best_path_df['ga+woc_aggregated_solution'] = aggregated_solution.current_path
        lowest_local_best_path_df['ga+woc_solution_distance'] = aggregated_solution.current_distance
        lowest_local_best_path_df['ga+woc_runtime'] = runtime
        lowest_local_best_path_df['resorted_to_greedy_insertion'] = resorted_to_greedy_insertion

        # Rename the 'local_best_path' column to 'ga_best_of_best'
        # This is the BEST path from a crowd of individual GAs, and this is NOT an aggregated results (unlike WoC)
        lowest_local_best_path_df = lowest_local_best_path_df.rename(columns={'local_best_path': 'ga_best_of_best'})
        lowest_local_best_path_df = lowest_local_best_path_df.rename(columns={'distance': 'ga_best_distance'})


        # Save the runtime data to a CSV file
        if os.path.exists(f'{run_directory}/ga_and_ga+woc_solution_data.csv'):
            lowest_local_best_path_df.to_csv(f'{run_directory}/ga_and_ga+woc_solution_data.csv', mode='a', header=False, index=False)
        else:
            lowest_local_best_path_df.to_csv(f'{run_directory}/ga_and_ga+woc_solution_data.csv', index=False)  


        # Save the edge counts to a CSV file, so we first convert the dictionary to a DataFrame
        # Convert the dictionary to a DataFrame
        edge_counts_df = pd.DataFrame(list(crowd_edge_counts.items()), columns=['edge', 'count'])
        edge_counts_df.to_csv(f'{run_directory}/run_{run_number}/csv/crowd_edge_counts.csv')

        # What if we superimpose the crowd solutions? This reveals the weighted edges of combined solutions.
        # This helps us to visualize stronger edges among the crowd solutions
        problem.plot_path_heatmap(crowd_path_list, crowd_size)
        plt.savefig(f'{run_directory}/run_{run_number}/{run_title}_crowd_size_{crowd_size}_heatmap.png')
        if show_plots == 1:  
            plt.show()
        else:
            plt.close()
    # if we need anything else to happen after the run, put it here.
    return
    # end run_number loop



def ga_mainloop():
    '''
    This function is the main loop for the Genetic Algorithm (GA) and GA + Wisdom of the Crowds (GA+WoC) GUI.
    Parameters are self-explanatory. More detailed parameter definitions found in Project 4 report.
    Project 5 addition includes the GA+WoC functionality.
    Each individual is a separate thread running the GA.
    Individual solutions aggregated based on edge counts and a 'majority vote'.
    Average performance (average path distance, runtime) is computed for GA and GA+WoC across runs.
    GA+WoC generates heatmap showing edge counts.
    '''

    # Create the main window
    root = tk.Tk()
    root.title("Genetic Algorithm Parameters")

    # Labels and entry fields for all of our parameters
    tk.Label(root, text="Generations").grid(row=0, column=0)
    global entry_G
    entry_G = tk.Entry(root)
    entry_G.grid(row=0, column=1)
    entry_G.insert(0, "1000")

    tk.Label(root, text="Population Size").grid(row=1, column=0)
    global entry_N
    entry_N = tk.Entry(root)
    entry_N.grid(row=1, column=1)
    entry_N.insert(0, "100")

    tk.Label(root, text="Percent Preserved").grid(row=2, column=0)
    global entry_percent_preserved
    entry_percent_preserved = tk.Entry(root)
    entry_percent_preserved.grid(row=2, column=1)
    entry_percent_preserved.insert(0, "0.5")

    tk.Label(root, text="Lambda Value").grid(row=3, column=0)
    global entry_lambda_value
    entry_lambda_value = tk.Entry(root)
    entry_lambda_value.grid(row=3, column=1)
    entry_lambda_value.insert(0, "1")

    tk.Label(root, text="Xover Method").grid(row=4, column=0)
    global entry_xover_method
    entry_xover_method = tk.Entry(root)
    entry_xover_method.grid(row=4, column=1)
    entry_xover_method.insert(0, "chunk_and_sweep")

    tk.Label(root, text="Split Bound Lower").grid(row=5, column=0)
    global entry_lower
    entry_lower = tk.Entry(root)
    entry_lower.grid(row=5, column=1)
    entry_lower.insert(0, "0.4")

    tk.Label(root, text="Split Bound Upper").grid(row=6, column=0)
    global entry_upper 
    entry_upper = tk.Entry(root)
    entry_upper.grid(row=6, column=1)
    entry_upper.insert(0, "0.6")

    tk.Label(root, text="Mutation Rate").grid(row=7, column=0)
    global entry_mutation_rate
    entry_mutation_rate = tk.Entry(root)
    entry_mutation_rate.grid(row=7, column=1)
    entry_mutation_rate.insert(0, "0.05")

    tk.Label(root, text="Abominate Threshold Value").grid(row=8, column=0)
    global entry_abominate_threshold_value
    entry_abominate_threshold_value = tk.Entry(root)
    entry_abominate_threshold_value.grid(row=8, column=1)
    entry_abominate_threshold_value.insert(0, "9999")

    tk.Label(root, text="Abomination Percentage").grid(row=9, column=0)
    global entry_abominate_percentage
    entry_abominate_percentage = tk.Entry(root)
    entry_abominate_percentage.grid(row=9, column=1)
    entry_abominate_percentage.insert(0, "0.5")

    tk.Label(root, text="Runs (also applies to WoC)").grid(row=10, column=0)
    global entry_runs
    entry_runs = tk.Entry(root)
    entry_runs.grid(row=10, column=1)
    entry_runs.insert(0, "1")

    tk.Label(root, text="Filename").grid(row=11, column=0)
    global entry_filename
    entry_filename = tk.Entry(root)
    entry_filename.grid(row=11, column=1)
    entry_filename.insert(0, "Random100.tsp")  # Default value

    tk.Label(root, text="Run Title").grid(row=12, column=0)
    global entry_run_title
    entry_run_title = tk.Entry(root)
    entry_run_title.grid(row=12, column=1)
    entry_run_title.insert(0, "my_run")  # Default value

    # Main execute button for running the genetic algorithm
    execute_button = tk.Button(root, text="Run Genetic Algorithm", command=_run_genetic_algorithm)
    execute_button.grid(row=14, column=0, columnspan=2)


    tk.Label(root, text="Run Directory").grid(row=15, column=0)
    global run_directory_entry
    run_directory_entry = tk.Entry(root)
    run_directory_entry.grid(row=15, column=1)
    run_directory_entry.insert(0, "my_run")  # Default value

    # use tours entry
    tk.Label(root, text="Use Tours").grid(row=16, column=0)
    global entry_use_tours
    entry_use_tours = tk.Entry(root)
    entry_use_tours.grid(row=16, column=1)
    entry_use_tours.insert(0, "true")  # Default value


    # Main execute button for running the genetic algorithm
    execute_button = tk.Button(root, text="Re-plot Final Graph (for GA, not WoC)", command=_plot_final)
    execute_button.grid(row=18, column=0, columnspan=2)

    # show plots
    tk.Label(root, text="Show WoC heatmap plot per run (1 for yes, 0 for no)").grid(row=19, column=0)
    global show_plots_entry
    show_plots_entry = tk.Entry(root)
    show_plots_entry.grid(row=19, column=1)
    show_plots_entry.insert(0, "0")  # Default value


    plus_minus = "\u00B1"
    # Define lambda noise p
    tk.Label(root, text=f"Lambda noise (randomizes crowd lambda {plus_minus} noise)").grid(row=27, column=0)
    global lambda_noise_entry
    lambda_noise_entry = tk.Entry(root)
    lambda_noise_entry.grid(row=27, column=1)
    lambda_noise_entry.insert(0, "0.2")  # Default value


    # Define Verbose for print outputs
    tk.Label(root, text="Verbose (0 to 4, for more/less prints").grid(row=28, column=0)
    global verbose_entry
    verbose_entry = tk.Entry(root)
    verbose_entry.grid(row=28, column=1)
    verbose_entry.insert(0, "3")  # Default value

    # Define crowd size
    tk.Label(root, text="Crowd size (number of threads)").grid(row=29, column=0)
    global crowd_size_entry
    crowd_size_entry = tk.Entry(root)
    crowd_size_entry.grid(row=29, column=1)
    crowd_size_entry.insert(0, "20")  # Default value

    # Main execute button for running the genetic algorithm
    execute_button = tk.Button(root, text="GA Wisdom of the Crowds (GA-WoC)", command=woc_start_threads)
    execute_button.grid(row=30, column=0, columnspan=2)


    # Run the Tkinter event loop
    root.mainloop()
# end ga_mainloop()
# ====================================================================
# END ----- TKINTER FUNCTION FOR GENETIC ALGORITHM AND WISDOM OF THE CROWDS
# ====================================================================
