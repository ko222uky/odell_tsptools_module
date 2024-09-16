# ================================================ #
#               IMPORTED MODULES
# ================================================ #

import math # to calculate square roots when implementing distance formula
import glob # to iterate through my directory via wildcard * (e.g., *.tsp) and read all .tsp files
import time # for the .time() method so that we can time how long the TSP algorithms take
import copy # to create deep copies when defining recursive methods
import matplotlib.pyplot as plt # for plotting our lists of tuples, i.e., creating a 'geographical map'
import queue # for implementing FIFO and priority queue for various search algorithms

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
    def current_vertex(self):
        return self.__current_vertex

    @property
    def current_distance(self):
        return self.__current_distance

    @property
    def edges(self):
        if self.__edges is None:
            # Now we can compute it, since we need it.
            self.__edges = self._calculate_edge()
        return self.__edges

    def _calculate_edge(self):
        vertex_labels = self.__current_path.split('-') # get a list of our vertex labels
        edge_dict = {} # edges will be stored as dictionary. Key will be in the form of '1-2' for vertices 1 and 2
        for i in range(len(vertex_labels) - 1):
            j = i + 1 # i and j to iterate through subsequent pairs of vertex labels
            if j > len(vertex_labels):
                break
            # the key is the starting vertex, value is the ending vertex
            edge_dict[vertex_labels[i]] = vertex_labels[j]
        return edge_dict
        
    def _edge_insert(self, edge, vertex):
        '''
        Inserts an edge between two vertices in the current path.
        NOTE: This method does not update the current distance nor path string.
        Not intended for direct use by the user.
        Requires TSPMap object to rebuild path.
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

    def __getitem__(self, _):
        return self

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
        x_y_list = []
        for vertex in self.__nodes.values():
            # iterate through vertex values in our instance's dictionary and append (x, y) tuples
            x_y_list.append(vertex.coords)
        return x_y_list

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

    ######################
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

    def closest_edge_insertion(self, initial_vertices, plot_steps=False, line_segment=False):
        
        '''
        Greedy algorithm for the TSP problem.
        The closest edge insertion is a heuristic used in a greedy algorithm that builds a solution by iteratively adding the vertex that minimizes the total distance.
        The vertex is added to the edge that is closest to the current path.
        Distance to edge is approximated as a point projected to a line (whether infinite or finite)
        The algorithm starts with a circular path from the initial vertices, usually 3 vertices.
        
        '''

        solution_steps = [] # list of PathDistance objects representing the solution as it builds
        step = 0 # counter for the number of steps taken    
        unvisited_nodes = copy.deepcopy(self.__nodes)


        # Construct circular path from initial vertices.
        # Assumed to be triangular (3 vertices).
        initial_tour = PathDistance(self[str(initial_vertices[0])].label, self[str(initial_vertices[0])], 0.0)
        del unvisited_nodes[str(initial_vertices[0])]

        for vertex_label in initial_vertices[1:]:

            initial_tour += self.__nodes[str(vertex_label)]

            # remove the node from our unvisited nodes
            del unvisited_nodes[str(vertex_label)]

        initial_tour += self[str(initial_vertices[0])] # close the loop
        # print(initial_tour) # for testing
        # print(unvisited_nodes.keys()) # for testing

        # add the initial tour to our list of solution steps
        solution_steps.append(initial_tour)
        if plot_steps:
            self.plot_path(initial_tour, "Step " + str(step), save=True)

        next_tour = copy.deepcopy(initial_tour)

        # print(step, next_tour) # for testing
        
        return self._recursive_closest_edge_insertion(next_tour,     # initial tour
                                                      unvisited_nodes,  # unvisited nodes
                                                      step,             # counter for the number of steps taken, for naming the plots
                                                      solution_steps,   # list of PathDistance objects representing the solution as it builds
                                                      plot_steps,       # specify whether we need to plot the steps, individually
                                                      line_segment      # specify whether we need to calc distance to a finite line (=True) or infinite line (=False)
                                                      )
        # Now we can start the greedy edge insertion recursive call...
        # Pass the initial tour and the unvisited nodes, and the solution steps list

        #return self._recursive_closest_edge_insertion(self, initial_tour, unvisited_nodes, step, solution_steps)

    def _recursive_closest_edge_insertion(self, tour, unvisited_nodes, step, solution_steps, plot_steps=False, line_segment=False):
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

        unvisited_nodes_keys = list(unvisited_nodes.keys())                    # keys of unvisited nodes
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
            self.plot_path(min_tour, "Step " + str(step), save=True)
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
                                                      line_segment      # specify whether we need to calc distance to a finite line (=True) or infinite line (=False)
                                                      )
    # end of recursive_closest_edge_insertion() method  

    def _find_closest_edge(self, tour, vertex, line_segment=False):
        '''
        Finds the closest edge to a given vertex in a given PathDistance object.
        Returns the edge in format '1-2' for vertices 1 and 2.
        '''
        # Actually, we want to find the closest edge to a vertex in a given PathDistance object.
        minimum_distance = float('inf') # set to infinity, a more pythonic way to do this
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


    def _point_to_edge_distance(self, vertex, edge_points, line_segment=False):
        '''
        Calculates the distance from a point to an edge.
        Returns the distance.
        NOTE: Caveat is whether we mean an infinite line or a line segment.
        NOTE: Uses of 'infinite line' assumptions for an edge may give odd results.
        NOTE: These odd results in TSP show up as cross-overs.
        '''
        # point is a tuple of (x, y) coordinates
        # edge_points is a tuple of two tuples of (x, y) coordinates
        # we calculate the distance from the point to the line segment defined by the edge points
        # we return the distance; source: https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html

        # unpack the edge points
        x1, y1 = edge_points[0]
        x2, y2 = edge_points[1]

        # unpack the point
        x0, y0 = vertex.coords

        diff_x = x2 - x1
        diff_y = y2 - y1

        # distance between the two edge points
        numerator = abs(diff_x * (y1 - y0) - (x1 - x0) * diff_y)
        edge_length = math.sqrt(diff_x**2 + diff_y**2)

        if not line_segment:
            # The edge line is treated as infinite (i.e., not a line segment)

            return numerator / edge_length
        else:

            # edge line treated as line segment
            # More approximate measures may be needed.

            # The edge is treated as a line segment
            # Compute the projection scalar, that is the scalar that projects the point onto the line segment
            #
            projection_scalar = ((x0 - x1) * diff_x + (y0 - y1) * diff_y) / (edge_length**2)

            if projection_scalar < 0:
                # The projection point is beyond the first edge point,
                # so we compute the distance from the point to the first edge point
                # NOTE: This is a more approximate measure, and differs a lot from an infinite line
                distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

                return distance

            #######################
            #   Test if the point is closer to one of the end points
            #######################
            elif projection_scalar > 1:
                # The projection point is beyond the second edge point,
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



    def calculate_edges(self, tour):
        '''
        Calculates the edges of a given PathDistance object.
        Returns a dictionary of edges, where the key is in format '1-2' for vertices 1 and 2.
        The values are pairs of vertex coordinates.
        Mainly used to allow ease of access to edges, reconstructed from the path string.
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



    def _edge_insert(self, tour, edge, vertex):
        '''
        Inserts vertex into edge of a given PathDistance object.
        Reconstructs object and returns new object with updated path and distance.
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







    # object method to plot a particular path over the problem map
    def plot_path(self, path: PathDistance, plot_title: str, save=False): 
        plot = _plot_coords(self.map_coords,       # TSPMap coords
                    self.str_to_coords(path),           # path coords
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


    # object method to plot only the map


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
#               Inherited Classe                 
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
                path_color='blue'
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
        plt.savefig(title.removesuffix(suffix) + '_map.png', format='png')

    return plt.gca()
# end _plot_coords() definition
