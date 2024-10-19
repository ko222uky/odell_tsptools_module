# ====================================================================
#   IMPORT MODULE(S)
# ====================================================================
# import the module that I wrote for this project

import odell_tsptools as tsp    # My custom module
import time                     # for timing individual function calls when timing was not implemented in my module
import pandas as pd             # for reading runtime data which I wrote to CSV file
import numpy as np
import scipy.stats as stats     # for the Kruskal-Wallis test
import scikit_posthocs as sp    # for the post-hodc tests (Dunn's with Bonferoroni correction)
                                # Needs to be installed using pip:
                                # $ pip install scikit_posthocs
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# ====================================================================
#   conda environment created with name 'cse545'
#   environment used conda install to install following:
#
#   Python 3.12.4
#   matplotlib 3.8.4 
#
#
# ====================================================================
#                             MAIN FUNCTION
# ====================================================================
#
#   main() function intended to be executed directly as script, not as an imported module


def main():


    ###################################
    #   Initialize our Problem   
    ###################################

    ##############################################
    # Testing PathDistance methods separately
    ##############################################

    problem = tsp.TSPMap("Random10.tsp")
     
    # Testing PathDistance functionality
    # Building path manually with hard-coded Vertex objects
    v1 = tsp.Vertex(1, 1)
    v2 = tsp.Vertex(2, 2)
    v3 = tsp.Vertex(3, 3)
    v4 = tsp.Vertex(4, 4)
    v5 = tsp.Vertex(5, 5)
    v6 = tsp.Vertex(6, 6)
    v7 = tsp.Vertex(7, 7)
    v8 = tsp.Vertex(8, 8)
    v9 = tsp.Vertex(9, 9)
    v10 = tsp.Vertex(10, 10)

    # Instantiate PathDistance object via chaining
    PathV10 = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10

    # Testing the length property for my PathDistance object
    print(PathV10.length)

    # This iterates through the vertex labels of the first half of the path
    split_point = 3
    for i in range(PathV10.length // 3):
        print(PathV10[i]) 
    print("second half")
    for i in range(PathV10.length // 3, PathV10.length):
        print(PathV10[i])

    # Testing the build_path() TSPMap object method
    
    vertex_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    built_path = problem.build_path(vertex_labels)
    built_path2 = problem.build_path('1-2-3')
    print(built_path)

    print('12-6-3-1-2-3-4')
    print(built_path2)
    # Testing the TSPMap object's split_path() method
    first, second = built_path.split()
    print(first)
    print(second)


    first, second = built_path.split(0.88888888)
    print(first)
    print(second)
    
    # Test the generation of random paths
    population = problem.generate_random_paths(10)
    for path in population:
        print(path)

    # Test the ranking of populations
    ranked_population = problem.rank_population(population)

    for rank, path in ranked_population.items():
        print(f"rank: {rank}, path: {path}")


    ranked_sub = problem.select_subpopulation(ranked_population, 5)

    for rank, path in ranked_sub.items():

        print(f"rank: {rank}, path: {path}")

    # test cross-over method chunk-and-sweep
    first_parent = ranked_sub[1]
    second_parent = ranked_sub[2]
    print("CHUNK AND SWEEP")
    print("Here are the parents: ")
    print(first_parent)
    print(second_parent)

    # Generate offspring with the chunk_and_sweep crossover method
    offspring = problem.chunk_and_sweep(first_parent, second_parent, 0.5)
    
    print("Here are the offspring")
    
    for o in offspring:
        print(o)
    
    # We have four offspring returned. Let's select the best one:
    print("Select the best offspring")
    best_offspring = sorted(offspring)[0]
    print(best_offspring)
    
    # Now, we can mutate the best offspring. We call the simple_mutate() function to swap two vertices
    mutated_best_offspring = problem.simple_mutate(best_offspring, 1)
    print("Here is the mutated offspring:")
    print(mutated_best_offspring)


    # Or, we can mutate all of the four offspring and see which one turned out the best!
    print("Mutate all offspring with probability 0.5 and return: ")
    mutated_offspring = [problem.simple_mutate(o, mutation_probability = 0.5) for o in offspring]
    for mo in mutated_offspring:
        print(mo)
    print("The best offspring after mutation: ")
    print(sorted(mutated_offspring)[0])

    print("TESTS FOR PROJECT 5")
    # simple function to trim off the last path
    path = problem.build_path('1-2-3-4-5')
    print(path)
    trimmed_path = problem.trim_path(path, 1)
    print(trimmed_path)
    print(problem.build_path('1-2-3-4'))

    tour = problem.loop_path(path)
    print(tour)

    broken_tour = problem.trim_path(tour, 1)
    print(broken_tour)

    print(broken_tour.edges)

    print(broken_tour.normalized_edges)


    # Testing the edge_counts method
    print("\n\nTesting edge_counts method!")
    # Test the generation of random paths
    population = problem.generate_random_paths(100)

    # initialize our dictionary
    edge_counts = {}

    edge_counts = problem.edges_counts(edge_counts, population)

    print(edge_counts)





# end main()

if __name__ == "__main__":
    main()
