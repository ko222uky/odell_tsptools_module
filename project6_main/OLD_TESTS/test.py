from odell_tools import odell_hitting_strings as ohs
import matplotlib.pyplot as plt
import warnings
import copy

# ================================================ #
#               MAIN FUNCTION
# ================================================ #

def main():
    
    #problem = ohs.HittingStringProblem(name = "HSP30_30") # The default problem has 30 strings of length 30.

    #for string in problem.string_list:
    #    print(string)

    #print("Wildcard coverage: ", problem.wildcard_coverage_string)
    #print("Wildcard coverage percentage: ", problem.wildcard_coverage_percent)


    # Now, let's constrain the coverage!
    #print("\n\nConstraining coverage to 10%")
    #problem.reduce_wildcard_coverage(20)
    #for string in problem.string_list:
    #   print(string)
    #print("Wildcard coverage: ", problem.wildcard_coverage_string)
    #print("Wildcard coverage percentage: ", problem.wildcard_coverage_percent)


    # Now, let's test the SolutionString object.
    #cstring = ohs.SolutionString(length = problem.string_length, random_init = True)

    #print("\n\nRandomly generated solution string:")
    #print(cstring)

    #cstring.evaluate_fitness(problem)
    #print("Fitness: ", cstring.fitness)

    ################################
    # Closest String Problem
    ################################
    
    csp = ohs.ClosestStringProblem(initialize_solution_population = True)

    # Evaluate and update for generation 0
    csp.evaluate_population_fitness(0)
    csp.update_population_heatmaps(0)

    ############################
    # We're keeping track of the best solution by minimizing MAX distance, and the best solution by minimizing average .
    ############################

    # at generation 0, best solution is the first one we see. 
    # Get the initial data for the best solution.
    # The default comparators are by max distance, but in this method, we will specify to select by average distance.
    best_by_average = csp.select_best_solution(by_average = True)
    best_by_average.append_heatmap_data("best_by_average")


    
    # Now, let's run the genetic algorithm for 10000 generations.s
    for i in range(10):

        parents = copy.deepcopy(csp.select_parents())             # Parents selected after evaluation of fitness 

        csp.reproduce(parents, i)                                 # Parents reproduce up to population size. Handles mutation, too. Mutation targets bottom 10% match-sum locations, via point mutations.

        csp.evaluate_population_fitness(i)                        # After reproduction and mutation of offspring, evaluate fitness.


        this_gen_best_by_average = csp.select_best_solution(by_average = True)
        if this_gen_best_by_average < best_by_average:
            best_by_average = this_gen_best_by_average
        
        if (i % 10 == 0):
            print(f"Generation {i}, {best_by_average.string}: {best_by_average.fitness}")

  
        # Next, record the heatmap data for the best solutions...
        if (i % 1 == 0):   
                    
            best_by_average.append_heatmap_data("best_by_average")
            best_by_average.plot_hamming_distribution(i, lower_x = 50, upper_x = 150)
            plt.savefig(f"vanilla_experiment/run_0/best_by_average_hamming_dist_gen_{i}.png")
            plt.close()

        best_by_average.append_fitness_data(i, "best_by_average", run_number = 0)

    # Now, let's test selecting the best solution from a population.


# end main



if __name__ == "__main__":
    main()