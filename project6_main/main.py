import odell_hitting_strings as ohs
import matplotlib.pyplot as plt
import warnings

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
    
    csp = ohs.ClosestStringProblem(name = "CSP30_30", initialize_solution_population = True)

    for string in csp.string_list:
        print(string)

    # Evaluate and update for generation 0
    csp.evaluate_population_fitness(0)
    csp.update_population_heatmaps(0)

    # at generation 0, best solution is the first one we see 
    best_solution = csp.select_best_solution()

    # Get the initial data for the best solution
    best_solution.append_heatmap_data("best_over_time_heatmap")

    # Now, let's run the genetic algorithm for 10000 generations.s
    for i in range(100000):
        print(f"Generation: {i}, best solution fitness is {best_solution.fitness}")

        csp.reset_population()                      # akin to reproduction
        csp.evaluate_population_fitness(i)          # akin to evaluation

        possible_best = csp.select_best_solution()  # akin to selection of parent subpopulation
        if best_solution > possible_best:
            best_solution = possible_best

                                                    # Next, record data for the best solution
        if (i % 1000 == 0):   
            best_solution.update_heatmap(csp, i)                        
            best_solution.append_heatmap_data("best_over_time_heatmap")

            # Print unique Hamming distances and their counts
            for hamming_distance, count in best_solution._hamming_counts.items():
                print(f"Hamming Distance BEST SOLUTION: {hamming_distance}, Count: {count}")
            best_solution.plot_smoothed_hamming_distribution(i)
            plt.show()

        best_solution.append_fitness_data(i, "best_over_time_fitness")
   
    # Now, let's test selecting the best solution from a population.


# end main



if __name__ == "__main__":
    main()