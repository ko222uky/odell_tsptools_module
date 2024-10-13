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

    problem12 = tsp.TSPMap("Random12.tsp")
     
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

    built_path = problem12.build_path(vertex_labels)
    built_path2 = problem12.build_path('12-6-3-1-2-3-4')
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
    population = problem12.generate_random_paths(10)
    for path in population:
        print(path)

    # Test the ranking of populations
    ranked_population = problem12.rank_population(population)

    for rank, path in ranked_population.items():
        print(f"rank: {rank}, path: {path}")


    ranked_sub = problem12.select_subpopulation(ranked_population, 5)

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
    offspring = problem12.chunk_and_sweep(first_parent, second_parent, 0.5)
    
    print("Here are the offspring")
    
    for o in offspring:
        print(o)
    
    # We have four offspring returned. Let's select the best one:
    print("Select the best offspring")
    best_offspring = sorted(offspring)[0]
    print(best_offspring)
    
    # Now, we can mutate the best offspring. We call the simple_mutate() function to swap two vertices
    mutated_best_offspring = problem12.simple_mutate(best_offspring, 1)
    print("Here is the mutated offspring:")
    print(mutated_best_offspring)


    # Or, we can mutate all of the four offspring and see which one turned out the best!
    print("Mutate all offspring with probability 0.5 and return: ")
    mutated_offspring = [problem12.simple_mutate(o, mutation_probability = 0.5) for o in offspring]
    for mo in mutated_offspring:
        print(mo)
    print("The best offspring after mutation: ")
    print(sorted(mutated_offspring)[0])


    #############################################
    # MAIN TEST HERE
    #############################################
    
    # directories for results
    os.makedirs('results', exist_ok=True)

    # filename
    filename = 'Random100.tsp'
    run_title = 'my_run'

    run_directory = f'results/{run_title}_{filename.rstrip(".tsp")}'
    
    # directories for runs
    os.makedirs(run_directory, exist_ok=True)
    os.makedirs(f'{run_directory}/csv', exist_ok=True)    
    os.makedirs(f'{run_directory}/csv/ga', exist_ok=True)    
    os.makedirs(f'{run_directory}/csv/events', exist_ok=True)    
    ##########################################################################
    # Parameters:
    ############
    G = 1 
    N = 10
    percent_preserved = 0.2
    lambda_value = 10 
    xover_method = 'chunk_and_sweep'
    lower = 0.1
    upper = 0.9
    mutation_rate = 0.5
    abominate_threshold_value = 1
    runs = 22       # how many times to run the GA?
    ###########################################################################

    # save the parameters for this run as .csv
    with open(f'{run_directory}/parameters_{filename}.csv', 'w') as file:
        file.write("G,N,percent_preserved,lambda_value,xover_method,lower,upper,mutation_rate,abomination_threshold_value,runs\n")
        file.write(f"{G},{N},{percent_preserved},{lambda_value},{xover_method},{lower},{upper},{mutation_rate},{abominate_threshold_value},{runs}")
    
    # create our TSPMap object
    problem = tsp.TSPMap(filename)

    ####################
    # Best paths of run series
    ####################

    best_path = tsp.PathDistance('', distance = float('inf'))

    worst_path = tsp.PathDistance('', distance = float('-inf'))
    
    # BEGIN GA RUN SERIES
    
    latest_run = 0

    with open(f'{run_directory}/run_series_best_worst_and_runtime.csv', 'w') as file:
        file.write('run_number,best_path,best_distance,worst_path,worst_distance,run_time')

        run_start = time.perf_counter()
        for run in range(runs):
            latest_run = run # to reference specific .csv to plot. We plot the late run's individual data for demonstration purposes
            
            _ = time.perf_counter()    
            best, worst = problem.genetic_algorithm(generations = G,
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
                                run_number = latest_run
                                )
            elapsed_t = time.perf_counter() - _
            
            # record data on best and worst per run
            file.write(f'{_},{best.current_path},{best.current_distance},{worst.current_path},{worst.current_distance},{elapsed_t}')
            
            # Same the best and worst path of this run
            if best < best_path:
                best_path = best
            if worst > worst_path:
                worst_path = worst

            print(f"Single Runtime: {elapsed_t} seconds")
        run_elapsed = run_start - time.perf_counter()
        print(f"Total runtime for the series of runs: {run_elapsed}")

        print("Plotting the best and worst path from the run series...")

        problem.plot_path(best, 'Best: Distance = ' + str(best.current_distance) + '\nN = ' + str(problem.dimension), save=False)
        plt.savefig(f'{run_directory}/best_{str(best.current_distance)}.png')
        plt.show()

        problem.plot_path(worst, 'Worst: Distance = ' + str(worst.current_distance) + '\nN = ' + str(problem.dimension), save=False)
        plt.savefig(f'{run_directory}/worst_{str(best.current_distance)}.png')
        plt.show()

    ##################################################################
    # PLOT THE MAIN DATA HERE
    ##################################################################
    # read in the .csv that our GA wrote
    complete_rank_offspring_df = pd.read_csv(f'{run_directory}/csv/events/run_{latest_run}_parent_rank_offspring_number_df.csv', index_col=0)


    # Replace infinite values with NaN in the DataFrame before plotting
    complete_rank_offspring_df = complete_rank_offspring_df.replace([np.inf, -np.inf], np.nan)


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
    plt.title(f'Distribution of Reproductive Events by Parent Ranks\nLambda = {lambda_value}' )
    plt.savefig(f'{run_directory}/reproduction_distribution_figure')
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
    plt.savefig(f'{run_directory}/population_fitness_trend_figure')
    plt.show()




# end main()

if __name__ == "__main__":
    main()
