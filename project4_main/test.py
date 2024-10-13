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
    G = 10 
    N = 100
    percent_preserved = 0.5
    lambda_value = 2.5 
    xover_method = 'chunk_and_sweep'
    lower = 0.1
    upper = 0.9
    mutation_rate = 0.5
    abominate_threshold_value = 1
    runs = 22       # how many times to run the GA?
    ###########################################################################

    # save the parameters for this run as .csv
    with open(f'{run_directory}/csv/parameters_{filename}.csv', 'w') as file:
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

    with open(f'{run_directory}/csv/run_series_best_worst_and_runtime.csv', 'w') as file:
        file.write('run_number,best_path,best_distance,worst_path,worst_distance,run_time\n')

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
            file.write(f'{latest_run},{best.current_path},{best.current_distance},{worst.current_path},{worst.current_distance},{elapsed_t}\n')
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

    ################################################
    # Process final data for GA run series fitness trend
    ################################################
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








# end main()

if __name__ == "__main__":
    main()
