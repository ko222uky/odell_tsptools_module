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
import warnings
import tkinter as tk        # for our simple GUI

import threading
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


def run_genetic_algorithm():
    #############################################
    # MAIN TEST HERE
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

def plot_final():

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


def run_crowd_member(
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

    
# end worker function


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

    #############################################
    # CROWD RESULTS STORED HERE 
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

    # number of threads, i.e., crowd members
    crowd_size = int(crowd_size_entry.get())
    
    # instantiate the problem here, so we can pass it to the crowd members
    problem = tsp.TSPMap(entry_filename.get())

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

        print(f"WoC run number {run_number} of {num_runs} WoC runs...")

        # start the threads! Each thread is a member in our crowd!
        _ = time.perf_counter()
        for i in range(crowd_size):
            
            noisy_lambda = lambda_value + np.random.uniform(-lambda_noise, lambda_noise, 1)[0]
            noisy_lambda = max(noisy_lambda, 1)

            print(f"Starting crowd member {i} with lambda = {noisy_lambda}...")
            worker = threading.Thread(target=run_crowd_member, # our worker function
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
        # End the crowd work here! Take the runtime recording in the next line
        runtime = time.perf_counter() - _


        # Save the crowd data to a CSV file. This is for the individual solutions of crowd members
        df = pd.DataFrame(crowd_df_data, columns=['crowd_member', 'local_best_path', 'distance', 'elapsed_t'])
        os.makedirs(f'{run_directory}/run_{run_number}/csv/', exist_ok=True)
        df.to_csv(f'{run_directory}/run_{run_number}/csv/crowd_results.csv', index=False)

        # The run's runtime to be saved to a CSV file
        runtime_data = {'woc_run_number': run_number, 'runtime': runtime}
        runtime_df = pd.DataFrame(runtime_data, index=[0])

        # Save the runtime dataq to a CSV file
        if os.path.exists(f'{run_directory}/woc_runtime.csv'):
            runtime_df.to_csv(f'{run_directory}/woc_runtime.csv', mode='a', header=False, index=False)
        else:
            runtime_df.to_csv(f'{run_directory}/woc_runtime.csv', index=False)

        # GA best and WoC best per run, to get AVERAGE data on solutions
        # Sort the DataFrame by the 'local_best_path' column
        df_sorted = df.sort_values(by='local_best_path')

        # Pull the row with the lowest value of 'local_best_path'
        lowest_local_best_path_row = df_sorted.iloc[0]

        # Convert the row to a DataFrame and add the run_number column
        lowest_local_best_path_df = pd.DataFrame([lowest_local_best_path_row])
        lowest_local_best_path_df['run_number'] = run_number

        lowest_local_best_path_df['ga+woc_best'] = 'na'

        # Rename the 'local_best_path' column to 'ga_best_of_best'
        # This is the BEST path from a crowd of individual GAs, and this is NOT an aggregated results (unlike WoC)
        lowest_local_best_path_df = lowest_local_best_path_df.rename(columns={'local_best_path': 'ga_best_of_best'})
        lowest_local_best_path_df = lowest_local_best_path_df.rename(columns={'distance': 'ga_best_distance'})


        # Save the runtime dataq to a CSV file
        if os.path.exists(f'{run_directory}/ga_and_ga+woc_solution_data.csv'):
            lowest_local_best_path_df.to_csv(f'{run_directory}/ga_and_ga+woc_solution_data.csv', mode='a', header=False, index=False)
        else:
            lowest_local_best_path_df.to_csv(f'{run_directory}/ga_and_ga+woc_solution_data.csv', index=False)  


        # Save the edge counts to a CSV file
        crowd_edge_counts = problem.count_edges(
                                                crowd_edge_counts, # pass the empty dictionary
                                                crowd_path_list    # pass the list of PathDistance objects
                                                )
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

    return

    # end run_number loop
    
    # if we need anything else to happen after the run, put it here.


def new_func():
    '''
    '''


def main():
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
    execute_button = tk.Button(root, text="Run Genetic Algorithm", command=run_genetic_algorithm)
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
    execute_button = tk.Button(root, text="Re-plot Final Graph (for GA, not WoC)", command=plot_final)
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

# end main





# end main()

if __name__ == "__main__":
    main()
