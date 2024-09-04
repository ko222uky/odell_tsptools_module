# ====================================================================
#   IMPORT MODULE(S)
# ====================================================================
# import the module that I wrote for this project

import odell_tsptools as tsp

# ====================================================================
#   conda environment created with name 'cse545'
#   environment used conda install to install following:
#
#   Python 3.12.4
#   matplotlib 3.8.4 
#
#
# Use class method to perform brute force algorithm on all .tsp files in relative directory
# Parameter uses Unix pattern matching to find appropriate .tsp files
# brute_force_all() returns the results in the form of three dictionaries, packed into a 3-tuple:
#
#   1) dictionary with list of all possible path permutations (specific to brute_force_all() )
#   2) dictionary with list of the minimum path solutions
#   3) dictionary of tuples containing (dimension: int, runtime: float), keyed to the *.tsp filenames
#
# I added keywords to allow the option of writing the results to .txt files.
# The text files containing the results are as follows:
#
#   1) brute_force_all_paths.txt => contains all path permutations, headed by .tsp file name
#   2) brute_force_minimum_paths.txt => contains the minimum path for each .tsp problem file. Implies corresponding symmetric path.
#   3) brute_force_dimension_vs_time.txt => contains runtime data
#
#
# The three returned dictionary results from brute_force_all() can be passed to my driver function plot_results()
# The plot functions use matplotlib.pyplot
# The plot_results() function will print out the minimum path solutions, if print_min_paths=True.
# Also, the plot functions can save the figures to .png files
# The current available figures are:
#
#   1) A figure showing the minimum path overlayed on the TSP problem map
#   2) A figure plotting dimension (x axis) against time (y-axis)
#
# Runtimes were computed using perf_counter() in the time module
#
#
# ====================================================================
#                             MAIN FUNCTION
# ====================================================================
#
#   main() function intended to be executed directly as script, not as an imported module
#

def main():
    
    tsp.plot_results(                                       # unpack and pass brute_force_all() return values to plot_results()
                       *tsp.TSPMap.brute_force_all("*.tsp", # brute force performed on all .tsp files in relative directory
                               write_all_paths=False,       # Do not save all path permutations to .txt (too large for N = 10!)
                               write_all_mins=True,         # save all min paths to .txt
                               write_dim_time=True),        # save runtime data to .txt
                        
                        write_runtime=True,                 # save runtime plot to .png
                        write_min_paths=True,               # save min path plot to .png
                        print_min_paths=True                # print min path solutions in terminal
                    )
# end main()

if __name__ == "__main__":
    main()
