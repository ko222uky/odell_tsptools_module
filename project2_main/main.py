# ====================================================================
#   IMPORT MODULE(S)
# ====================================================================
# import the module that I wrote for this project

import odell_tsptools as tsp    # My custom module
import time                     # for timing individual function calls when timing was not implemented in my module
import pandas as pd             # for reading runtime data which I wrote to CSV file
import scipy.stats as stats     # for the Kruskal-Wallis test
import scikit_posthocs as sp    # for the post-hodc tests (Dunn's with Bonferoroni correction)
                                # Needs to be installed using pip:
                                # $ pip install scikit_posthocs
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
#

def main():

    ###################################
    #   Initialize our Problem   
    ###################################
    # Here, I hard-code the edges
    # The dict[str, list[tuple]] is converted to a corresponding dict[str, list[PathDistance]] in constructor
    edges = {}
    edges["1"] = [(1,2), (1,3), (1,4)]
    edges["2"] = [(2,3)]
    edges["3"] = [(3,4), (3,5)]
    edges["4"] = [(4,5), (4,6), (4,7)]
    edges["5"] = [(5,7), (5,8)]
    edges["6"] = [(6,8)]
    edges["7"] = [(7,9), (7,10)]
    edges["8"] = [(8,9), (8,10), (8,11)]
    edges["9"] = [(9,11)]
    edges["10"] = [(10,11)]

    # Define our starting state and end goal state via Vertex labels
    start = "1"
    goal = "11"

    # Create an instance of our problem
    problem = tsp.TSPMapWithEdges("11PointDFSBFS.tsp", edges)
    # Uncomment to save plot of the problem map; ignore module warning
    # problem.plot_map(save=True)

    ###############################
    #   Runtime Iterations
    ###############################
    # Run the algorithms several times to get some average runtime data
    # Format it as a comma-separated file

    with open("11PointDFSBFS_runtimes.csv", "w") as f:
        f.write("BFS,DFS,UCS\n")
        for i in range(1000):           # 1000 iterations
            _ = time.perf_counter()
            problem.uninformed_search(start, goal, method="breadth-first")
            elapsed_timeBFS = time.perf_counter() - _

            _ = time.perf_counter()
            problem.uninformed_search(start, goal, method="depth-first")
            elapsed_timeDFS = time.perf_counter() - _

            _ = time.perf_counter()
            problem.uninformed_search(start, goal, method="uniform-cost")
            elapsed_timeUCS = time.perf_counter() - _

            # Write the elapsed times as tuple to file
            f.write(f"{elapsed_timeBFS:.6},{elapsed_timeDFS:.6},{elapsed_timeUCS:.6}\n")
    print("\nRuntimes written to 11PointDFSBFS_runtimes.csv")

    # Read the file I just wrote
    runtimes = pd.read_csv("11PointDFSBFS_runtimes.csv")

    print("\nKruskal-Wallis Test for BFS, DFS, and UCS Runtimes")
    statistic, p_values = stats.kruskal(runtimes["BFS"], runtimes["DFS"], runtimes["UCS"]) 
    print(f"Kruskal-Wallis statistic: {statistic:.6f}")
    print(f"P-value: {p_values}")

    ##################################
    #   Plotting the Solution Paths
    ##################################
    #
    # The following code is for obtaining figures for the solution paths
    #
    #
    ###################################
    #   Breadth-first Search
    ###################################
    # Specify breadth-first in uninformed_search() call

    solution_path = problem.uninformed_search(start, goal, method="breadth-first")
    print(f"BFS solution path: {solution_path}")

    # Save the solution path as a plot for BFS
    problem.plot_path(solution_path, "11PointDFSBFS_Breadth_First_Search", save=True)

    print("")

    ###################################
    #   Depth-first Search
    ###################################
    # Specify depth-first in function call

    solution_path = problem.uninformed_search(start, goal, method="depth-first")
    print(f"DFS solution path: {solution_path}")

    # Save a plot showing the solution path for DFS
    problem.plot_path(solution_path, "11PointDFSBFS_Depth_First_Search", save=True)

    print("")

    ###################################
    #   Uniform-cost Search
    ###################################
    # Specify uniform-cost

    solution_path = problem.uninformed_search(start, goal, method="uniform-cost")
    print(f"UCS solution path: {solution_path}")

    # Save a plot showing the solution path for DFS
    problem.plot_path(solution_path, "11PointDFSBFS_Uniform_Cost_Search", save=True)

# end main()

if __name__ == "__main__":
    main()
