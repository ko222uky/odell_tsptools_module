# ====================================================================
#   IMPORT MODULE(S)
# ====================================================================
# import the module that I wrote for this project

import odell_tsptools as tsp
import time # for timing individual function calls when timing was not implemented in my module

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
    # Show the problem map
    problem.plot_map(save=True)

    ###################################
    #   Breadth-first Search
    ###################################
    # Specify breadth-first in uninformed_search() call
    _ = time.perf_counter()                     # start timer
    solution_path = problem.uninformed_search(start, goal, method="breadth-first")
    elapsed_time = time.perf_counter() - _      # end timer and calculate elapsed time
    print(f"Breadth-First Search runtime: {elapsed_time:.4}")
    print(f"Solution path: {solution_path}")

    # Save the solution path as a plot for BFS
    problem.plot_path(solution_path, "11PointDFSBFS_Breadth_First_Search", save=True)

    print("")

    ###################################
    #   Depth-first Search
    ###################################
    # Specify depth-first in function call
    _ = time.perf_counter()
    solution_path = problem.uninformed_search(start, goal, method="depth-first")
    elapsed_time = time.perf_counter() - _
    print(f"Depth-First Search runtime: {elapsed_time:.4}")
    print(f"Solution path: {solution_path}")

    # Save a plot showing the solution path for DFS
    problem.plot_path(solution_path, "11PointDFSBFS_Depth_First_Search", save=True)

    print("")

    ###################################
    #   Uniform-cost Search
    ###################################
    # Specify uniform-cost
    _ = time.perf_counter()
    solution_path = problem.uninformed_search(start, goal, method="uniform-cost")
    elapsed_time = time.perf_counter() - _
    print(f"Uniform-Cost Search runtime: {elapsed_time:.4}")
    print(f"Solution path: {solution_path}")

    # Save a plot showing the solution path for DFS
    problem.plot_path(solution_path, "11PointDFSBFS_Uniform_Cost_Search", save=True)

# end main()

if __name__ == "__main__":
    main()
