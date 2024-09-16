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


    ###################################
    #   Initialize our Problem   
    ###################################

    
    ##############################################
    # New problems too large for brute force
    ##############################################
    # PLOTTING BUILT-IN TO ALGORITHM, just set plot_steps= True
    # For faster experimentation with different starting vertices, run the GUI.py scrip

    problem30 = tsp.TSPMap("Random30.tsp")
    # initial vertices chosen because, after visual inspection of problem map,...
    # ...they appear to be away from the center of the map, near a corner, and close to each other
    initial_vertices_30 = [30, 1, 24]
    problem30.closest_edge_insertion(initial_vertices_30, plot_steps=False, line_segment=True)
   
    problem40 = tsp.TSPMap("Random40.tsp")
    # initial vertices chosen because, after visual inspection of problem map,...
    # ...they appear to be away from the center of the map, near a corner, and close to each other
    initial_vertices_40 = [30, 1, 24]
    problem40.closest_edge_insertion(initial_vertices_40, plot_steps=False, line_segment=True)



    ##############################################
    # Previous Problems
    ##############################################

    problem9 = tsp.TSPMap("Random9.tsp")     # N = 9 was solvable for us with brute force; how does runtime compare?
    # initial vertices chosen because, visual, they were closer together,...
    # like a cluster
    initial_vertices_9 = [8, 4, 9]
    problem9.closest_edge_insertion(initial_vertices_9, plot_steps=False, line_segment=True)


    problem10 = tsp.TSPMap("Random10.tsp")    # N = 10 was solvable for us with brute force how does runtime compare?
    # initial vertices chosen because they were located closer to a corner (bottom right-corner)
    initial_vertices_10 = [6, 7, 8]    
    problem10.closest_edge_insertion(initial_vertices_10, plot_steps=False, line_segment=True)

    problem11 = tsp.TSPMap("Random11.tsp")    # N = 11 was NOT solvable for us with brute force;
    # vertices chosen because they were located near a corner, close together, at the edge of the points
    initial_vertices_11 = [1, 2, 4]
    problem11.closest_edge_insertion(initial_vertices_11, plot_steps=False, line_segment=True)

    problem12 = tsp.TSPMap("Random12.tsp")     # N = 12 was NOT solvable for us with brute force;
    # vertices chosen because they were located away from the other points, near the egde of the map
    initial_vertices_12 = [6, 7, 10]
    problem12.closest_edge_insertion(initial_vertices_12, plot_steps=True, line_segment=True)

    ###############################
    #   Runtime Iterations
    ###############################
    # Run the algorithms several times to get some average runtime data
    # Runtime for each of the old problems...
    # Format it as a comma-separated file
    
    with open("closest_edge_insertion_runtimes.csv", "w") as f:
        f.write("random40,random30,random12,random11,random10,random9\n")
        for i in range(1000):           # 1000 iterations


            _ = time.perf_counter()
            problem40.closest_edge_insertion(initial_vertices_40, plot_steps=False, line_segment=True)        # algorithm called here
            elapsed_time40 = time.perf_counter() - _

            _ = time.perf_counter()
            problem30.closest_edge_insertion(initial_vertices_30, plot_steps=False, line_segment=True)        # algorithm called here
            elapsed_time30 = time.perf_counter() - _

            _ = time.perf_counter()
            problem12.closest_edge_insertion(initial_vertices_12, plot_steps=False, line_segment=True)        # algorithm called here
            elapsed_time12 = time.perf_counter() - _

            _ = time.perf_counter()
            problem11.closest_edge_insertion(initial_vertices_11, plot_steps=False, line_segment=True)        # algorithm called here
            elapsed_time11 = time.perf_counter() - _

            _ = time.perf_counter()
            problem10.closest_edge_insertion(initial_vertices_10, plot_steps=False, line_segment=True)        # algorithm called here
            elapsed_time10 = time.perf_counter() - _

            _ = time.perf_counter()
            problem9.closest_edge_insertion(initial_vertices_9, plot_steps=False, line_segment=True)        # algorithm called here
            elapsed_time9 = time.perf_counter() - _



            # Write the elapsed times as tuple to file
            f.write(f"{elapsed_time40:.6},{elapsed_time30:.6},{elapsed_time12:.6},{elapsed_time11:.6},{elapsed_time10:.6},{elapsed_time9:.6}\n")


# end main()

if __name__ == "__main__":
    main()
