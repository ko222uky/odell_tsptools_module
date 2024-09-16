
import odell_tsptools as tsp    # My custom module
import time                     # for timing individual function calls when timing was not implemented in my module
import pandas as pd             # for reading runtime data which I wrote to CSV file
import scipy.stats as stats     # for the Kruskal-Wallis test
import scikit_posthocs as sp    # for the post-hodc tests (Dunn's with Bonferoroni correction)
                                # Needs to be installed using pip:
                                # $ pip install scikit_posthocs

import tkinter as tk            #  for GUI
from tkinter import messagebox  # for GUI


def process_inputs():
    filename = filename_entry.get()
    vertices_str = vertices_entry.get()
    steps = plot_steps_entry.get()
    line_segment = line_segments_entry.get()

    # convert inputs to boolean
    if steps == '1':
        steps = True
    else:
        steps = False
    if line_segment == '1':
        line_segment = True
    else:
        line_segment = False

    # try to parse inputs to get appropriate data types
    try:
        vertices = list(map(int, vertices_str.split(',')))
        run_experiment(filename, vertices, steps, line_segment, vertices_str)
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid list of integers separated by commas.")

def run_experiment(filename, vertices, steps, line_segment, vertices_str):

    # replace vertex_str commas with hyphens
    vertices_str = vertices_str.replace(',', '-')

    # pas our processed inputs to instantiate our problem
    problem = tsp.TSPMap(filename)
    solution, _ = problem.closest_edge_insertion(vertices, steps, line_segment)
    messagebox.showinfo("Success", f"Experiment run with filename: {filename} and vertices: {vertices}")
    # write runtime to file
    with open(vertices_str + "_closest_edge_insertion_runtimes_" + filename + ".csv", "w") as f:
        f.write("starting: " + vertices_str + '\n')
        for i in range(1000):           # 1000 iterations
            _ = time.perf_counter()
            problem.closest_edge_insertion(vertices)
            elapsed_time = time.perf_counter() - _

            # Write the elapsed times as tuple to file
            f.write(f"{elapsed_time:.6}\n")

    with open(vertices_str + "_solution_" + filename + ".csv", "w") as f:
        f.write(str(solution))

# Create the main window
root = tk.Tk()
root.title("TSP Problem Input")

# Create and place the filename label and entry
filename_label = tk.Label(root, text="Filename:")
filename_label.grid(row=0, column=0, padx=10, pady=10)
filename_entry = tk.Entry(root)
filename_entry.grid(row=0, column=1, padx=10, pady=10)

# Create and place the vertices label and entry
vertices_label = tk.Label(root, text="Vertices (comma-separated):")
vertices_label.grid(row=1, column=0, padx=10, pady=10)
vertices_entry = tk.Entry(root)
vertices_entry.grid(row=1, column=1, padx=10, pady=10)

# create and place a plot_steps label and line_segments=True label
plot_steps_label = tk.Label(root, text="plot_steps (1 or 0):")
plot_steps_label.grid(row=2, column=0, padx=10, pady=10)
plot_steps_entry = tk.Entry(root)
plot_steps_entry.grid(row=2, column=1, padx=10, pady=10)

line_segments_label = tk.Label(root, text="line_segments (1 or 0):")
line_segments_label.grid(row=3, column=0, padx=10, pady=10)
line_segments_entry = tk.Entry(root)
line_segments_entry.grid(row=3, column=1, padx=10, pady=10)


# Create and place the process button for inputs
process_button = tk.Button(root, text="Process", command=process_inputs)
process_button.grid(row=4, columnspan=2, pady=10)


# Run the application
root.mainloop()