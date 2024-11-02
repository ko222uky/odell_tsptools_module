import odell_hitting_strings as ohs


# ================================================ #
#               MAIN FUNCTION
# ================================================ #

def main():
    
    problem = ohs.HittingStringProblem(name = "HSP30_30") # The default problem has 30 strings of length 30.

    for string in problem.string_list:
        print(string)

    print("Wildcard coverage: ", problem.wildcard_coverage_string)
    print("Wildcard coverage percentage: ", problem.wildcard_coverage_percent)


    # Now, let's constrain the coverage!
    print("\n\nConstraining coverage to 10%")
    problem.reduce_wildcard_coverage(10)
    for string in problem.string_list:
        print(string)
    print("Wildcard coverage: ", problem.wildcard_coverage_string)
    print("Wildcard coverage percentage: ", problem.wildcard_coverage_percent)
# end main



if __name__ == "__main__":
    main()