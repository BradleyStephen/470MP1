"""
plot_performance.py
Name: Bradley Stephen
Date: April 4, 2025
Assignment: MP1 - Performance Graphing Script

Description:
    This script reads two CSV files:
      - performance_results_part2.csv (for shared-memory tests, Part 2)
      - performance_results_part3.csv (for MPI tests, Part 3)
    It then creates individual graphs for each unique test (as indicated by the "Test" column)
    showing the relationship between Thread/Process Count and Average Time (s).
    The graphs are saved as PNG files in separate directories for Part 2 and Part 3.
    
Usage:
    Ensure that "performance_results_part2.csv" and "performance_results_part3.csv"
    are in the same directory as this script.
    Then run:
        chmod +x plot_performance.py
        ./plot_performance.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_test_results(df, test_name, output_dir):
    """
    Filters the dataframe for the given test_name and creates a line graph
    of Count vs. Average Time (s). The graph is saved in output_dir.
    """
    # Filter for the test group
    test_df = df[df["Test"] == test_name].copy()
    if test_df.empty:
        print(f"No data found for test {test_name}")
        return

    # Ensure "Count" is numeric and sort by it
    test_df["Count"] = pd.to_numeric(test_df["Count"])
    test_df.sort_values("Count", inplace=True)
    
    # Prepare x and y data
    x = test_df["Count"]
    y = test_df["Average Time (s)"]
    
    # Create the plot
    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-")
    plt.xlabel("Thread/Process Count")
    plt.ylabel("Average Time (seconds)")
    plt.title(f"{test_name} Performance")
    plt.grid(True)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{test_name}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Graph saved: {filename}")

def plot_all_from_csv(csv_file, output_prefix):
    """
    Reads a CSV file, groups data by unique Test values, and plots each group.
    """
    df = pd.read_csv(csv_file)
    tests = df["Test"].unique()
    out_dir = f"graphs_{output_prefix}"
    os.makedirs(out_dir, exist_ok=True)
    for test in tests:
        plot_test_results(df, test, out_dir)

if __name__ == "__main__":
    # Define the CSV filenames for Part 2 and Part 3.
    part2_csv = "performance_results_part2.csv"
    part3_csv = "performance_results_part3.csv"
    
    print("Generating graphs for Part 2 results...")
    plot_all_from_csv(part2_csv, "part2")
    
    print("Generating graphs for Part 3 results...")
    plot_all_from_csv(part3_csv, "part3")
    
    print("All graphs have been generated and saved.")
