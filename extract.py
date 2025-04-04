"""
extract_results_two_csv.py
Name: Bradley Stephen
Date: April 4, 2025
Assignment: MP1 - Parts 2 & 3 - Performance Log Extraction

Description:
    This script scans the current directory for all log files (*.txt) produced by the
    performance automation scripts. It separates the logs into two groups:
      - Part 2 logs (non-MPI logs)
      - Part 3 logs (files whose names contain 'mpi')
    For each block in a log file, it extracts parameters (like thread/process count,
    scaling type, global vector/matrix size) and the corresponding average runtime.
    It then writes two CSV files:
       1. "performance_results_part2.csv" for Part 2 logs.
       2. "performance_results_part3.csv" for Part 3 logs.
       
Usage:
    Ensure all log files are in the same directory as this script.
    Make it executable:
        chmod +x extract_results_two_csv.py
    Then run:
        ./extract_results_two_csv.py
    Two CSV files will be generated in the same directory.
"""

import glob
import re
import csv
import os

# Get all .txt log files in the directory.
log_files = glob.glob("*.txt")

# Lists to store extracted rows for each part.
part2_rows = []
part3_rows = []

# Regular expressions for extracting parameter lines and average time.
# This pattern looks for lines that contain:
#   - "Threads:" or "Processes:" and a number,
#   - followed by either a "Vector Size (strong/weak): <num>", 
#     or "Global Matrix Size (strong/weak): <rows>x<cols>", or
#     "Global Vector Size:" <num>.
param_pattern = re.compile(
    r"(Threads|Processes):\s*(\d+).*?(?:Vector Size \(?(strong|weak)\)?:\s*(\d+)|Global Matrix Size \(?(strong|weak)\)?:\s*(\d+)\s*[xX]\s*(\d+)|Global Vector Size:\s*(\d+))",
    re.DOTALL
)
# Pattern to capture the average time.
time_pattern = re.compile(r"Average Time \(seconds\):\s*([\d\.]+)")

# Process each log file.
for log_file in log_files:
    # Determine if this file is for Part 3 (MPI logs) or Part 2.
    is_mpi = "mpi" in log_file.lower()
    test_type = os.path.splitext(log_file)[0]  # File name without extension.
    
    with open(log_file, "r") as f:
        content = f.read()
    
    # Split the file into blocks; assuming blocks are separated by long dashed lines.
    blocks = re.split(r"[-]{10,}", content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Try to extract parameter info from the block.
        param_match = param_pattern.search(block)
        if not param_match:
            continue

        # Retrieve the count (threads or processes)
        try:
            count = int(param_match.group(2))
        except (TypeError, ValueError):
            count = None

        # Check which size was captured:
        if param_match.group(4):  # vector size from "Vector Size"
            size_str = param_match.group(4)
            scaling = param_match.group(3)
        elif param_match.group(6) and param_match.group(7):  # matrix size captured
            size_str = f"{param_match.group(6)}x{param_match.group(7)}"
            scaling = param_match.group(5)
        elif param_match.group(8):  # alternative vector size format ("Global Vector Size")
            size_str = param_match.group(8)
            # For MPI logs, infer scaling from filename; otherwise assume strong.
            scaling = "weak" if is_mpi and "weak" in test_type.lower() else "strong"
        else:
            size_str = ""
            scaling = ""
        
        # Search for the average time line.
        time_match = time_pattern.search(block)
        if time_match:
            try:
                avg_time = float(time_match.group(1))
            except ValueError:
                continue
        else:
            continue

        # Create a row with the extracted info.
        row = {
            "Log File": log_file,
            "Test": test_type,
            "Scaling": scaling,
            "Count": count,
            "Size": size_str,
            "Average Time (s)": avg_time
        }
        
        # Append row to the appropriate list.
        if is_mpi:
            part3_rows.append(row)
        else:
            part2_rows.append(row)

# Write CSV for Part 2 logs.
csv_file_part2 = "performance_results_part2.csv"
with open(csv_file_part2, "w", newline="") as csvfile:
    fieldnames = ["Log File", "Test", "Scaling", "Count", "Size", "Average Time (s)"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in part2_rows:
        writer.writerow(row)

# Write CSV for Part 3 logs.
csv_file_part3 = "performance_results_part3.csv"
with open(csv_file_part3, "w", newline="") as csvfile:
    fieldnames = ["Log File", "Test", "Scaling", "Count", "Size", "Average Time (s)"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in part3_rows:
        writer.writerow(row)

print(f"Extraction complete. Part 2 results written to {csv_file_part2}")
print(f"Extraction complete. Part 3 results written to {csv_file_part3}")
