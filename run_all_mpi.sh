#!/bin/bash
# File: run_all_mpi.sh
# Name: Bradley Stephen
# Date: April 4, 2025
# Assignment: MP1 - Part 3 - MPI Performance Evaluation Automation
#
# Description:
#   This script compiles the MPI programs for the dot product and matrix-vector multiplication,
#   and then runs a series of tests for both strong and weak scaling scenarios.
#   For MPI Dot Product:
#     - Strong scaling: Global vector size remains constant.
#     - Weak scaling: Global vector size = BASE_VECTOR * num_processes.
#   For MPI Matrix-Vector Multiplication:
#     - Strong scaling: Global matrix rows = BASE_M.
#     - Weak scaling: Global matrix rows = BASE_M * num_processes.
#
# Usage:
#   Make the script executable: chmod +x run_all_mpi.sh
#   Then run: ./run_all_mpi.sh

# Base parameters for MPI Dot Product
BASE_VECTOR=1000000    # Global vector size for strong scaling
DOT_NUM_RUNS=5

# Base parameters for MPI Matrix-Vector Multiplication
BASE_M=1000            # Base number of rows for strong scaling (global rows = BASE_M)
BASE_N=1000            # Number of columns (remains constant)
MV_NUM_RUNS=5

# Process counts to test
PROCESS_COUNTS=(1 2 4 8 16 32)

echo "Compiling MPI programs for Part 3..."

mpicc mpi_dot_product.c -o mpi_dot_product
mpicc mpi_matrix_vector.c -o mpi_matrix_vector

echo "Compilation complete."

# MPI Dot Product Tests
echo "Running MPI Dot Product (Strong Scaling) Tests..."
for proc in "${PROCESS_COUNTS[@]}"; do
    echo "------------------------------------------------------------" | tee -a mpi_dot_product_strong.txt
    echo "Processes: $proc, Global Vector Size (strong): $BASE_VECTOR" | tee -a mpi_dot_product_strong.txt
    mpirun -np $proc ./mpi_dot_product $BASE_VECTOR $DOT_NUM_RUNS | tee -a mpi_dot_product_strong.txt
done

echo "Running MPI Dot Product (Weak Scaling) Tests..."
for proc in "${PROCESS_COUNTS[@]}"; do
    weak_vector=$(($BASE_VECTOR * proc))
    echo "------------------------------------------------------------" | tee -a mpi_dot_product_weak.txt
    echo "Processes: $proc, Global Vector Size (weak): $weak_vector" | tee -a mpi_dot_product_weak.txt
    mpirun -np $proc ./mpi_dot_product $weak_vector $DOT_NUM_RUNS | tee -a mpi_dot_product_weak.txt
done

# MPI Matrix-Vector Multiplication Tests
echo "Running MPI Matrix-Vector Multiplication (Strong Scaling) Tests..."
for proc in "${PROCESS_COUNTS[@]}"; do
    echo "------------------------------------------------------------" | tee -a mpi_matrix_vector_strong.txt
    echo "Processes: $proc, Global Matrix Size (strong): ${BASE_M}x${BASE_N}" | tee -a mpi_matrix_vector_strong.txt
    mpirun -np $proc ./mpi_matrix_vector $BASE_M $BASE_N strong $MV_NUM_RUNS | tee -a mpi_matrix_vector_strong.txt
done

echo "Running MPI Matrix-Vector Multiplication (Weak Scaling) Tests..."
for proc in "${PROCESS_COUNTS[@]}"; do
    # For weak scaling, global rows = BASE_M * num_processes.
    EFFECTIVE_M=$(($BASE_M * proc))
    echo "------------------------------------------------------------" | tee -a mpi_matrix_vector_weak.txt
    echo "Processes: $proc, Global Matrix Size (weak): ${EFFECTIVE_M}x${BASE_N}" | tee -a mpi_matrix_vector_weak.txt
    mpirun -np $proc ./mpi_matrix_vector $BASE_M $BASE_N weak $MV_NUM_RUNS | tee -a mpi_matrix_vector_weak.txt
done

echo "MPI tests complete lets go."
