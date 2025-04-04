#!/bin/bash
# File: run_all_perf.sh
# Name: Bradley Stephen
# Date: April 4, 2025
# Assignment: MP1 - Part 2 - Performance Evaluation Automation
#
# Description:
#   This script compiles the performance evaluation programs for the dot product (Pthreads and OpenMP)
#   and matrix-vector multiplication (OpenMP). It then runs each executable with a set of thread counts
#   (1, 2, 4, 8, 16, 32) for both strong scaling (constant problem size) and weak scaling (problem size
#   increases with the number of threads). The results are appended to output log files.
#
# Usage:
#   Make this script executable with: chmod +x run_all_perf.sh
#   Then run: ./run_all_perf.sh

# Base parameters (change as needed)
DOT_BASE_SIZE=1000000       # Base vector size for dot product tests
MAT_BASE_M=1000             # Base number of rows for matrix-vector tests
MAT_BASE_N=1000             # Number of columns for matrix-vector tests
NUM_RUNS=5                  # Number of runs for each configuration

# Thread counts to test
THREADS=(1 2 4 8 16 32)

echo "Compiling performance evaluation codes..."

# Compile the dot product tests for Pthreads and OpenMP, and the matrix-vector multiplication test
gcc -fopenmp perf_dot_product_omp.c -o perf_dot_product_omp
gcc perf_dot_product_pthreads.c -o perf_dot_product_pthreads -lpthread
gcc -fopenmp perf_matrix_vector_omp.c -o perf_matrix_vector_omp

echo "Compilation complete."

echo "Running performance tests for Dot Product (Pthreads) - Strong Scaling"
for t in "${THREADS[@]}"; do
    echo "------------------------------------------------------------" | tee -a perf_dot_product_pthreads_strong.txt
    echo "Threads: $t, Vector Size (strong): $DOT_BASE_SIZE" | tee -a perf_dot_product_pthreads_strong.txt
    ./perf_dot_product_pthreads $t $DOT_BASE_SIZE strong $NUM_RUNS | tee -a perf_dot_product_pthreads_strong.txt
done

echo "Running performance tests for Dot Product (Pthreads) - Weak Scaling"
for t in "${THREADS[@]}"; do
    echo "------------------------------------------------------------" | tee -a perf_dot_product_pthreads_weak.txt
    echo "Threads: $t, Vector Size (weak): $(($DOT_BASE_SIZE * t))" | tee -a perf_dot_product_pthreads_weak.txt
    ./perf_dot_product_pthreads $t $DOT_BASE_SIZE weak $NUM_RUNS | tee -a perf_dot_product_pthreads_weak.txt
done

echo "Running performance tests for Dot Product (OpenMP) - Strong Scaling"
for t in "${THREADS[@]}"; do
    echo "------------------------------------------------------------" | tee -a perf_dot_product_omp_strong.txt
    echo "Threads: $t, Vector Size (strong): $DOT_BASE_SIZE" | tee -a perf_dot_product_omp_strong.txt
    ./perf_dot_product_omp $t $DOT_BASE_SIZE strong $NUM_RUNS | tee -a perf_dot_product_omp_strong.txt
done

echo "Running performance tests for Dot Product (OpenMP) - Weak Scaling"
for t in "${THREADS[@]}"; do
    echo "------------------------------------------------------------" | tee -a perf_dot_product_omp_weak.txt
    echo "Threads: $t, Vector Size (weak): $(($DOT_BASE_SIZE * t))" | tee -a perf_dot_product_omp_weak.txt
    ./perf_dot_product_omp $t $DOT_BASE_SIZE weak $NUM_RUNS | tee -a perf_dot_product_omp_weak.txt
done

echo "Running performance tests for Matrix-Vector Multiplication (OpenMP) - Strong Scaling"
for t in "${THREADS[@]}"; do
    echo "------------------------------------------------------------" | tee -a perf_matrix_vector_omp_strong.txt
    echo "Threads: $t, Matrix Size (strong): ${MAT_BASE_M}x${MAT_BASE_N}" | tee -a perf_matrix_vector_omp_strong.txt
    ./perf_matrix_vector_omp $t $MAT_BASE_M $MAT_BASE_N strong $NUM_RUNS | tee -a perf_matrix_vector_omp_strong.txt
done

echo "Running performance tests for Matrix-Vector Multiplication (OpenMP) - Weak Scaling"
for t in "${THREADS[@]}"; do
    # In weak scaling, the effective number of rows is scaled with threads.
    EFFECTIVE_M=$(($MAT_BASE_M * t))
    echo "------------------------------------------------------------" | tee -a perf_matrix_vector_omp_weak.txt
    echo "Threads: $t, Matrix Size (weak): ${EFFECTIVE_M}x${MAT_BASE_N}" | tee -a perf_matrix_vector_omp_weak.txt
    ./perf_matrix_vector_omp $t $MAT_BASE_M $MAT_BASE_N weak $NUM_RUNS | tee -a perf_matrix_vector_omp_weak.txt
done

echo "Tets Complete lets gooo"
