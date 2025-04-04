// File: perf_dot_product_omp.c
// Name: Bradley Stephen
// Date: April 4, 2025
// Assignment: MP1 - Part 2 - Performance Evaluation (Dot Product using OpenMP)
//
// Description:
//   This program evaluates the performance of the dot product kernel using OpenMP.
//   It accepts command-line arguments for the number of threads, a base vector size,
//   a scaling mode (strong or weak), and the number of runs.
//   For weak scaling, the effective vector size = base_vector_size * num_threads.
//   It uses omp_get_wtime() to time the parallel region, repeats the measurement for multiple runs,
//   and then prints the average execution time.
// Usage:
//   gcc -fopenmp perf_dot_product_omp.c -o perf_dot_product_omp
//   ./perf_dot_product_omp <num_threads> <base_vector_size> <strong|weak> [num_runs]
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define DEFAULT_NUM_RUNS 5

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <num_threads> <base_vector_size> <strong|weak> [num_runs]\n", argv[0]);
        return 1;
    }
    int num_threads = atoi(argv[1]);
    int base_size = atoi(argv[2]);
    char *scaling = argv[3];
    int num_runs = (argc >= 5) ? atoi(argv[4]) : DEFAULT_NUM_RUNS;
    int vector_size = (strcmp(scaling, "weak") == 0) ? base_size * num_threads : base_size;

    double total_time = 0.0;
    double dot_product, seq_dot;

    for (int run = 0; run < num_runs; run++) {
        double *A = (double*) malloc(vector_size * sizeof(double));
        double *B = (double*) malloc(vector_size * sizeof(double));
        if (!A || !B) {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        // Initialize arrays with 1.0
        for (int i = 0; i < vector_size; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
        }
        dot_product = 0.0;
        omp_set_num_threads(num_threads);
        double t_start = omp_get_wtime();
#pragma omp parallel for reduction(+:dot_product)
        for (int i = 0; i < vector_size; i++) {
            dot_product += A[i] * B[i];
        }
        double t_end = omp_get_wtime();
        double elapsed = t_end - t_start;
        total_time += elapsed;

        // Sequential verification
        seq_dot = 0.0;
        for (int i = 0; i < vector_size; i++) {
            seq_dot += A[i] * B[i];
        }
        if (dot_product != seq_dot) {
            printf("Run %d: Error! Parallel = %f, Sequential = %f\n", run+1, dot_product, seq_dot);
        }
        free(A);
        free(B);
    }
    double avg_time = total_time / num_runs;
    printf("OpenMP Dot Product Performance\n");
    printf("Threads: %d, Vector Size: %d, Scaling: %s, Runs: %d\n", num_threads, vector_size, scaling, num_runs);
    printf("Average Time (seconds): %f\n", avg_time);
    return 0;
}
