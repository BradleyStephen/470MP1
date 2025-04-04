// File: perf_matrix_vector_omp.c
// Name: Bradley Stephen
// Date: April 4, 2025
// Assignment: MP1 - Part 2 - Performance Evaluation (Matrix-Vector Multiplication using OpenMP)
//
// Description:
//   This program measures the performance of a matrix-vector multiplication kernel using OpenMP.
//   It accepts command-line arguments for the number of threads, base dimensions (M and N) of the matrix,
//   a scaling mode (strong or weak), and the number of runs.
//   For strong scaling, the matrix size remains constant;
//   for weak scaling, the number of rows is scaled: M_effective = base_M * num_threads.
//   The program times the parallel region using omp_get_wtime(), runs several iterations, and then outputs
//   the average execution time.
// Usage:
//   gcc -fopenmp perf_matrix_vector_omp.c -o perf_matrix_vector_omp
//   ./perf_matrix_vector_omp <num_threads> <base_M> <base_N> <strong|weak> [num_runs]
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define DEFAULT_NUM_RUNS 5

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <num_threads> <base_M> <base_N> <strong|weak> [num_runs]\n", argv[0]);
        return 1;
    }
    int num_threads = atoi(argv[1]);
    int base_M = atoi(argv[2]);  // base number of rows
    int base_N = atoi(argv[3]);  // number of columns
    char *scaling = argv[4];
    int num_runs = (argc >= 6) ? atoi(argv[5]) : DEFAULT_NUM_RUNS;

    int M = (strcmp(scaling, "weak") == 0) ? base_M * num_threads : base_M;
    int N = base_N;  // For simplicity, let N remain constant.

    double total_time = 0.0;
    int error;

    for (int run = 0; run < num_runs; run++) {
        // Allocate matrix A (M x N)
        double **A = (double**) malloc(M * sizeof(double*));
        for (int i = 0; i < M; i++) {
            A[i] = (double*) malloc(N * sizeof(double));
        }
        // Allocate vector B and result vector P.
        double *B = (double*) malloc(N * sizeof(double));
        double *P = (double*) malloc(M * sizeof(double));
        if (!A || !B || !P) {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        // Initialize A and B with 1.0.
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = 1.0;
            }
        }
        for (int j = 0; j < N; j++) {
            B[j] = 1.0;
        }
        omp_set_num_threads(num_threads);
        double t_start = omp_get_wtime();
#pragma omp parallel for
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A[i][j] * B[j];
            }
            P[i] = sum;
        }
        double t_end = omp_get_wtime();
        double elapsed = t_end - t_start;
        total_time += elapsed;

        // Sequential verification
        double *P_seq = (double*) malloc(M * sizeof(double));
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A[i][j] * B[j];
            }
            P_seq[i] = sum;
        }
        error = 0;
        for (int i = 0; i < M; i++) {
            if (P[i] != P_seq[i]) {
                error = 1;
                break;
            }
        }
        if (error) {
            printf("Run %d: Error in matrix-vector multiplication!\n", run+1);
        }
        // Free memory for this run.
        for (int i = 0; i < M; i++) {
            free(A[i]);
        }
        free(A);
        free(B);
        free(P);
        free(P_seq);
    }
    double avg_time = total_time / num_runs;
    printf("OpenMP Matrix-Vector Multiplication Performance\n");
    printf("Threads: %d, Matrix Size: %d x %d, Scaling: %s, Runs: %d\n", num_threads, M, N, scaling, num_runs);
    printf("Average Time (seconds): %f\n", avg_time);
    return 0;
}
