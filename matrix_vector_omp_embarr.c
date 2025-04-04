// File: matrix_vector_omp_embarr.c
// Name: Bradley Stephen
// Date: April 4, 2025
// Assignment: MP1 - Part 1(b) - Matrix-Vector Multiplication (Embarrassingly Parallel Approach)
// Description:
//   This program computes matrix-vector multiplication using OpenMP in an embarrassingly parallel way.
//   Each thread processes a chunk of rows independently with no shared variable updates.
//   A sequential version is used to verify the result.
// Usage:
//   Compile with: gcc -fopenmp matrix_vector_omp_embarr.c -o matrix_vector_omp_embarr
//   Run with: ./matrix_vector_omp_embarr

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 10   // Number of rows.
#define N 10   // Number of columns.
#define NUM_THREADS 8

int main() {
    double **A;      // Matrix A.
    double *B, *P;   // Vector B and result vector P.

    // Allocate memory for matrix A.
    A = (double**) malloc(M * sizeof(double*));
    for (int i = 0; i < M; i++) {
        A[i] = (double*) malloc(N * sizeof(double));
    }
    // Allocate memory for vectors B and P.
    B = (double*) malloc(N * sizeof(double));
    P = (double*) malloc(M * sizeof(double));

    if (!A || !B || !P) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Initialize matrix A and vector B with 1.0.
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = 1.0;
    for (int j = 0; j < N; j++)
        B[j] = 1.0;

    omp_set_num_threads(NUM_THREADS);

    // Use an embarrassingly parallel approach where each thread works on its own rows.
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        // Calculate the number of rows per thread.
        int rows_per_thread = M / num_threads;
        int remainder = M % num_threads;

        // Determine the start and end indices for this thread.
        int start = tid * rows_per_thread + (tid < remainder ? tid : remainder);
        int end = start + rows_per_thread + (tid < remainder ? 1 : 0);

        // Compute the partial result for assigned rows.
        for (int i = start; i < end; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A[i][j] * B[j];
            }
            P[i] = sum;
        }
    }

    // Sequential computation for verification.
    double *P_seq = (double*) malloc(M * sizeof(double));
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i][j] * B[j];
        }
        P_seq[i] = sum;
    }

    // Verify that parallel and sequential results match.
    int error = 0;
    for (int i = 0; i < M; i++) {
        if (P[i] != P_seq[i]) {
            error = 1;
            break;
        }
    }

    if (!error)
        printf("=== Matrix-Vector Multiplication (Embarrassingly Parallel) ===\nResult is correct.\n");
    else
        printf("=== Matrix-Vector Multiplication (Embarrassingly Parallel) ===\nThere was an error in the computation.\n");

    // Free all allocated memory.
    for (int i = 0; i < M; i++) {
        free(A[i]);
    }
    free(A);
    free(B);
    free(P);
    free(P_seq);

    return 0;
}
