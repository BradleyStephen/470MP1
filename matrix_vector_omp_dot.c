// File: matrix_vector_omp_dot.c
// Name: Bradley Stephen
// Date: April 4, 2025
// Assignment: MP1 - Part 1(b) - Matrix-Vector Multiplication (Dot-Product Approach)
// Description:
//   This program performs matrix-vector multiplication using OpenMP.
//   The approach taken is similar to computing a dot product for each row of the matrix.
//   It also computes the same operation sequentially to ensure correctness.
// Usage:
//   Compile with: gcc -fopenmp matrix_vector_omp_dot.c -o matrix_vector_omp_dot
//   Run with: ./matrix_vector_omp_dot

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 10   // Number of rows in the matrix.
#define N 10   // Number of columns in the matrix.
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

    // Initialize the matrix and vector with 1.0.
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = 1.0;
    for (int j = 0; j < N; j++)
        B[j] = 1.0;

    omp_set_num_threads(NUM_THREADS);

    // Use OpenMP to calculate each row's dot product.
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i][j] * B[j];
        }
        P[i] = sum;
    }

    // Sequential calculation for verification.
    double *P_seq = (double*) malloc(M * sizeof(double));
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i][j] * B[j];
        }
        P_seq[i] = sum;
    }

    // Check the results.
    int error = 0;
    for (int i = 0; i < M; i++) {
        if (P[i] != P_seq[i]) {
            error = 1;
            break;
        }
    }

    if (!error)
        printf("=== Matrix-Vector Multiplication (Dot Approach) ===\nResult is correct.\n");
    else
        printf("=== Matrix-Vector Multiplication (Dot Approach) ===\nThere was an error in the computation.\n");

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
