// File: dot_product_omp.c
// Name: Bradley Stephen
// Date: April 4, 2025
// Assignment: MP1 - Part 1(a) - Dot Product using OpenMP
// Description:
//   This program calculates the dot product of two vectors using OpenMP.
//   It uses a parallel for loop with a reduction clause so that no manual synchronization is needed.
//   A sequential version verifies the computed result.
// Usage:
//   Compile with: gcc -fopenmp dot_product_omp.c -o dot_product_omp
//   Run with: ./dot_product_omp

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define VECTOR_SIZE 10
#define NUM_THREADS 8

int main() {
    double *A = (double*) malloc(VECTOR_SIZE * sizeof(double));
    double *B = (double*) malloc(VECTOR_SIZE * sizeof(double));
    double dot_product = 0.0;

    if (!A || !B) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Initialize both vectors to 1.0.
    for (int i = 0; i < VECTOR_SIZE; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    // Set the number of threads for OpenMP.
    omp_set_num_threads(NUM_THREADS);

    // Use OpenMP to compute the dot product in parallel.
#pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < VECTOR_SIZE; i++) {
        dot_product += A[i] * B[i];
    }

    // Compute the dot product sequentially for verification.
    double seq_dot = 0.0;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        seq_dot += A[i] * B[i];
    }

    // Print the results.
    printf("=== Dot Product (OpenMP) ===\n");
    printf("Parallel  : %f\n", dot_product);
    printf("Sequential: %f\n", seq_dot);

    free(A);
    free(B);
    return 0;
}
