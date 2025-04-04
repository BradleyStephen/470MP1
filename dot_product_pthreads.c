// File: dot_product_pthreads.c
// Name: Bradley Stephen
// Date: April 4, 2025
// Assignment: MP1 - Part 1(a) - Dot Product using Pthreads
// Description:
//   This program computes the dot product of two vectors using Pthreads.
//   Each thread calculates a part of the dot product, then safely updates the shared result.
//   A sequential computation is also done to verify the correctness.
// Usage:
//   Compile with: gcc dot_product_pthreads.c -o dot_product_pthreads -lpthread
//   Run with: ./dot_product_pthreads

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define VECTOR_SIZE 10   // Small size for demonstration; change as needed.
#define NUM_THREADS 8    // We'll use 8 threads as per the assignment requirements.

double *A, *B;              // Global vectors to be processed.
double dot_product = 0.0;   // Global result variable for the dot product.
pthread_mutex_t mutex;      // Mutex to protect the shared dot_product during updates.

typedef struct {
    int start;
    int end;
} ThreadData;

// Thread function that calculates a partial dot product.
void* dot_product_thread(void* arg) {
    ThreadData *data = (ThreadData*) arg;
    double partial_sum = 0.0;
    // Each thread processes its chunk of the vector.
    for (int i = data->start; i < data->end; i++) {
        partial_sum += A[i] * B[i];
    }
    // Lock the mutex to update the global dot_product safely.
    pthread_mutex_lock(&mutex);
    dot_product += partial_sum;
    pthread_mutex_unlock(&mutex);

    free(data);
    return NULL;
}

int main() {
    // Allocate memory for the two vectors.
    A = (double*) malloc(VECTOR_SIZE * sizeof(double));
    B = (double*) malloc(VECTOR_SIZE * sizeof(double));
    if (!A || !B) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Initialize both vectors with 1.0 for simplicity.
    for (int i = 0; i < VECTOR_SIZE; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    // Initialize mutex for thread synchronization.
    pthread_mutex_init(&mutex, NULL);
    pthread_t threads[NUM_THREADS];

    // Determine the workload for each thread.
    int chunk_size = VECTOR_SIZE / NUM_THREADS;
    int remainder = VECTOR_SIZE % NUM_THREADS;
    int start_index = 0;

    // Create threads to compute parts of the dot product.
    for (int t = 0; t < NUM_THREADS; t++) {
        ThreadData *data = (ThreadData*) malloc(sizeof(ThreadData));
        data->start = start_index;
        // Distribute any leftover elements among the first few threads.
        data->end = start_index + chunk_size + (t < remainder ? 1 : 0);
        pthread_create(&threads[t], NULL, dot_product_thread, data);
        start_index = data->end;
    }

    // Wait for all threads to finish.
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
    pthread_mutex_destroy(&mutex);

    // Do a sequential computation to verify correctness.
    double seq_dot = 0.0;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        seq_dot += A[i] * B[i];
    }

    // Print results.
    printf("=== Dot Product (Pthreads) ===\n");
    printf("Parallel  : %f\n", dot_product);
    printf("Sequential: %f\n", seq_dot);

    free(A);
    free(B);
    return 0;
}
