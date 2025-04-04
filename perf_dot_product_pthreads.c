// File: perf_dot_product_pthreads.c
// Name: Bradley Stephen
// Date: April 4, 2025
// Assignment: MP1 - Part 2 - Performance Evaluation (Dot Product using Pthreads)
//
// Description:
//   This program evaluates the performance of the dot product kernel using Pthreads.
//   It accepts command-line arguments for number of threads, a base vector size, scaling mode (strong or weak),
//   and number of runs. For strong scaling, the vector size remains constant;
//   for weak scaling, the effective vector size = base_vector_size * num_threads.
//   The code times only the parallel portion (from thread creation to join) using gettimeofday(),
//   repeats the measurement for multiple runs, and reports the average runtime.
//
// Usage:
//   gcc perf_dot_product_pthreads.c -o perf_dot_product_pthreads -lpthread
//   ./perf_dot_product_pthreads <num_threads> <base_vector_size> <strong|weak> [num_runs]
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>

#define DEFAULT_NUM_RUNS 5

double *A, *B;      // Vectors (allocated dynamically)
double dot_product; // Global dot product result
pthread_mutex_t mutex;

typedef struct {
    int start;
    int end;
} ThreadData;

// Thread function: computes partial dot product
void* dot_product_thread(void* arg) {
    ThreadData *data = (ThreadData*) arg;
    double partial = 0.0;
    for (int i = data->start; i < data->end; i++) {
        partial += A[i] * B[i];
    }
    pthread_mutex_lock(&mutex);
    dot_product += partial;
    pthread_mutex_unlock(&mutex);
    free(data);
    return NULL;
}

// Helper: compute elapsed time in seconds between two timevals
double get_elapsed(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

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
    double seq_dot;

    for (int run = 0; run < num_runs; run++) {
        // Allocate and initialize vectors with 1.0
        A = (double*) malloc(vector_size * sizeof(double));
        B = (double*) malloc(vector_size * sizeof(double));
        if (!A || !B) {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < vector_size; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
        }
        dot_product = 0.0;
        pthread_mutex_init(&mutex, NULL);
        pthread_t threads[num_threads];
        int chunk = vector_size / num_threads;
        int remainder = vector_size % num_threads;
        int start = 0;

        struct timeval t_start, t_end;
        gettimeofday(&t_start, NULL);
        // Create threads
        for (int t = 0; t < num_threads; t++) {
            ThreadData *data = (ThreadData*) malloc(sizeof(ThreadData));
            data->start = start;
            data->end = start + chunk + (t < remainder ? 1 : 0);
            pthread_create(&threads[t], NULL, dot_product_thread, data);
            start = data->end;
        }
        // Join threads
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }
        gettimeofday(&t_end, NULL);
        double elapsed = get_elapsed(t_start, t_end);
        total_time += elapsed;

        // Sequential computation for correctness
        seq_dot = 0.0;
        for (int i = 0; i < vector_size; i++) {
            seq_dot += A[i] * B[i];
        }
        if (dot_product != seq_dot) {
            printf("Run %d: Error! Parallel = %f, Sequential = %f\n", run+1, dot_product, seq_dot);
        }
        free(A);
        free(B);
        pthread_mutex_destroy(&mutex);
    }
    double avg_time = total_time / num_runs;
    printf("Pthreads Dot Product Performance\n");
    printf("Threads: %d, Vector Size: %d, Scaling: %s, Runs: %d\n", num_threads, vector_size, scaling, num_runs);
    printf("Average Time (seconds): %f\n", avg_time);
    return 0;
}
