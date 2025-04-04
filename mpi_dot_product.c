// File: mpi_dot_product.c
// Name: Bradley Stephen
// Date: April 4, 2025
// Assignment: MP1 - Part 3 - MPI Dot Product
//
// Description:
//   This MPI program computes the dot product of two vectors.
//   Process 0 initializes two global vectors (filled with 1.0) and distributes
//   them among all processes using MPI_Scatterv. Each process computes its local
//   dot product, and then all partial sums are reduced (summed) to process 0.
//   The parallel computation is timed using MPI_Wtime() over several runs,
//   and the average runtime is printed along with a correctness check.
//
// Usage:
//   mpicc mpi_dot_product.c -o mpi_dot_product
//   mpirun -np <num_processes> ./mpi_dot_product <global_vector_size> [num_runs]
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int global_n, num_runs = 5;
    double *A = NULL, *B = NULL; // Full vectors on root.
    double *local_A, *local_B;
    double local_dot, global_dot;
    double start_time, end_time, total_time = 0.0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2) {
        if (rank == 0)
            printf("Usage: %s <global_vector_size> [num_runs]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    global_n = atoi(argv[1]);
    if (argc >= 3) {
        num_runs = atoi(argv[2]);
    }
    
    // Prepare counts and displacements for scattering the vectors.
    int *sendcounts = (int*) malloc(size * sizeof(int));
    int *displs = (int*) malloc(size * sizeof(int));
    int base = global_n / size;
    int rem = global_n % size;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base + (i < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + sendcounts[i-1];
    }
    
    int local_n = sendcounts[rank];
    local_A = (double*) malloc(local_n * sizeof(double));
    local_B = (double*) malloc(local_n * sizeof(double));
    
    // Process 0 initializes full vectors A and B.
    if (rank == 0) {
        A = (double*) malloc(global_n * sizeof(double));
        B = (double*) malloc(global_n * sizeof(double));
        for (int i = 0; i < global_n; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
        }
    }
    
    // Repeat runs to compute average time.
    for (int run = 0; run < num_runs; run++) {
        // Scatter the vectors to all processes.
        MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                     local_A, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(B, sendcounts, displs, MPI_DOUBLE,
                     local_B, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        local_dot = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        
        // Each process computes its local dot product.
        for (int i = 0; i < local_n; i++) {
            local_dot += local_A[i] * local_B[i];
        }
        
        end_time = MPI_Wtime();
        double elapsed = end_time - start_time;
        total_time += elapsed;
        
        // Reduce local dot products to get the global dot product on process 0.
        MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // For verification: Since all elements are 1.0, the dot product should equal global_n.
            if (global_dot != global_n * 1.0) {
                printf("Run %d: Error! Parallel dot product = %f, Expected = %d\n", run+1, global_dot, global_n);
            }
        }
    }
    
    if (rank == 0) {
        double avg_time = total_time / num_runs;
        printf("MPI Dot Product Performance\n");
        printf("Processes: %d, Global Vector Size: %d, Runs: %d\n", size, global_n, num_runs);
        printf("Average Time (seconds): %f\n", avg_time);
    }
    
    free(local_A);
    free(local_B);
    free(sendcounts);
    free(displs);
    if (rank == 0) {
        free(A);
        free(B);
    }
    
    MPI_Finalize();
    return 0;
}
