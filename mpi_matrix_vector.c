// File: mpi_matrix_vector.c
// Name: Bradley Stephen
// Date: April 4, 2025
// Assignment: MP1 - Part 3 - MPI Matrix-Vector Multiplication
//
// Description:
//   This MPI program performs matrix-vector multiplication.
//   Process 0 initializes a global matrix A (stored in flattened row-major form)
//   and a vector B. The matrix is distributed row-wise among all processes using MPI_Scatterv.
//   Each process computes its portion of the product, and the local results are gathered
//   using MPI_Gatherv into the global result vector P.
//   For weak scaling, the global number of rows is base_M multiplied by the number of processes;
//   for strong scaling, the global matrix rows equal base_M. The parallel portion is timed
//   using MPI_Wtime() over several runs, and the average time is printed along with a correctness check.
//
// Usage:
//   mpicc mpi_matrix_vector.c -o mpi_matrix_vector
//   mpirun -np <num_processes> ./mpi_matrix_vector <base_M> <base_N> <strong|weak> [num_runs]
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int base_M, N, num_runs = 5;
    char scaling_mode[10];
    int global_M; // Total number of rows.
    double *global_A_flat = NULL; // Flattened global matrix (only on root)
    double *B = NULL;  // Global vector B (on root, then broadcast)
    double *P = NULL;  // Global result vector (on root)
    double *local_A;   // Local portion of the matrix (flattened)
    double *local_P;   // Local result for matrix-vector multiplication
    double start_time, end_time, total_time = 0.0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 4) {
        if (rank == 0)
            printf("Usage: %s <base_M> <base_N> <strong|weak> [num_runs]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    base_M = atoi(argv[1]);
    N = atoi(argv[2]);
    strcpy(scaling_mode, argv[3]);
    if (argc >= 5) {
        num_runs = atoi(argv[4]);
    }
    
    // Determine global number of rows based on scaling mode.
    if (strcmp(scaling_mode, "weak") == 0)
        global_M = base_M * size;
    else
        global_M = base_M;
    
    // Prepare counts and displacements for scattering the matrix rows.
    int *sendcounts = (int*) malloc(size * sizeof(int));
    int *displs = (int*) malloc(size * sizeof(int));
    int base = global_M / size;
    int rem = global_M % size;
    for (int i = 0; i < size; i++) {
        int rows = base + (i < rem ? 1 : 0);
        sendcounts[i] = rows * N;  // Number of matrix elements for process i.
    }
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + sendcounts[i-1];
    }
    
    // Prepare separate arrays for gathering the result vector P.
    int *recvcounts = (int*) malloc(size * sizeof(int));
    int *rdispls = (int*) malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        int rows = base + (i < rem ? 1 : 0);
        recvcounts[i] = rows;  // Each process contributes its number of rows.
    }
    rdispls[0] = 0;
    for (int i = 1; i < size; i++) {
        rdispls[i] = rdispls[i-1] + recvcounts[i-1];
    }
    
    int local_elements = sendcounts[rank];  // Number of matrix elements for this process.
    int local_rows = local_elements / N;
    
    local_A = (double*) malloc(local_elements * sizeof(double));
    local_P = (double*) malloc(local_rows * sizeof(double));
    
    // Process 0 initializes the global matrix and vector.
    if (rank == 0) {
        global_A_flat = (double*) malloc(global_M * N * sizeof(double));
        for (int i = 0; i < global_M * N; i++) {
            global_A_flat[i] = 1.0;
        }
        B = (double*) malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) {
            B[j] = 1.0;
        }
        P = (double*) malloc(global_M * sizeof(double));
    }
    
    // Scatter the global matrix rows.
    MPI_Scatterv(global_A_flat, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Broadcast vector B to all processes.
    if (rank == 0) {
        MPI_Bcast(B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        B = (double*) malloc(N * sizeof(double));
        MPI_Bcast(B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Repeat runs and measure performance.
    for (int run = 0; run < num_runs; run++) {
        // Zero local result.
        for (int i = 0; i < local_rows; i++) {
            local_P[i] = 0.0;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        
        // Each process computes its local matrix-vector multiplication.
        for (int i = 0; i < local_rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += local_A[i * N + j] * B[j];
            }
            local_P[i] = sum;
        }
        
        end_time = MPI_Wtime();
        double elapsed = end_time - start_time;
        total_time += elapsed;
        
        // Gather the local result vectors into the global result vector P.
        MPI_Gatherv(local_P, local_rows, MPI_DOUBLE, P, recvcounts, rdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // Verification: Each row's dot product should equal N.
            int error = 0;
            for (int i = 0; i < global_M; i++) {
                if (P[i] != (double) N) {
                    error = 1;
                    break;
                }
            }
            if (error) {
                printf("Run %d: Error in matrix-vector multiplication!\n", run+1);
            }
        }
    }
    
    if (rank == 0) {
        double avg_time = total_time / num_runs;
        printf("MPI Matrix-Vector Multiplication Performance\n");
        printf("Processes: %d, Global Matrix Size: %d x %d, Scaling: %s, Runs: %d\n", size, global_M, N, scaling_mode, num_runs);
        printf("Average Time (seconds): %f\n", avg_time);
    }
    
    free(local_A);
    free(local_P);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(rdispls);
    free(B);
    if (rank == 0) {
        free(P);
        free(global_A_flat);
    }
    
    MPI_Finalize();
    return 0;
}
