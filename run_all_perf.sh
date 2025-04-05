# run_all_perf.sh
# Bradley Stephen
# April 4, 2025

# Base parameters (change as needed)
DOT_BASE_SIZE=1000000      
MAT_BASE_M=1000            
MAT_BASE_N=1000        
NUM_RUNS=5                  

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
