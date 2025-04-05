# run_all_mpi.sh
# Bradley Stephen
# April 4, 2025

# Base parameters for MPI Dot Product
BASE_VECTOR=1000000    
DOT_NUM_RUNS=5

# Base parameters for MPI Matrix-Vector Multiplication
BASE_M=1000            
BASE_N=1000           
MV_NUM_RUNS=5

# Process counts to test
PROCESS_COUNTS=(1 2 4 8 16 32)

echo "Compiling MPI programs for Part 3..."

mpicc mpi_dot_product.c -o mpi_dot_product
mpicc mpi_matrix_vector.c -o mpi_matrix_vector

echo "Compilation complete."

# MPI Dot Product Tests
echo "Running MPI Dot Product (Strong Scaling) Tests..."
for proc in "${PROCESS_COUNTS[@]}"; do
    echo "------------------------------------------------------------" | tee -a mpi_dot_product_strong.txt
    echo "Processes: $proc, Global Vector Size (strong): $BASE_VECTOR" | tee -a mpi_dot_product_strong.txt
    mpirun -np $proc ./mpi_dot_product $BASE_VECTOR $DOT_NUM_RUNS | tee -a mpi_dot_product_strong.txt
done

echo "Running MPI Dot Product (Weak Scaling) Tests..."
for proc in "${PROCESS_COUNTS[@]}"; do
    weak_vector=$(($BASE_VECTOR * proc))
    echo "------------------------------------------------------------" | tee -a mpi_dot_product_weak.txt
    echo "Processes: $proc, Global Vector Size (weak): $weak_vector" | tee -a mpi_dot_product_weak.txt
    mpirun -np $proc ./mpi_dot_product $weak_vector $DOT_NUM_RUNS | tee -a mpi_dot_product_weak.txt
done

# MPI Matrix-Vector Multiplication Tests
echo "Running MPI Matrix-Vector Multiplication (Strong Scaling) Tests..."
for proc in "${PROCESS_COUNTS[@]}"; do
    echo "------------------------------------------------------------" | tee -a mpi_matrix_vector_strong.txt
    echo "Processes: $proc, Global Matrix Size (strong): ${BASE_M}x${BASE_N}" | tee -a mpi_matrix_vector_strong.txt
    mpirun -np $proc ./mpi_matrix_vector $BASE_M $BASE_N strong $MV_NUM_RUNS | tee -a mpi_matrix_vector_strong.txt
done

echo "Running MPI Matrix-Vector Multiplication (Weak Scaling) Tests..."
for proc in "${PROCESS_COUNTS[@]}"; do
    # For weak scaling, global rows = BASE_M * num_processes.
    EFFECTIVE_M=$(($BASE_M * proc))
    echo "------------------------------------------------------------" | tee -a mpi_matrix_vector_weak.txt
    echo "Processes: $proc, Global Matrix Size (weak): ${EFFECTIVE_M}x${BASE_N}" | tee -a mpi_matrix_vector_weak.txt
    mpirun -np $proc ./mpi_matrix_vector $BASE_M $BASE_N weak $MV_NUM_RUNS | tee -a mpi_matrix_vector_weak.txt
done

echo "MPI tests complete lets go."
