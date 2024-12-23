#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int is_prime_number(int num) {
    for (int i = 2; i <= (int)sqrt(num); i++) {
        if (num % i == 0) {
            return 0;
        }
    }
    return 1;
}

int generate_prime_list(int* prime_array, int start_value, int end_value) {
    int count = 0;
#pragma omp parallel shared(start_value, end_value, prime_array, count) default(none)
    {
#pragma omp for
        for (int i = start_value; i < end_value; i++) {
            if (is_prime_number(i)) {
#pragma omp critical
                {
                    prime_array[count] = i;
                    count++;
                }
            }
        }
    }
    return count;
}

int estimate_prime_count(int range_value) {
    return range_value / log(range_value);
}

void initialize_mpi(int argc, char** argv, int* rank, int* num_processes, int* provided) {
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, provided);
    if (*provided < MPI_THREAD_MULTIPLE) {
        printf("Error\n");
    }
    MPI_Comm_size(MPI_COMM_WORLD, num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

void finalize_mpi() {
    MPI_Finalize();
}

void set_openmp_threads(int num_threads) {
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
}

int main(int argc, char** argv) {
    int range = 100000000;

    int rank = 0;
    int num_processes = 0;
    int provided;
    initialize_mpi(argc, argv, &rank, &num_processes, &provided);

    int num_threads = 16 / num_processes;
    set_openmp_threads(num_threads);
    if (!rank) {
        printf("Processors: %d, threads: %d\n", num_processes, num_threads);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int local_range = range / num_processes;

    int start_value = local_range * rank;
    int end_value = local_range * (rank + 1);
    double time_start, time_end;
    int array_size = 2 * (estimate_prime_count(end_value) - estimate_prime_count(start_value));
    int* prime_array = (int*) calloc(array_size, sizeof(int));
    int* sizes = NULL;
    int* displacements = NULL;
    if (!rank) {
        sizes = (int*) malloc(num_processes * sizeof(int));
        displacements = (int*)calloc(num_processes, sizeof(int));
    }
    time_start = MPI_Wtime();
    int found_count = generate_prime_list(prime_array, start_value, end_value);
    time_end = MPI_Wtime() - time_start;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&found_count, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!rank) {
        for (int i = 1; i < num_processes; i++) {
            displacements[i] = displacements[i - 1] + sizes[i - 1];
        }
    }
    int result_size = 0;
    MPI_Reduce(&found_count, &result_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int* result = (int*)calloc(result_size, sizeof(int));
    MPI_Gatherv(prime_array, found_count, MPI_INT, result, sizes, displacements, MPI_INT, 0, MPI_COMM_WORLD);

    double total_time = MPI_Wtime() - time_start;

    if (!rank) {
        printf("Total time = %lf\n", total_time);
    }

    double total_execution_time = 0;
    MPI_Reduce(&time_end, &total_execution_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    printf("%d - %lf ", rank, time_end);
    printf("\n");
    finalize_mpi();

    return 0;
}