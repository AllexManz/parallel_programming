#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define NUM_RUNS 10
#define ARRAY_SIZE 10000000
#define TOTAL_PROCESSES 5

void initialize_mpi(int argc, char** argv, int* status, int* num_procs, int* rank) {
    *status = MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

void open_result_file(FILE** file, int rank) {
    if (rank == 0) {
        *file = fopen("/home/papkovas/CLionProjects/par_prog/lab5/results", "a");
        if (*file == NULL) {
            printf("Could not open the results file!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void initialize_array(int* array, int seed, int rank) {
    if (rank == 0) {
        srand(seed);
        for (int i = 0; i < ARRAY_SIZE; i++) {
            array[i] = rand();
        }
    }
}

void find_local_max(int* array, int rank, int num_procs, int* local_max) {
    int start = rank * ARRAY_SIZE / num_procs;
    int end = (rank + 1) * ARRAY_SIZE / num_procs;
    *local_max = -1;
    for (int i = start; i < end; i++) {
        if (array[i] > *local_max) {
            *local_max = array[i];
        }
    }
}

int main(int argc, char** argv) {
    int status = -1;
    int num_procs = 0;
    int rank = 0;
    const int seed = 1111;
    int* array = NULL;
    int global_max = -1;
    double start_time, total_time = 0.0;
    FILE* result_file = NULL;

    initialize_mpi(argc, argv, &status, &num_procs, &rank);
    open_result_file(&result_file, rank);

    array = (int*)malloc(sizeof(int) * ARRAY_SIZE);
    for (int run = 0; run < NUM_RUNS; run++) {
        int local_max = -1;

        initialize_array(array, seed + run, rank);
        MPI_Bcast(array, ARRAY_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        find_local_max(array, rank, num_procs, &local_max);

        MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            total_time += MPI_Wtime() - start_time;
        }
    }

    if (rank == 0) {
        total_time /= NUM_RUNS;
        fprintf(result_file, "%.7f\n", total_time);
        fclose(result_file);
    }

    MPI_Finalize();
    free(array);

    return 0;
}
