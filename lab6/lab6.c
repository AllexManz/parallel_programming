#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ARRAY_SIZE 1000000
#define ITERATIONS 10
#define SEED_VALUE 42


void merge_sorted_sections(int *array, int num_sections, int section_size, int total_elements) {
    int *temp_buffer = (int *)malloc(total_elements * sizeof(int));
    int *indices = (int *)malloc(num_sections * sizeof(int));

    for (int section = 0; section < num_sections; section++) {
        indices[section] = section * section_size;
    }

    for (int pos = 0; pos < total_elements; pos++) {
        int min_idx = -1;
        int min_val = INT_MAX;
        for (int section = 0; section < num_sections; section++) {
            if (indices[section] < (section + 1) * section_size && indices[section] < total_elements) {
                if (array[indices[section]] < min_val) {
                    min_val = array[indices[section]];
                    min_idx = section;
                }
            }
        }

        temp_buffer[pos] = min_val;
        if (min_idx != -1) {
            indices[min_idx]++;
        }
    }

    for (int pos = 0; pos < total_elements; pos++) {
        array[pos] = temp_buffer[pos];
    }

    free(temp_buffer);
    free(indices);
}

void insertion_sort_gap(int arr[], int gap, int length, int start) {
    for (int index = start + gap; index < length; index += gap) {
        int value = arr[index];
        int location = index;

        while (location - gap >= 0 && value < arr[location - gap]) {
            arr[location] = arr[location - gap];
            location -= gap;
        }
        arr[location] = value;
    }
}

void shell_sort(int arr[], int length) {
    int gap = length / 2;
    while (gap > 0) {
        for (int offset = 0; offset < gap; ++offset) {
            insertion_sort_gap(arr, gap, length, offset);
        }
        gap /= 2;
    }
}

void initialize_array(int *array, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        array[i] = (rand() + 1) * rand();
    }
}

void open_output_file(FILE **file, const char *filename) {
    *file = fopen(filename, "a");
    if (*file == NULL) {
        printf("Error: Unable to open file!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char **argv) {
    int rank, size;
    double start_time, total_time = 0.0;
    FILE *result_file = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        open_output_file(&result_file, "/home/papkovas/ClionProjects/par_prog/lab6/new_results");
    }

    int chunk_size = ARRAY_SIZE / size;
    int *global_array = NULL;
    int *local_array = (int *)malloc(chunk_size * sizeof(int));

    if (rank == 0) {
        global_array = (int *)malloc(ARRAY_SIZE * sizeof(int));
    }

    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        if (rank == 0) {
            initialize_array(global_array, ARRAY_SIZE, SEED_VALUE + iteration);
        }

        MPI_Scatter(global_array, chunk_size, MPI_INT, local_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) start_time = MPI_Wtime();
        shell_sort(local_array, chunk_size);
        MPI_Gather(local_array, chunk_size, MPI_INT, global_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            merge_sorted_sections(global_array, size, chunk_size, ARRAY_SIZE);
            total_time += MPI_Wtime() - start_time;
        }
    }

    if (rank == 0) {
        total_time /= ITERATIONS;
        fprintf(result_file, "%d\t%.7f\n", size, total_time);
        fclose(result_file);
        free(global_array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
