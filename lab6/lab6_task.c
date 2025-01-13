#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int getMax(int *arr, int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
    return max;
}


void countingSort(int *arr, int n, int exp) {
    int *output = (int *)malloc(n * sizeof(int));
    int count[10] = {0};

    for (int i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    for (int i = 0; i < n; i++)
        arr[i] = output[i];

    free(output);
}


void my_radixsort(int *arr, int n) {
    int max = getMax(arr, n);

    for (int exp = 1; max / exp > 0; exp *= 10)
        countingSort(arr, n, exp);
}


void generateRandomArray(int *arr, int n, int max_value) {
    for (int i = 0; i < n; i++)
        arr[i] = rand() % max_value;
}


double measureTime(void (*sortFunc)(int *, int), int *arr, int n, int iterations) {
    clock_t start, end;
    double cpu_time_used;
    double total_time = 0.0;

    for (int i = 0; i < iterations; i++) {
        generateRandomArray(arr, n, 1000000);
        start = clock();
        sortFunc(arr, n);
        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        total_time += cpu_time_used;
    }

    return total_time / iterations;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000000; // Размер массива
    int *arr = NULL;
    int *local_arr = NULL;
    int local_n = n / size;
    int iterations = 5;

    if (rank == 0) {
        arr = (int *)malloc(n * sizeof(int));
    }

    local_arr = (int *)malloc(local_n * sizeof(int));

    double total_time = 0.0;
    if (size == 1) {
        // Seq
        total_time = measureTime(my_radixsort, arr, n, iterations);
        printf("SEQUENTIAL: %f seconds\n", total_time);
    } else {
        // Par
        for (int i = 0; i < iterations; i++) {
            if (rank == 0) {
                generateRandomArray(arr, n, 1000000);
            }

            MPI_Scatter(arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

            double start_time = MPI_Wtime();
            my_radixsort(local_arr, local_n);
            double end_time = MPI_Wtime();

            total_time += (end_time - start_time);

            MPI_Gather(local_arr, local_n, MPI_INT, arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);
        }

        double avg_time = total_time / iterations;
        if (rank == 0) {
            printf("PARALLEL: %f seconds\n", avg_time);
        }
    }

    free(local_arr);
    if (rank == 0) {
        free(arr);
    }

    MPI_Finalize();
    return 0;
}
