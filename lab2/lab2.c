#include <stdio.h>
#include <stdlib.h>
#include "omp.h"


int* array_creation(int length, int random_seed){
    int* array = (int*) calloc(length, sizeof(int));

    for (int i = 0; i < length; i++)
        (array)[i] = rand();

    return array;
}


int sequential_find(int n, const int* array, int target){
    int index = -1;
    for (int i = 0; i < n; i++) {
        if (array[i] == target) {
            index = i;
            return index;
        }
    }
    return index;
}


void sequential_calculations(int n, int avg, int random_seed){
    double time = 0.0, start, end;
    for (int iter = 0; iter < avg; iter++) {
        int* array = array_creation(n, random_seed);
        array[0] = -1;
        int target = -1;

        start = omp_get_wtime();
        sequential_find(n, array, target);
        end = omp_get_wtime();

        time += end - start;
        free(array);
    }

    time /= avg;
    printf("BEST SEQUENTIAL TIME: %lf\n", time);

    time = 0.0;
    for (int iter = 0; iter < avg; iter++) {
        int* array = array_creation(n, random_seed);
        array[n - 1] = -1;
        int target = -1;

        start = omp_get_wtime();
        sequential_find(n, array, target);
        end = omp_get_wtime();

        time += end - start;
        free(array);
    }

    time /= avg;
    printf("WORST SEQUENTIAL TIME: %lf\n\n", time);
}


int parallel_find(int n, int threads, const int* array, int target){
    int index = -1;
    #pragma omp parallel num_threads(threads) shared(array, n, target, index) default(none)
    #pragma omp for
    for (int i = 0; i < n; i++) {
        if (array[i] == target) {
            #pragma omp critical
            {
                index = array[i];
            }
            #pragma omp cancel for
        }
        #pragma omp cancellation point for
    }
    return index;
}


void parallel_time(int n, int avg, int random_seed){
    for (int threads = 2; threads <= omp_get_num_procs(); threads++) {
        double time = 0.0, start, end;
        for (int i = 0; i < avg; i++) {
            int* array = array_creation(n, random_seed);
            array[0] = -1;
            int target = -1;

            start = omp_get_wtime();
            parallel_find(n, threads, array, target);
            end = omp_get_wtime();

            time += end - start;
            free(array);
        }

        time /= avg;
        printf("BEST PARALLEL (%d thr): TIME = %lf\n", threads, time);
    }
    printf("\n");

    for (int threads = 2; threads <= omp_get_num_procs(); threads++) {
        double time = 0.0, start, end;
        for (int i = 0; i < avg; i++) {
            int* array = array_creation(n, random_seed);
            array[n - 1] = -1;
            int target = -1;

            start = omp_get_wtime();
            parallel_find(n, threads, array, target);
            end = omp_get_wtime();

            time += end - start;
            free(array);
        }

        time /= avg;
        printf("WORST PARALLEL (%d thr): TIME = %lf\n", threads, time);
    }
    printf("\n\n");
}


int main(int argc, char** argv)
{
    /* Determine the OpenMP support */
    printf("OpenMP: %d\n", _OPENMP);
    printf("threads_num: %d\n\n", omp_get_num_procs());

    const int elements[] = {200000000, 400000000, 600000000, 800000000, 1000000000};
    const int avg = 10;                     ///< Number of calculations before
    const int random_seed = 920215;         ///< RNG seed

    /* Initialize the RNG */
    srand(random_seed);

    for (int i = 0; i < 5; i++) {
        int n_array = elements[i];          ///< Number of array elements
        printf("Number of elements = %d\n", n_array);
        /* Calculate sequential time */
        sequential_calculations(n_array, avg, random_seed);

        /* Calculate parallel time */
        parallel_time(n_array, avg, random_seed);
    }

    return 0;
}
