#include <stdio.h>
#include <stdlib.h>
#include "omp.h"


int* array_creation(int length, int random_seed){
    int* array = (int*) calloc(length, sizeof(int));

    for (int i = 0; i < length; i++)
        (array)[i] = rand();

    return array;
}


void sequential_find_max(int n, const int* array, int* max){
    for (int i = 0; i < n; i++)
        if (array[i] > *max)
            *max = array[i];
}


int sequential_calculations(int n, int avg, int random_seed){
    int max = -1;
    double time = 0.0, start, end;
    for (int iter = 0; iter < avg; iter++) {
        int* array = array_creation(n, random_seed);
        max = -1;

        start = omp_get_wtime();
        sequential_find_max(n, array, &max);
        end = omp_get_wtime();

        time += end - start;
        free(array);
    }

    time /= avg;
    printf("SEQUENTIAL TIME: %lf\n\n", time);

    return max;
}


void parallel_find_max(int n, int threads, const int* array, int* max){
    int _max = -1;
    #pragma omp parallel num_threads(threads) shared(array, n) reduction(max: _max) default(none)
    #pragma omp for
    for (int i = 0; i < n; i++)
        if (array[i] > _max)
            _max = array[i];
    *max = _max;
}


int parallel_time(int n, int avg, int random_seed){
    int max = -1;
    for (int threads = 2; threads <= omp_get_num_procs(); threads++) {
        double time = 0.0, start, end;
        for (int i = 0; i < avg; i++) {
            int* array = array_creation(n, random_seed);
            max = -1;

            start = omp_get_wtime();
            parallel_find_max(n, threads, array, &max);
            end = omp_get_wtime();

            time += end - start;
            free(array);
        }

        time /= avg;
        printf("PARALLEL (%d thr): TIME = %lf;\n", threads, time);
    }
    return max;
}


int main(int argc, char** argv)
{
    /* Determine the OpenMP support */
    printf("OpenMP: %d\n", _OPENMP);
    printf("threads_num: %d\n\n", omp_get_num_procs());

    const int n_array = 10000000;         ///< Number of array elements
    const int avg = 10;                     ///< Number of calculations before
    const int random_seed = 920215;         ///< RNG seed
    int seq_max;                            ///< The maximal element for sequential algorithm
    int par_max;                            ///< The maximal element for parallel algorithm

    /* Initialize the RNG */
    srand(random_seed);

    /* Calculate sequential time */
    seq_max = sequential_calculations(n_array, avg, random_seed);

    /* Calculate parallel time */
    par_max = parallel_time(n_array, avg, random_seed);

    printf("======\nSeq_Max is: %d;\n", seq_max);
    printf("======\nPar_Max is: %d;\n", par_max);

    return 0;
}


/*
SEQUENTIAL TIME: 2.517700

PARALLEL (2 thr): TIME = 1.191100;
PARALLEL (3 thr): TIME = 0.892800;
PARALLEL (4 thr): TIME = 0.643700;
PARALLEL (5 thr): TIME = 0.586300;
PARALLEL (6 thr): TIME = 0.481200;
PARALLEL (7 thr): TIME = 0.449800;
PARALLEL (8 thr): TIME = 0.401100;
*/
