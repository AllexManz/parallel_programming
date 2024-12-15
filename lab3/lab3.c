#include <stdio.h>
#include <stdlib.h>
#include <omp.h>



void traverse(int *array, int size, int gap, int start) {
    for (int i = start + gap; i < size; i += gap){
        int j = i;
        int cur = array[i];
        while (j >= gap && array[j - gap] > cur){
            array[j] = array[j - gap];
            j -= gap;
        }
        array[j] = cur;
    }
}


void shell_sort_parallel(int *array, int size, int threads) {
    int gap = size / 2;
    while (gap > 0) {
        #pragma omp parallel num_threads(threads) shared(array, size, gap) default(none)
            {
                #pragma omp for
                for (int start = 0; start < gap; start++)
                    traverse(array, size, gap, start);
            }
        gap /= 2;
    }
}


void shell_sort_sequential(int* array, int size) {
    int gap = size / 2;
    while (gap) {
        for (int i = gap; i < size; i++) {
            int j = i;
            int cur = array[i];
            while (j >= gap && array[j - gap] > cur) {
                array[j] = array[j - gap];
                j -= gap;
            }
            array[j] = cur;
        }
        gap /= 2;
    }
}


int* array_creation(int length, int random_seed){
    int* array = (int*) calloc(length, sizeof(int));

    for (int i = 0; i < length; i++)
        (array)[i] = rand();

    return array;
}


int* create_reversed_array(int length, int random_seed) {
    int* array = (int*) calloc(length, sizeof(int));

    for (int i = 0; i < length; i++)
        array[i] = length - i;

    return array;
}


int* create_partially_sorted_array(int length, int random_seed) {
    int* array = (int*) calloc(length, sizeof(int));
    int left = length / 4;
    int right = length - left;

    for (int i = 0; i < left; i++) {
        array[i] = rand();
        array[length - 1 - i] = rand();
    }
    for (int i = left; i < right; i++)
        array[i] = i;

    return array;
}


void timing_sequential(int avg, int size, int random_seed) {
    int *array;
    for (int i = 0; i < 3; i++) {
        double time = 0.0;
        for (int j = 0; j < avg; j++) {
            if (i == 0)
                array = array_creation(size, random_seed);
            else if (i == 1)
                array = create_reversed_array(size, random_seed);
            else
                array = create_partially_sorted_array(size, random_seed);

            double start = omp_get_wtime();
            shell_sort_sequential(array, size);
            double end = omp_get_wtime();

            time += end - start;
            free(array);
        }

        if (i == 0)
            printf("[RANDOM] ");
        else if (i == 1)
            printf("[REVERSED] ");
        else
            printf("[PARTIALLY SORTED] ");

        printf("SEQUENTIAL: time = %f;\n", time / avg);
    }
    printf("\n");
}


void timing_parallel(int avg, int size, int min_threads, int random_seed) {
    int *array;
    for (int threads = min_threads; threads <= omp_get_num_procs(); threads++) {
        for (int i = 0; i < 3; i++) {
            double time = 0.0;
            for (int j = 0; j < avg; j++) {
                if (i == 0)
                    array = array_creation(size, random_seed);
                else if (i == 1)
                    array = create_reversed_array(size, random_seed);
                else
                    array = create_partially_sorted_array(size, random_seed);

                double start = omp_get_wtime();
                shell_sort_parallel(array, size, threads);
                double end = omp_get_wtime();

                time += end - start;
                free(array);
            }
            if (i == 0)
                printf("[RANDOM] ");
            else if (i == 1)
                printf("[REVERSED] ");
            else
                printf("[PARTIALLY SORTED] ");

            printf("PARALLEL (%d thr): time = %f;\n", threads, time / avg);
        }
        printf("\n");
    }
}


int main(){
    printf("OpenMP: %d\n", _OPENMP);
    printf("threads: %d\n", omp_get_num_procs());

    int sizes[] = {1000000, 2500000, 5000000, 7500000, 10000000};
    const int repetitions_number = 10;
    const int random_seed = 920215;
    srand(random_seed);

    for (int i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        int size = sizes[i];
        printf("\n\nTIME MEASUREMENT (%d elements)\n", size);
        timing_sequential(repetitions_number, size, random_seed);
        timing_parallel(repetitions_number, size, 2, random_seed);
    }
    return 0;
}
