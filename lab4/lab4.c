#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void get_schedule_info(int *schedule, int *chunk_size) {
    omp_sched_t kind;
    omp_get_schedule(&kind, chunk_size);

    switch (kind) {
        case omp_sched_static:
            *schedule = 1;
            break;
        case omp_sched_dynamic:
            *schedule = 2;
            break;
        case omp_sched_guided:
            *schedule = 3;
            break;
        case omp_sched_auto:
            *schedule = 4;
            break;
        default:
            *schedule = 0;
            break;
    }
}

int function_with_locks(int n, int *unique_values) {
    omp_lock_t lock;
    omp_init_lock(&lock);

    int sum = 0;
    int unique_count = 0;

    for (int i = 0; i < n; i++) {
#pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            omp_set_lock(&lock);
            sum += thread_num;
            omp_unset_lock(&lock);
        }

#pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            sum -= thread_num;
        }

        int is_unique = 1;
        for (int j = 0; j < unique_count; j++) {
            if (unique_values[j] == sum) {
                is_unique = 0;
                break;
            }
        }

        if (is_unique) {
            unique_values[unique_count] = sum;
            unique_count++;
        }
    }
    omp_destroy_lock(&lock);

    return unique_count;
}

int find_max_static(const int *array, int n) {
    int max_value = array[0];
#pragma omp parallel for schedule(static)
    for (int i = 1; i < n; i++)
#pragma omp critical
            if (array[i] > max_value)
                max_value = array[i];
    return max_value;
}

int find_max_dynamic(const int *array, int n) {
    int max_value = array[0];
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < n; i++)
#pragma omp critical
            if (array[i] > max_value)
                max_value = array[i];
    return max_value;
}

int find_max_guided(const int *array, int n) {
    int max_value = array[0];
#pragma omp parallel for schedule(guided)
    for (int i = 1; i < n; i++)
#pragma omp critical
            if (array[i] > max_value)
                max_value = array[i];
    return max_value;
}

int find_max_auto(const int *array, int n) {
    int max_value = array[0];
#pragma omp parallel for schedule(auto)
    for (int i = 1; i < n; i++)
#pragma omp critical
            if (array[i] > max_value)
                max_value = array[i];
    return max_value;
}

int find_max_sequential(const int *array, int n) {
    int max_value = array[0];
    for (int i = 1; i < n; i++)
        if (array[i] > max_value)
            max_value = array[i];
    return max_value;
}

void create_random_array(int length, int **array) {
    *array = (int *)calloc(length, sizeof(int));
    for (int i = 0; i < length; i++)
        (*array)[i] = rand();
}

int main() {
    printf("1) OpenMP: %d\n", _OPENMP);

    printf("2) Number of available processors: %d\n", omp_get_num_procs());
    printf("   Maximum number of threads: %d\n", omp_get_max_threads());

    printf("3) Dynamic threads adjustment is %s\n", omp_get_dynamic() ? "enabled" : "disabled");

    printf("4) Timer resolution: %f seconds\n", omp_get_wtick());

    printf("5) Nested parallelism is %s\n", omp_get_nested() ? "enabled" : "disabled");
    printf("   Maximum active parallel levels: %d\n", omp_get_max_active_levels());

    int schedule;
    int chunk_size;
    get_schedule_info(&schedule, &chunk_size);
    printf("6) Schedule kind: %d\n", schedule);
    printf("   Chunk size: %d\n", chunk_size);

    int n = 1000;
    int unique_values[n];
    int unique_count = function_with_locks(n, unique_values);
    printf("7) Unique values of the variable for %d runs: ", n);
    for (int i = 0; i < unique_count; i++) {
        printf("%d", unique_values[i]);
        printf(i < unique_count - 1 ? ", " : "\n");
    }

    printf("\n8) Finding the maximum element in an array\n");
    int sizes[] = {1000000, 25000000, 50000000, 75000000, 100000000};
    const int repetitions_number = 3;
    const int random_seed = 920215;

    for (int i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        int size = sizes[i];
        printf("\n   TIME MEASUREMENT (%d elements)\n", size);

        double time = 0;
        int *array;
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < repetitions_number; k++) {
                create_random_array(size, &array);

                double t0 = omp_get_wtime();
                if (j == 0) find_max_static(array, size);
                else if (j == 1) find_max_dynamic(array, size);
                else if (j == 2) find_max_guided(array, size);
                else if (j == 3) find_max_auto(array, size);
                else if (j == 4) find_max_sequential(array, size);
                double t1 = omp_get_wtime();
                time += t1 - t0;

                free(array);
            }
            if (j == 0) printf("   STATIC:  time = ");
            else if (j == 1) printf("   DYNAMIC: time = ");
            else if (j == 2) printf("   GUIDED:  time = ");
            else if (j == 3) printf("   AUTO:    time = ");
            else printf("   SEQUENT: time = ");
            printf("%f;\n", time / repetitions_number);
        }
    }

    return 0;
}
