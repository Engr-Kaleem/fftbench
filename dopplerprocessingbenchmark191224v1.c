#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>

void benchmark_fft(int N, int cols, double *min_time, double *max_time, double *avg_time, double *variance, double *jitter) {
    int rows = N; // Number of rows is the FFT size

    // Allocate input and output arrays
    fftw_complex *in, *out;
    fftw_plan *plans;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
    plans = (fftw_plan*) malloc(sizeof(fftw_plan) * cols);

    if (!in || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input data with random values
    srand(time(NULL)); // Seed for reproducibility
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            in[i * cols + j][0] = rand() / (double)RAND_MAX; // Real part
            in[i * cols + j][1] = rand() / (double)RAND_MAX; // Imaginary part
        }
    }

    // Create a plan for each column
    for (int j = 0; j < cols; j++) {
        plans[j] = fftw_plan_dft_1d(rows, &in[j * rows], &out[j * rows], FFTW_FORWARD, FFTW_ESTIMATE);
    }

    // Measure the time taken for the FFT
    double total_time = 0.0;
    *min_time = __DBL_MAX__;
    *max_time = 0.0;
    double times[100]; // Array to store execution times

    for (int k = 0; k < 100; k++) { // Run the FFT 100 times
        clock_t start = clock();
        for (int j = 0; j < cols; j++) {
            fftw_execute(plans[j]); // Execute the FFT for column j
        }
        clock_t end = clock();
        double time_taken = (double)(end - start) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
        times[k] = time_taken; // Store the time taken

        // Update total time, min, and max
        total_time += time_taken;
        if (time_taken < *min_time) *min_time = time_taken;
        if (time_taken > *max_time) *max_time = time_taken;
    }

    *avg_time = total_time / 100.0; // Average time in milliseconds

    // Calculate variance
    double sum_squared_diff = 0.0;
    for (int k = 0; k < 100; k++) {
        double diff = times[k] - *avg_time;
        sum_squared_diff += diff * diff;
    }
    *variance = sum_squared_diff / 100.0; // Variance in milliseconds squared

    // Calculate jitter (standard deviation)
    *jitter = sqrt(*variance);

    // Print results
    int cpu_core = sched_getcpu();
    printf("FFT of size %d completed on CPU core %d.\n", N, cpu_core);
    printf("Min time: %.3f ms, Max time: %.3f ms, Avg time: %.3f ms\n", *min_time, *max_time, *avg_time);
    printf("Variance: %.6f ms^2, Jitter: %.3f ms\n", *variance, *jitter);

    // Clean up
    for (int j = 0; j < cols; j++) {
        fftw_destroy_plan(plans[j]);
    }

    free(plans);
    fftw_free(in);
    fftw_free(out);
}

int main() {
    int N; // Size of the FFT
    int cols; // Number of columns
    int core_number; // Processor core number

    // Ask the user for the FFT size
    printf("Enter the FFT size (e.g., 1024): ");
    scanf("%d", &N);

    // Ask the user for the number of columns
    printf("Enter the number of bins: ");
    scanf("%d", &cols);

       // Ask the user for the core number to run the benchmark on
    printf("Enter the core number to run the benchmark on (0 for first core, etc.): ");
    scanf("%d", &core_number);

    // Set CPU affinity to the specified core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset); // Clear the CPU set
    CPU_SET(core_number, &cpuset); // Add the specified core to the set

    // Set the CPU affinity for the current process
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
        perror("sched_setaffinity");
        exit(EXIT_FAILURE);
    }

    double min_time, max_time, avg_time, variance, jitter;

    // Run the FFT benchmark
    benchmark_fft(N, cols, &min_time, &max_time, &avg_time, &variance, &jitter);

    return 0;
}