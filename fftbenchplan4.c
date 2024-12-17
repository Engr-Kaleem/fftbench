#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <errno.h>


void benchmark_fft(int N) {
    // Allocate input and output arrays
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    if (!in || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input data
    for (int i = 0; i < N; i++) {
        in[i][0] = (double)rand(); // Real part
        in[i][1] = (double)rand(); // Imaginary part
    }

    // Create a plan for the FFT
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Start timing
    clock_t start = clock();

    // Execute the FFT
    fftw_execute(p);

    // Stop timing
    clock_t end = clock();

    // Calculate elapsed time in seconds
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    int cpu_core = sched_getcpu();

    // Print results
    printf("FFT of size %d completed in %f seconds on CPU core %d.\n", N, time_taken, cpu_core);

    // Print variable sizes
    printf("\nVariable Sizes and Types:\n");
    printf("Size of N: %zu bytes, Type: int\n", sizeof(N));
    printf("Size of in: %zu bytes, Type: fftw_complex*\n", sizeof(in));
    printf("Size of out: %zu bytes, Type: fftw_complex*\n", sizeof(out));
    printf("Size of p: %zu bytes, Type: fftw_plan\n", sizeof(p));
    printf("Size of time_taken: %zu bytes, Type: double\n", sizeof(time_taken));
    printf("Size of cpu_core: %zu bytes, Type: int\n", sizeof(cpu_core));

    // Clean up
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

int main() {
    int N; // Size of the FFT
    int core_number; // ARM core number

    // Ask the user for the FFT size
    printf("Enter the FFT size (e.g., 1024): ");
    scanf("%d", &N);

    // Ask the user for the ARM core number
    printf("Enter the ARM core number to run the FFT on: ");
    scanf("%d", &core_number);

    // Set the CPU affinity to the specified core (if applicable)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_number, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == -1) {
        perror("sched_setaffinity");
        return EXIT_FAILURE;
    }

    // Run the benchmark for the specified number of times
    for (int k = 1; k <= 100; k++) {
        benchmark_fft(N);
    }

    return 0;
}