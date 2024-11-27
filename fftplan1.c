#include <stdio.h>

#include <stdlib.h>

#include <fftw3.h>

#include <unistd.h>

#include <sched.h>

#include <pthread.h>

#include <sys/resource.h>

#include <sys/time.h>


#define N 1024 // Size of the FFT


void set_cpu_affinity(int cpu_id) {

    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);

    CPU_SET(cpu_id, &cpuset);

    pthread_t current_thread = pthread_self();

    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

}


void print_cpu_usage() {

    struct rusage usage;

    getrusage(RUSAGE_SELF, &usage);

    printf("User  CPU time: %ld.%06ld seconds\n",

           usage.ru_utime.tv_sec, usage.ru_utime.tv_usec);

    printf("System CPU time: %ld.%06ld seconds\n",

           usage.ru_stime.tv_sec, usage.ru_stime.tv_usec);

}


int main() {

    // Set CPU affinity to core 0

    set_cpu_affinity(0);


    // Allocate input and output arrays

    fftw_complex *in, *out;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);


    // Initialize input data

    for (int i = 0; i < N; i++) {

        in[i][0] = i;    // Real part

        in[i][1] = 0.0;  // Imaginary part

    }


    // Create a plan for FFT

    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_ESTIMATE);


    // Start timing

    struct timeval start, end;

    gettimeofday(&start, NULL);


    // Execute the FFT

    fftw_execute(p);


    // Stop timing

    gettimeofday(&end, NULL);


    // Print execution time

    long seconds = end.tv_sec - start.tv_sec;

    long microseconds = end.tv_usec - start.tv_usec;

    double elapsed = seconds + microseconds * 1e-6;

    printf("FFT execution time: %.6f seconds\n", elapsed);


    // Print CPU usage

    print_cpu_usage();


    // Cleanup

    fftw_destroy_plan(p);

    fftw_free(in);

    fftw_free(out);


    return 0;

}