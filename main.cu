#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_WORDS 67108864
#define NUMBER_OF_TESTS 100

// Device initialization
void init_gpu(cudaDeviceProp *deviceProp){
	// Detect GPU
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess){
		printf("Error: cudaGetDeviceCount returns %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
	if (deviceCount == 0){
		printf("No CUDA device found\n");
		exit(EXIT_FAILURE);
	}
	// Use the first device found
	cudaSetDevice(0);
	cudaGetDeviceProperties(deviceProp, 0);
}

int main(int argc, char **argv){
  // Host buffer
  double *buf;
  // Device buffer
  double *devbuf;
  // Measure execution time
	struct timespec starttime, endtime;
  double elapsedtime;
  clock_t starttic, endtic, tics;
  // Counters
  size_t i, w, t;

	// GPU identification and initialization
  cudaDeviceProp deviceProp;
	init_gpu(&deviceProp);

  printf("Iteration\tWords\tBytes\tTicks\tMbit/sec\n");

  for ( i = 0; pow(2, i) <= MAX_WORDS; i++ ){
    // Amount of words (word = double)
    w = pow(2, i);

    // Allocate host buffer
    cudaMallocHost(&buf, w * sizeof(double));

    // Allocate device buffer
    cudaMalloc(&devbuf, w * sizeof(double));

    elapsedtime = 0;
    tics = 0;
    for ( t = 0; t < NUMBER_OF_TESTS; t++ ){
      clock_gettime(CLOCK_MONOTONIC, &starttime);
      starttic = clock();
      cudaMemcpy(devbuf, buf, w * sizeof(double), cudaMemcpyHostToDevice);
      clock_gettime(CLOCK_MONOTONIC, &endtime);
      endtic = clock();
      elapsedtime += ( endtime.tv_sec - starttime.tv_sec ) + ( endtime.tv_nsec - starttime.tv_nsec ) / 1e9;
      tics += endtic - starttic;
    }

    // Elapsed time 
    elapsedtime /= NUMBER_OF_TESTS;
    tics /= NUMBER_OF_TESTS;
    // Mbits / sec
    double bandwidth;
    bandwidth = w * sizeof(double) * 1.0e-6 * 8 / elapsedtime;
    printf("%ld\t%ld\t%ld\t%ld\t%ld\n", i, w, w * sizeof(double), (size_t)tics, (size_t)bandwidth);
    elapsedtime = 0;

    cudaFree(devbuf);
    cudaFreeHost(buf);
  }

  exit(EXIT_SUCCESS);
}
