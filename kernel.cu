#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

using namespace std;

#define PI 3.14159265 

#define CSC(call)  \
do { \
	cudaError_t state = call; \
	if (state != cudaSuccess) { \
		fprintf(stderr, "ERROR: %s:%d. Message: %s\n", __FILE__,__LINE__,cudaGetErrorString(state)); \
		exit(0); \
	} \
} while (0); \

__global__ void kernel(double* v1, long long n) {
    int i, idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long offset = blockDim.x * gridDim.x;
    for (i = idx; i < n; i += offset) {
        v1[i] = sin(PI*((float)i/36));
    }
}

int main()
{
    long long n = 1000000;
    double* sin = (double*)malloc(n * sizeof(double));
    double* sin_dev = (double*)malloc(n * sizeof(double));

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CSC(cudaMalloc(&sin_dev, sizeof(double) * n));
    CSC(cudaMemcpy(sin_dev, sin, sizeof(double) * n, cudaMemcpyHostToDevice));

    cudaEventRecord(start, 0);

    kernel << <256, 256 >> > (sin_dev, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    CSC(cudaMemcpy(sin, sin_dev, sizeof(double) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(sin_dev));

    for (long long i = 0; i < n; i++) {
        printf("%.3f\n", sin[i]);
    }

    printf("\n");
    free(sin);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the kernel: %f ms\n", time);
    return 0;
}
