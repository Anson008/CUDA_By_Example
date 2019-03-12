#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_exercise.cuh"

#include <cstdio>
#include <cstdlib>

__global__ void initWith(float num, float *a, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x + blockDim.x;

	for (int i = index; i < N; i += stride)
	{
		a[i] = num;
	}
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = index; i < N; i += stride)
	{
		result[i] = a[i] + b[i];
	}
}

void checkElementsAre(float target, float *vector, int N)
{
	for (int i = 0; i < N; i++)
	{
		if (vector[i] != target)
		{
			printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
			exit(1);
		}
	}
	printf("Success! All values calculated correctly.\n");
}

void add_vectors()
{
	int deviceId;
	int numberOfSMs;
	int deviceCount = 0;

	cudaError_t addVectorsErr;
	cudaError_t asyncErr;
	
	cudaGetDeviceCount(&deviceCount);
	printf("Number of Devices: %d\n", deviceCount);

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	printf("Device ID: %d\nNumber of SMs: %d\n", deviceId, numberOfSMs);
	cudaSetDevice(deviceId);

	const int N = 2 << 24;
	size_t size = N * sizeof(float);
	float *a, *b, *c, *h_c;

	cudaMalloc(&a, size);
	cudaMalloc(&b, size);
	cudaMalloc(&c, size);
	cudaMallocHost(&h_c, size);

	size_t threadsPerBlock = 64;
	size_t numberOfBlocks = 32 * numberOfSMs;
	
	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	initWith<<<numberOfBlocks, threadsPerBlock, 0, stream1>>> (3, a, N);
	initWith<<<numberOfBlocks, threadsPerBlock, 0, stream2>>> (4, b, N);
	initWith<<<numberOfBlocks, threadsPerBlock, 0, stream3>>> (0, c, N);
	
	addVectorsInto<<<numberOfBlocks, threadsPerBlock>>> (c, a, b, N);

	cudaMemcpy(h_c, c, size, cudaMemcpyDeviceToHost);

	addVectorsErr = cudaGetLastError();
	if (addVectorsErr != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(addVectorsErr));
	}

	asyncErr = cudaDeviceSynchronize();
	if (asyncErr != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(asyncErr));
	}

	checkElementsAre(7, h_c, N);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFreeHost(h_c);
}