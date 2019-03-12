#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_exercise.cuh"

#include <cstdio>
#include <cstdlib>

#define N 64

__global__ void initVectors(int *a, int *b, int *c_cpu, int *c_gpu)
{
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int col = threadIdx.y + blockIdx.y * blockDim.y;

	if (row < N && col < N)
	{
		a[row*N + col] = row;
		b[row*N + col] = col + 2;
		c_cpu[row*N + col] = 0;
		c_gpu[row*N + col] = 0;
	}
}

__global__ void matrixMulGPU(int * a, int * b, int * c)
{
	int val = 0;

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < N && col < N)
	{
		for (int k = 0; k < N; ++k)
			val += a[row * N + k] * b[k * N + col];
		c[row * N + col] = val;
	}
}

void matrixMulCPU(int * a, int * b, int * c)
{
	int val = 0;

	for (int row = 0; row < N; ++row)
		for (int col = 0; col < N; ++col)
		{
			val = 0;
			for (int k = 0; k < N; ++k)
				val += a[row * N + k] * b[k * N + col];
			c[row * N + col] = val;
		}
}

void multiply_matrix()
{
	int *a, *b, *c_cpu, *c_gpu;
	size_t size = N * N * sizeof(int);

	cudaMalloc(&a, size);
	cudaMalloc(&b, size);
	cudaMalloc(&c_cpu, size);
	cudaMalloc(&c_gpu, size);

	int *h_a, *h_b, *h_c_cpu, *h_c_gpu;
	cudaMallocHost(&h_a, size);
	cudaMallocHost(&h_b, size);
	cudaMallocHost(&h_c_cpu, size);
	cudaMallocHost(&h_c_gpu, size);

	dim3 threads_per_block(16, 16, 1);
	dim3 numOfBlocks((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

	initVectors << <threads_per_block, numOfBlocks >> > (a, b, c_cpu, c_gpu);

	matrixMulGPU << <threads_per_block, numOfBlocks >> > (a, b, c_gpu);
	
	cudaMemcpy(h_a, a, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, b, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c_cpu, c_cpu, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c_gpu, c_gpu, size, cudaMemcpyDeviceToHost);

	cudaError_t codeErr;
	cudaError_t asyncErr;

	codeErr = cudaGetLastError();
	if (codeErr != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(codeErr));
	}

	asyncErr = cudaDeviceSynchronize();
	if (asyncErr != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(asyncErr));
	}

	matrixMulCPU(h_a, h_b, h_c_cpu);

	bool error = false;
	for (int row = 0; row < N && !error; ++row)
		for (int col = 0; col < N && !error; ++col)
			if (h_c_cpu[row * N + col] != h_c_gpu[row * N + col])
			{
				printf("FOUND ERROR at c[%d][%d]\n", row, col);
				error = true;
				break;
			}
	if (!error)
		printf("Success!\n");

	cudaFree(a);
	cudaFree(b);
	cudaFree(c_cpu);
	cudaFree(c_gpu);
	
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c_cpu);
	cudaFreeHost(h_c_gpu);
}