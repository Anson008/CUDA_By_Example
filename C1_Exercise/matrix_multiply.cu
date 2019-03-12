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


__global__ void matrixMulGPU(int *a, int *b, int *c)
{
	int val = 0;

	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int col = threadIdx.y + blockIdx.y * blockDim.y;

	if (row < N && col < N)
	{
		for (int k = 0; k < N; ++k)
		{
			val += a[k + row * N] * b[col + k * N];
		}
		c[col + row * N] = val;
	}
}

void matrixMulCPU(int *a, int *b, int *c)
{
	int val = 0;

	for (int row = 0; row < N; ++row)
	{
		for (int col = 0; col < N; ++col)
		{
			val = 0;
			for (int k = 0; k < N; ++k)
			{
				val += a[k + row * N] * b[col + k * N];
			}
			c[col + row * N] = val;
		}
	}
}


void multiply_matrix()
{
	int *a_dev, *b_dev, *c_cpu_dev, *c_gpu_dev;
	int *a_h, *b_h, *c_cpu_h, *c_gpu_h;
	size_t size = N * N * sizeof(int);

	cudaMalloc(&a_dev, size);
	cudaMalloc(&b_dev, size);
	cudaMalloc(&c_gpu_dev, size);
	cudaMalloc(&c_cpu_dev, size);

	cudaMallocHost(&a_h, size);
	cudaMallocHost(&b_h, size);
	cudaMallocHost(&c_cpu_h, size);
	cudaMallocHost(&c_gpu_h, size);

	dim3 threads_per_block(16, 16, 1);
	dim3 number_of_blocks((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

	initVectors<<<threads_per_block, number_of_blocks>>> (a_dev, b_dev, c_cpu_dev, c_gpu_dev);
	matrixMulGPU<<<number_of_blocks, threads_per_block>>> (a_dev, b_dev, c_gpu_dev);

	cudaMemcpy(a_h, a_dev, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b_h, b_dev, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c_cpu_h, c_cpu_dev, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c_gpu_h, c_gpu_dev, size, cudaMemcpyDeviceToHost);

	cudaError_t mulErr;
	cudaError_t asyncErr;

	mulErr = cudaGetLastError();
	if (mulErr != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(mulErr));
	}

	asyncErr = cudaDeviceSynchronize();
	if (asyncErr != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(asyncErr));
	}

	// Call the CPU version to check our work
	matrixMulCPU(a_h, b_h, c_cpu_h);

	// Compare the two answers to make sure they are equal
	bool error = false;
	for (int row = 0; row < N && !error; ++row)
	{
		for (int col = 0; col < N && !error; ++col)
		{
			if (c_cpu_h[row * N + col] != c_gpu_h[row * N + col])
			{
				printf("FOUND ERROR at c[%d][%d]\n", row, col);
				error = true;
				break;
			}
		}
	}
		
	if (!error)
		printf("Success!\n");

	cudaFree(a_dev);
	cudaFree(b_dev);
	cudaFree(c_gpu_dev);
	cudaFree(c_cpu_dev);

	cudaFreeHost(a_h);
	cudaFreeHost(b_h);
	cudaFreeHost(c_cpu_h);
	cudaFreeHost(c_gpu_h);
}