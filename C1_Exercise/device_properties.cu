#include "cuda_exercise.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstring>

void device_properties()
{
	int nDevices;
	int deviceId;

	cudaGetDeviceCount(&nDevices);
	printf("Number of devices: %d\n", nDevices);

	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);

		printf("Device No.: %d\n", i);
		printf("Device name: %s\n", props.name);
		printf("Compute capability: %d.%d\n", props.major, props.minor);
		printf("Number of SMs: %d\n", props.multiProcessorCount);
		printf("Total global memory: %zd\n", props.totalGlobalMem);
		printf("Shared memory per block: %zd\n", props.sharedMemPerBlock);
		printf("Warp size: %d\n", props.warpSize);
		printf("Maximum threads per block: %d\n", props.maxThreadsPerBlock);
		printf("Maximum threads dimesion: %d\n", props.maxThreadsDim[3]);
		printf("Maximum grid size: %d\n", props.maxGridSize[3]);
		printf("Device type: %s\n", props.integrated == 1 ? "Integrated" : "Discrete");
		printf("Concurrent kernels: %d\n", props.concurrentKernels);
		printf("Concurrent managed access: %d\n", props.concurrentManagedAccess);
		printf("Compute mode: %d\n", props.computeMode);
	}
}

