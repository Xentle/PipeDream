#include <stdio.h>

__global__ void dummy()
{
	int j = 0;
	for(int i = 0; i < 1000000; i++)
		j++;
}

int main()
{
	cudaStream_t stream1, stream2;

	double *A, *B, *C, *D;

	cudaSetDevice(1);
	cudaMalloc((void **) &C, 100000000 * sizeof(double));
	cudaMalloc((void **) &D, 10000000 * sizeof(double));

	cudaSetDevice(0);
	cudaMalloc((void **) &A, 100000000 * sizeof(double));
	cudaMalloc((void **) &B, 10000000 * sizeof(double));

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaDeviceEnablePeerAccess(1, 0);

	dummy<<<1, 1>>>();

	cudaSetDevice(0);
	cudaMemcpyPeerAsync(C, 1, A, 0, 100000000 * sizeof(double), stream1);

	cudaSetDevice(1);
	for(int i = 0; i < 10; i++)
		dummy<<<1, 1>>>();

	cudaSetDevice(0);
	cudaMemcpyPeerAsync(D, 1, B, 0, 10000000 * sizeof(double));

	
	
	for(int i = 0; i < 2; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
	return 0;
}

