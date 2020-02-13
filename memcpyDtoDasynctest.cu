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

	cudaSetDevice(2);
	cudaMalloc((void **) &C, 100000000 * sizeof(double));

	cudaSetDevice(0);
	cudaMalloc((void **) &D, 100000000 * sizeof(double));

	cudaSetDevice(1);
	cudaMalloc((void **) &A, 100000000 * sizeof(double));
	cudaMalloc((void **) &B, 100000000 * sizeof(double));

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaDeviceEnablePeerAccess(2, 0);
	cudaDeviceEnablePeerAccess(0, 0);

	cudaMemcpyPeerAsync(C, 2, A, 1, 100000000 * sizeof(double), stream1);
	cudaMemcpyPeerAsync(D, 0, B, 1, 100000000 * sizeof(double), stream2);

	for(int i = 0; i < 3; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
	return 0;
}

