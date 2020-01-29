#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sample(int *A)
{
	__shared__ int i;
	i = 0;
	if(threadIdx.x == 0)
	{
		for(int j = 0; j < 10000000; j++);
		A[i] = 1;
		atomicAdd(&i, 1);

		__syncthreads();
		
		for(int j = 0; j < 1000000; j++);
		A[i] = 2;
		atomicAdd(&i, 1);
	}
	else
	{
		A[i] = 3;
		atomicAdd(&i, 1);

		__syncthreads();

		A[i] = 4;
		atomicAdd(&i, 1);
	}
}

int main()
{
	int *D_A;
	cudaMallocManaged((void**) &D_A, 4 * sizeof(int));
	sample<<<1, 2>>>(D_A);
	cudaDeviceSynchronize();
	for(int i = 0; i < 4; i++)
		printf("%d , ", D_A[i]);
	printf("\n");
}


