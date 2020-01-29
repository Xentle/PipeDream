#include <stdio.h>

#include <cuda_runtime.h>

#define MEMSIZE 100000000

int *D_OA, *D_OB, *D_UA, *D_UB, *H, *Temp;

// utilities
cudaEvent_t start;
cudaEvent_t stop;
float msecTotal;
int error;

__global__ void Unifiedmemcpy(int *d_a, int *d_b)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < MEMSIZE)
		d_b[idx] = d_a[idx];
}

__global__ void MemcpyKernel(int *d_a, int *d_b)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < MEMSIZE)
		cudaMemcpyAsync(d_b + idx, d_a + idx, sizeof(int), cudaMemcpyDeviceToDevice);
	__syncthreads();
}

__global__ void Print(int *d)
{
	printf("D_OA\n");
	printf("%d", d[0]);
	for(int i = 1; i < 10; i++)
		printf(" / %d", d[i]);
	printf(" ... \n");
	printf("%d", d[9990]);
	for(int i = 9991; i < MEMSIZE; i++)
		printf(" / %d", d[i]);
	printf("\n");
}

int main () {

	printf("Array Size = %d\n", MEMSIZE);
	if(MEMSIZE * 4 >= 1000000000)
		printf("Memory Size = %.1f GB\n", float(MEMSIZE * 4) / 1000000000.0f);
	else if(MEMSIZE * 4 >= 1000000)
		printf("Memory Size = %.1f MB\n", float(MEMSIZE * 4) / 1000000.0f);
	else if(MEMSIZE * 4 >= 1000)
		printf("Memory Size = %.1f KB\n", float(MEMSIZE * 4) / 1000.0f);
	else
		printf("Memory Size = %d B\n", MEMSIZE * 4);
	printf("\n");
	
	// printf("---cudaDeviceEnablePeerAccess---\n\n");
	

	cudaSetDevice(1);
	cudaMallocManaged((void**) &D_UB, MEMSIZE * sizeof(int));
	cudaMalloc((void**) &D_OB, MEMSIZE * sizeof(int));

	cudaSetDevice(0);
	cudaMallocManaged((void**) &D_UA, MEMSIZE * sizeof(int));
	cudaMalloc((void**) &D_OA, MEMSIZE * sizeof(int));
	// cudaDeviceEnablePeerAccess(1, 0);

	H = (int *) malloc(MEMSIZE * sizeof(int));
	Temp = (int *) malloc(MEMSIZE * sizeof(int));
	

	////////////////////////////////////////////////////////////
	// Unified Memory B[i] = A[i] //////////////////////////////
	////////////////////////////////////////////////////////////

	// Set Value
	for(int i = 0; i < MEMSIZE; i++)
		D_UA[i] = i;

	// printf("D_UA\n");
	// printf("%d", D_UA[0]);
	// for(int i = 1; i < 10; i++)
	// 	printf(" / %d", D_UA[i]);
	// printf(" ... \n");
	// printf("%d", D_UA[9990]);
	// for(int i = MEMSIZE - 9; i < MEMSIZE; i++)
	// 	printf(" / %d", D_UA[i]);
	// printf("\n");

	// create and start timer
    cudaEventCreate(&start);
	cudaEventRecord(start, NULL); 

	// Memcpy
	Unifiedmemcpy<<<(MEMSIZE + 1023) / 1024, 1024>>>(D_UA, D_UB);

    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Unified memory B[i] = A[i]\n");
	printf("Processing time: %f (ms)\n", msecTotal);

	// print error
	for(int i = 0 ; i < MEMSIZE; i++)
		if(D_UA[i] - D_UB[i] > 0.1)
			error++;
	printf("error : %d\n\n", error);
	error = msecTotal = 0;

	////////////////////////////////////////////////////////////
	// Unified Memory CudaMemcpyAsync in Kernel ////////////////
	////////////////////////////////////////////////////////////
	
	// Set Value
	for(int i = 0 ; i < MEMSIZE; i++)
		D_UA[i] = MEMSIZE - i;

	// printf("D_UA\n");
	// printf("%d", D_UA[0]);
	// for(int i = 1; i < 10; i++)
	// 	printf(" / %d", D_UA[i]);
	// printf(" ... \n");
	// printf("%d", D_UA[9990]);
	// for(int i = MEMSIZE - 10; i < MEMSIZE; i++)
	// 	printf(" / %d", D_UA[i]);
	// printf("\n");
	
	// create and start timer
    cudaEventCreate(&start);
	cudaEventRecord(start, NULL); 

	// Memcpy
	// MemcpyKernel<<<(MEMSIZE + 1023) / 1024, 1024>>>(D_UA, D_UB);

    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    // printf("Unified memory cudaMemcpyAsync in Kernel\n");
	// printf("Processing time: %f (ms)\n", msecTotal);

	for(int i = 0 ; i < MEMSIZE; i++)
		if(D_UA[i] - D_UB[i] > 0.1)
			error++;
	// printf("error : %d\n\n", error);
	error = msecTotal = 0;

	////////////////////////////////////////////////////////////
	// Original Memory CudaMemcpy GPU0 -> CPU -> GPU1 //////////
	////////////////////////////////////////////////////////////
	
	// Set Value
	for(int i = 0 ; i < MEMSIZE; i++)
		H[i] = i;

	cudaMemcpy(D_OA, H, MEMSIZE * sizeof(int), cudaMemcpyHostToDevice);

	for(int i = 0; i < MEMSIZE; i++)
		H[i] = 0;

	// Print<<<1, 1>>>(D_OA);
	// cudaDeviceSynchronize();
	
	// create and start timer
    cudaEventCreate(&start);
	cudaEventRecord(start, NULL);

	// Memcpy
	cudaMemcpy(H, D_OA, MEMSIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(D_OB, H, MEMSIZE * sizeof(int), cudaMemcpyHostToDevice);

    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
	printf("Original memory cudaMemcpy GPU0 -> CPU -> GPU1\n");
	printf("Processing time: %f (ms)\n", msecTotal);


	cudaMemcpy(Temp, D_OB, MEMSIZE * sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 0 ; i < MEMSIZE; i++)
		if(H[i] - Temp[i] > 0.1)
			error++;
	printf("error : %d\n\n", error);
	error = msecTotal = 0;

	////////////////////////////////////////////////////////////
	// Original Memory CudaMemcpyPeer //////////////////////////
	////////////////////////////////////////////////////////////
	
	// Set Value
	for(int i = 0 ; i < MEMSIZE; i++)
		H[i] = MEMSIZE - i;

	cudaMemcpy(D_OA, H, MEMSIZE * sizeof(int), cudaMemcpyHostToDevice);

	// Print<<<1, 1>>>(D_OA);
	// cudaDeviceSynchronize();
	
	// create and start timer
    cudaEventCreate(&start);
	cudaEventRecord(start, NULL);

	// Memcpy
	cudaMemcpyPeer(D_OB, 1, D_OA, 0, MEMSIZE * sizeof(int));

    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
	printf("Original Memory cudaMemcpyPeer\n");
	printf("Processing time: %f (ms)\n", msecTotal);

	cudaMemcpy(Temp, D_OB, MEMSIZE * sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 0 ; i < MEMSIZE; i++)
		if(H[i] - Temp[i] > 0.1)
			error++;
	printf("error : %d\n\n", error);
	error = msecTotal = 0;
}

