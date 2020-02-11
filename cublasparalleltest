#include <stdio.h>
#include <cublas_v2.h>

#define NUMGPU 3

void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W, int i, cublasHandle_t h);
__global__ void Dummy();

int main() {

	cublasHandle_t *handle;
	cudaStream_t *stream;

	double **d_A = (double **)malloc(NUMGPU * sizeof(double *));
	double **d_B = (double **)malloc(NUMGPU * sizeof(double *));
	double **d_C = (double **)malloc(NUMGPU * sizeof(double *));
	handle = (cublasHandle_t *)malloc(NUMGPU * sizeof(cublasHandle_t));
	stream = (cudaStream_t *)malloc(NUMGPU * sizeof(cudaStream_t));

	for(int i = 0; i < NUMGPU; i++)
	{
		cudaSetDevice(i);
		cudaMalloc((void **) &d_A[i], 100 * sizeof(double));
		cudaMalloc((void **) &d_B[i], 100 * sizeof(double));
		cudaMalloc((void **) &d_C[i], 100 * sizeof(double));
		cublasCreate(&handle[i]);
		// cudaStreamCreate(&stream[i]);
		// cublasSetStream(handle[i], stream[i]);
	}

	cudaSetDevice(3);
	for(int i = 0; i < 100; i++)
		Dummy<<<1, 1>>>();
	
	for(int j = 0; j < 100000; j++)
	{
		for(int i = 0; i < NUMGPU; i++)
		{
			cudaSetDevice(i);
			MatrixMultiply(d_A[i], d_B[i], d_C[i], 100, 100, 100, i, handle[i]);
		}
			
	}

	for(int i = 0; i < NUMGPU + 1; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

	printf("finished\n");
}

__global__ void Dummy()
{
	int i = 0;
	for(int j = 0; j < 100000; j++)
		i++;
}

void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W, int i, cublasHandle_t h)
{
	const double alp = 1.0f;
	const double bet  = 0.0f;
		
	cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, B_W, A_H, A_W, &alp, d_B, B_W, d_A, A_W, &bet, d_C, B_W);
}
