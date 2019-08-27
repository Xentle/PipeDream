#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define alpha 0.001

__global__ void SetResult(double *result_d, int correct_result, int size);
__global__ void GetOutputLayerDelta(double *output_a, double *output_delta, double *result, int size);
__global__ void GetHiddenLayerDelta(double *cur_delta, double *cur_a, double *cur_weight, double *prev_delta, int size);
__global__ void UpdateWeight(double *cur_a, double *cur_weight, double *next_delta, int W_W, int size, double lr);
__global__ void Print(double *a, int size);
__global__ void Sigmoid(double *a, int size);
__global__ void Exponential(double *a, int size);
__global__ void Softmax(double *a, double sum, int size);
void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W);
void SetLayer(int layer_index);
void train_model(int epoch);
void test_accuracy();

struct layer_info {
	double *a;
	double *weight;
	double *delta;
};

struct layer_info *layer;
double *result_d, *input_host, *result_host, bias = 1.0, zero = 0.0;
int num_layer = 1, *num_node_arr;
clock_t start;
cublasHandle_t handle;

int main() {

	int epoch = 0;
	cublasCreate(&handle);

	// input model's information
	printf("number of layers : ");
	scanf(" %d", &num_layer);
	num_node_arr = (int *)malloc(sizeof(int) * (num_layer + 1));
	printf("number of nodes : ");
	for(int i = 1; i <= num_layer; i++)
		scanf(" %d", &num_node_arr[i]);
	for(int i = 1; i <= num_layer; i++)
		num_node_arr[i]++;

	
	// build model
	layer = (struct layer_info *)malloc((num_layer + 1) * sizeof(struct layer_info));
	for(int i = 1; i <= num_layer; i++)
		SetLayer(i);

	// allocate device_result and host memory
	input_host = (double *)malloc(num_node_arr[1] * sizeof(double));
	cudaMalloc((void**) &result_d, num_node_arr[num_layer] * sizeof(double));
	result_host = (double *)malloc(num_node_arr[num_layer] * sizeof(double));
	

	while (1)
	{
		printf("epoch : ");
		scanf(" %d", &epoch);
		train_model(epoch);
	}
	return 0;
}

void SetLayer(int layer_index)
{
	int cur_node = num_node_arr[layer_index], next_node;
	double *weight_host;

	// initialize a
	cudaMalloc((void**) &layer[layer_index].a, cur_node * sizeof(double));
	cudaMemcpy(&layer[layer_index].a[0], &bias, sizeof(double), cudaMemcpyHostToDevice);

	// initialize delta (except input layer)
	if(layer_index != 1)
		cudaMalloc((void**) &layer[layer_index].delta, cur_node * sizeof(double));

	// initialize weight (except output layer)
	if(layer_index != num_layer)
	{
		// initialize weight
		next_node = num_node_arr[layer_index + 1];
		weight_host = (double *)malloc(cur_node * next_node * sizeof(double));
		for (int i = 0; i < cur_node; i++)
			weight_host[i] = 0.0;
		for (int i = cur_node; i < cur_node * next_node; i++)
			weight_host[i] = sqrt(6.0 / (cur_node + next_node)) * (rand() / (double)RAND_MAX * 2.0 - 1.0);
		cudaMalloc((void**) &layer[layer_index].weight, cur_node * next_node * sizeof(double));
		cudaMemcpy(layer[layer_index].weight, weight_host, cur_node * next_node * sizeof(double), cudaMemcpyHostToDevice);
		free(weight_host);
	}
}

void train_model(int epoch) 
{
	FILE* pFile = NULL;
	char str_tmp[num_node_arr[1] * 3], *p;
	double sum;

	for (int e = 0; e < epoch; e++)
	{
		start = clock();
		pFile = fopen("mnist_train.csv", "r");
		if (pFile != NULL)
		{
			while (1)
			{	
				// load data from file
				fgets(str_tmp, num_node_arr[1] * 3, pFile);
				if (feof(pFile))
					break;
				p = strtok(str_tmp, ",");

				// set result
				if(atoi(p) == 0)
					SetResult<<<(num_node_arr[num_layer] + 1023) / 1024, 1024>>>(result_d, num_node_arr[num_layer] - 1, num_node_arr[num_layer]);
				else
					SetResult<<<(num_node_arr[num_layer] + 1023) / 1024, 1024>>>(result_d, atoi(p), num_node_arr[num_layer]);
				

				// set input
				p = strtok(NULL, ",");
				input_host[0] = 1.0;
				for (int i = 1; i < num_node_arr[1]; i++)
				{
					input_host[i] = atof(p) / 255.0;
					p = strtok(NULL, ",");
				}
				cudaMemcpy(layer[1].a, input_host, num_node_arr[1] * sizeof(double), cudaMemcpyHostToDevice);

				// forward pass
				for(int i = 1; i < num_layer - 1; i++)
				{
					MatrixMultiply(layer[i].weight, layer[i].a, layer[i+1].a, num_node_arr[i+1], num_node_arr[i], 1);
					Sigmoid<<<(num_node_arr[i+1] + 1023) / 1024, 1024>>>(layer[i+1].a, num_node_arr[i+1]);
				}
				MatrixMultiply(layer[num_layer - 1].weight, layer[num_layer - 1].a, layer[num_layer].a, num_node_arr[num_layer], num_node_arr[num_layer - 1], 1);
				Exponential<<<(num_node_arr[num_layer] + 1023) / 1024, 1024>>>(layer[num_layer].a, num_node_arr[num_layer]);
				cublasDasum(handle, num_node_arr[num_layer], layer[num_layer].a, 1, &sum);
				Softmax<<<(num_node_arr[num_layer] + 1023) / 1024, 1024>>>(layer[num_layer].a, sum, num_node_arr[num_layer]);

				//Print<<<1, 1>>>(layer[num_layer].a, num_node_arr[num_layer]);
				//cudaDeviceSynchronize();

				// Get output layer's delta
				GetOutputLayerDelta<<<(num_node_arr[num_layer] + 1023) / 1024, 1024>>>(layer[num_layer].a, layer[num_layer].delta, result_d, num_node_arr[num_layer]);

				// Get hidden layer's delta
				for(int i = num_layer - 1; i >= 1; i--)
				{
					if(i > 1)
					{
						MatrixMultiply(layer[i + 1].delta, layer[i].weight, layer[i].delta, 1, num_node_arr[i + 1], num_node_arr[i]);
						GetHiddenLayerDelta<<<(num_node_arr[i] + 1023) / 1024, 1024>>>(layer[i].delta, layer[i].a, layer[i].weight, layer[i + 1].delta, num_node_arr[i]);	
					}
					UpdateWeight<<<(num_node_arr[i] * num_node_arr[i + 1] + 1023) / 1024, 1024>>>(layer[i].a, layer[i].weight, layer[i+1].delta, num_node_arr[i], num_node_arr[i + 1] * num_node_arr[i], alpha);
				}
			}
		}

		printf("%fs\n", (double)(clock() - start)/CLOCKS_PER_SEC);

		test_accuracy();

		if (pFile != NULL)
			fclose(pFile);

	}
	return; 
}

void test_accuracy() 
{	
	double correct = 0, num_test_examples = 0, result_index = 0, sum;
	int max_index;
	FILE* pFile = NULL;
	char str_tmp[num_node_arr[1] * 3];
	char* p;

	pFile = fopen("mnist_test.csv", "r");

	if (pFile != NULL)
	{
		while (1)
		{
			// load data from file
			fgets(str_tmp, num_node_arr[1] * 3, pFile);
			if (feof(pFile))
				break;
			p = strtok(str_tmp, ",");

			// set result
			if(atoi(p) == 0)
				result_index = num_node_arr[num_layer] - 1;
			else
				result_index = atof(p);

			// set input
			p = strtok(NULL, ",");
			input_host[0] = 1.0;
			for (int i = 1; i < num_node_arr[1]; i++)
			{
				input_host[i] = atof(p) / 255.0;
				p = strtok(NULL, ",");
			}
			cudaMemcpy(layer[1].a, input_host, num_node_arr[1] * sizeof(double), cudaMemcpyHostToDevice);

			// forward pass
			for(int i = 1; i < num_layer - 1; i++)
			{
				MatrixMultiply(layer[i].weight, layer[i].a, layer[i+1].a, num_node_arr[i+1], num_node_arr[i], 1);
				Sigmoid<<<(num_node_arr[i+1] + 1023) / 1024, 1024>>>(layer[i+1].a, num_node_arr[i+1]);
			}
			MatrixMultiply(layer[num_layer - 1].weight, layer[num_layer - 1].a, layer[num_layer].a, num_node_arr[num_layer], num_node_arr[num_layer - 1], 1);
			Exponential<<<(num_node_arr[num_layer] + 1023) / 1024, 1024>>>(layer[num_layer].a, num_node_arr[num_layer]);
			cublasDasum(handle, num_node_arr[num_layer], layer[num_layer].a, 1, &sum);
			Softmax<<<(num_node_arr[num_layer] + 1023) / 1024, 1024>>>(layer[num_layer].a, sum, num_node_arr[num_layer]);

			//Print<<<1, 1>>>(layer[num_layer].a, num_node_arr[num_layer]);
			//cudaDeviceSynchronize();
			
			cublasIdamax(handle, num_node_arr[num_layer], layer[num_layer].a, 1, &max_index);

			if (result_index == --max_index)
				correct++;

			num_test_examples++;
		}
	}

	if(pFile != NULL)
		fclose(pFile);
	
	printf("%lf%%\n", correct / num_test_examples * 100);

	return;
}


__global__ void SetResult(double *result_d, int correct_result, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
	{
		if(i == correct_result)
			result_d[i] = 1.0;
		else
			result_d[i] = 0.0;
	}
}

void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W)
{
	const double alp = 1.0f;
	const double bet  = 0.0f;
		
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_W, A_H, A_W, &alp, d_B, B_W, d_A, A_W, &bet, d_C, B_W);
}

__global__ void GetOutputLayerDelta(double *output_a, double *output_delta, double *result, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		output_delta[i] = result[i] - output_a[i];
}

__global__ void GetHiddenLayerDelta(double *cur_delta, double *cur_a, double *cur_weight, double *prev_delta, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		cur_delta[i] = (1.0 - cur_a[i]) * cur_a[i] * (cur_delta[i] - cur_weight[i] * prev_delta[0]);
}


__global__ void UpdateWeight(double *cur_a, double *cur_weight, double *next_delta, int W_W, int size, double lr)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		cur_weight[i] = cur_weight[i] + lr * cur_a[i % W_W] * next_delta[i / W_W];
}

__global__ void Sigmoid(double *a, int size) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		a[i] = 1.0 / (1.0 + exp(-a[i]));
	if(i == 0)
		a[i] = 1.0;
}

__global__ void Exponential(double *a, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		a[i] = exp(a[i]);
	if(i == 0)
		a[i] = 0.0;
}

__global__ void Softmax(double *a, double sum, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		a[i] /= sum;
}

__global__ void Print(double *a, int size)
{
	for(int i=0; i<size; i++)
		printf("%lf\n", a[i]);
}



