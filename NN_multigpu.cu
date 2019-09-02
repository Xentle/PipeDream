#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define alpha 0.001

__global__ void SetResult(double *result_d, int correct_result, int size);
__global__ void GetOutputLayerDelta(double *output_a, double *output_delta, double result, int size);
__global__ void GetHiddenLayerDelta(double *cur_delta, double *cur_a, double *cur_weight, double *prev_delta, int size);
__global__ void UpdateWeight(double *cur_a, double *cur_weight, double *next_delta, int W_W, int size, double lr);
__global__ void Print(double *a, int size);
__global__ void Printint(int *a, int size);
__global__ void Sigmoid(double *a, int size);
__global__ void Exponential(double *a, int size);
__global__ void Softmax(double *a, double sum, int size);
__global__ void ForwardWait(int *ready, int device, int index);
__global__ void BackwardWait(int *ready, int device, int index);
__global__ void SetBackwardNotReady(int *ready, int index);
__global__ void SetForwardNotReady(int *ready, int index);
__global__ void PrintBw(int device, int index);
__global__ void PrintFw(int device, int index);
void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W, int i);
void SetLayer(int layer_index);
void train_model();
void test_accuracy();

struct layer_info {
	double **a;
	double **weight;
	double **delta;
	double **delta_next;
	double **a_next;
	int *one;
	int *fw_ready;
	int *bw_ready;
};

struct layer_info *layer;
cublasHandle_t *handle;
cudaStream_t *stream;

double *input_d, *input_host;
int *result_d, *result_host, *num_node_arr, *cur_fw, *cur_bw;

int num_layer = 1, num_data, epoch;

clock_t start;


int main() {

	// input model's information
	printf("number of layers : ");
	scanf(" %d", &num_layer);

	num_node_arr = (int *)malloc(sizeof(int) * num_layer);
	printf("number of nodes : ");
	for(int i = 0; i < num_layer; i++)
		scanf(" %d", &num_node_arr[i]);
	for(int i = 0; i < num_layer; i++)
		num_node_arr[i]++;

	printf("number of data : ");
	scanf(" %d", &num_data);
	
	// make tool
	handle = (cublasHandle_t *)malloc(num_layer * sizeof(cublasHandle_t));
	stream = (cudaStream_t *)malloc(num_layer * sizeof(cudaStream_t));

	// build model
	layer = (struct layer_info *)malloc(num_layer * sizeof(struct layer_info));
	for(int i = 0; i < num_layer; i++)
		SetLayer(i);
	cur_fw = (int *)malloc(num_layer * sizeof(int));
	cur_bw = (int *)malloc(num_layer * sizeof(int));
	for(int i = 0; i < num_layer; i++)
		cur_fw[i] = cur_bw[i] = 0;

	cudaDeviceSynchronize();

	// enable peer access
	for(int i = 0; i < num_layer - 1; i++)
	{
		cudaSetDevice(i);
		cudaDeviceEnablePeerAccess(i + 1, 0);
	}
	
	// train and test
	printf("epoch : ");
	scanf(" %d", &epoch);
	train_model();

	return 0;
}

void SetLayer(int layer_index)
{
	int cur_node = num_node_arr[layer_index], next_node, *one_h;
	double *weight_host;

	cudaSetDevice(layer_index);
	cudaStreamCreate(&stream[layer_index]);
	cublasCreate(&handle[layer_index]);
	cublasSetStream(handle[layer_index], stream[layer_index]);

	cudaMalloc((void**) &layer[layer_index].fw_ready, num_layer * sizeof(int));
	cudaMalloc((void**) &layer[layer_index].bw_ready, num_layer * sizeof(int));
	cudaMalloc((void**) &layer[layer_index].one, sizeof(int));
	one_h = (int *)malloc(sizeof(int));
	one_h[0] = 1;
	cudaMemcpy(layer[layer_index].one, one_h, sizeof(int), cudaMemcpyHostToDevice);


	layer[layer_index].a = (double **)malloc(num_layer * sizeof(double *));
	for(int i = 0; i < num_layer; i++)
		cudaMalloc((void**) &layer[layer_index].a[i], num_node_arr[layer_index] * sizeof(double));

	// except input layer
	if(layer_index != 0)
	{
		layer[layer_index].delta = (double **)malloc(num_layer * sizeof(double *));
		for(int i = 0; i < num_layer; i++)
			cudaMalloc((void**) &layer[layer_index].delta[i], num_node_arr[layer_index] * sizeof(double));
	}
	// except output layer
	if(layer_index < num_layer - 1)
	{
		next_node = num_node_arr[layer_index + 1];
		layer[layer_index].a_next = (double **)malloc(num_layer * sizeof(double *));
		layer[layer_index].delta_next = (double **)malloc(num_layer * sizeof(double *));
		for(int i = 0; i < num_layer; i++)
		{
			cudaMalloc((void**) &layer[layer_index].a_next[i], next_node * sizeof(double));
			cudaMalloc((void**) &layer[layer_index].delta_next[i], next_node * sizeof(double));
		}
		layer[layer_index].weight = (double **)malloc(num_layer * sizeof(double *));
		for(int i = 0; i < num_layer; i++)
		{
			weight_host = (double *)malloc(cur_node * next_node * sizeof(double));
			for (int j = 0; j < cur_node * next_node; j++)
				weight_host[j] = sqrt(6.0 / (cur_node + next_node)) * (rand() / (double)RAND_MAX * 2.0 - 1.0);
			cudaMalloc((void**) &layer[layer_index].weight[i], cur_node * next_node * sizeof(double));
			cudaMemcpy(layer[layer_index].weight[i], weight_host, cur_node * next_node * sizeof(double), cudaMemcpyHostToDevice);
			free(weight_host);
		}
	}
}


void train_model() 
{
	int result_index = 0, input_index = 0;
	FILE* pFile = NULL;
	char str_tmp[num_node_arr[0] * 3], *p;
	double sum;

	pFile = fopen("mnist_train.csv", "r");
	result_host = (int *)malloc(num_data * sizeof(int));
	input_host = (double *)malloc(num_data * num_node_arr[0] * sizeof(double));
	if(pFile != NULL)
    {   
		while(1)
		{
			fgets(str_tmp, num_node_arr[0] * 5, pFile);
			if (feof(pFile))
				break;
			
			// set result
			p = strtok(str_tmp, ",");
			if(atoi(p) == 0)
				result_host[result_index++] = num_node_arr[num_layer - 1] - 1;
			else
				result_host[result_index++] = atoi(p);

			// set input
			input_host[input_index++] = 1.0;
			for (int i = 1; i < num_node_arr[0]; i++)
			{
				p = strtok(NULL, ",");
				input_host[input_index++] = atof(p) / 255.0;
			}
        }       
	}
	cudaSetDevice(num_layer - 1);
	cudaMalloc((void**) &result_d, num_data * sizeof(int));
	cudaMemcpy(result_d, result_host, num_data * sizeof(int), cudaMemcpyHostToDevice);
	cudaSetDevice(0);
	cudaMalloc((void**) &input_d, num_data * num_node_arr[0] * sizeof(double));
	cudaMemcpy(input_d, input_host, num_data * num_node_arr[0] * sizeof(double), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	result_index = input_index = 0;

	// startup stage
	// input layer
	cudaSetDevice(0);
	for(int j = 0; j < num_layer; j++)
	{	
		cudaMemcpyAsync(layer[0].a[j], input_d + result_index++ * num_node_arr[0], num_node_arr[0] * sizeof(double), cudaMemcpyDeviceToDevice, stream[0]);
		MatrixMultiply(layer[0].a[j], layer[0].weight[j], layer[0].a_next[j], 1, num_node_arr[0], num_node_arr[1], 0);
		Sigmoid<<<(num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a_next[j], num_node_arr[1]);
		cudaMemcpyPeerAsync(layer[1].a[j], 1, layer[0].a_next[j], 0, num_node_arr[1] * sizeof(double), stream[0]);
		cudaMemcpyPeerAsync(layer[1].fw_ready + j, 1, layer[0].one , 0, sizeof(int), stream[0]);
		PrintFw<<<1, 1, 0, stream[0]>>>(0, j);
	} // Forward  X num_layer
	cudaDeviceSynchronize();
	BackwardWait<<<1, 1, 0, stream[0]>>>(layer[0].bw_ready, 0, 0);
	UpdateWeight<<<(num_node_arr[0] * num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a[0], layer[0].weight[0], layer[0].delta_next[0], num_node_arr[1], num_node_arr[1] * num_node_arr[0], alpha);
	PrintBw<<<1, 1, 0, stream[0]>>>(0, 0);
	SetBackwardNotReady<<<1, 1, 0, stream[0]>>>(layer[0].bw_ready, 0);

	cudaSetDevice(1);
	Printint<<<1, 1, 0, stream[1]>>>(layer[1].fw_ready, num_layer);
	cudaStreamSynchronize(stream[1]);

	cudaSetDevice(0);
	cudaStreamSynchronize(stream[0]);
	
	/*
	// hidden layer
	for(int i = 1; i < num_layer - 1; i++)
	{
		cudaSetDevice(i);
		for(int j = 0; j < num_layer; j++)
		{
			Printint<<<1, 1, 0, stream[i]>>>(layer[i].fw_ready, num_layer);
			ForwardWait<<<1, 1, 0, stream[i]>>>(layer[i].fw_ready, i, j);
			MatrixMultiply(layer[i].a[j], layer[i].weight[j], layer[i].a_next[j], 1, num_node_arr[i], num_node_arr[i + 1], i);
			if(i != num_layer - 2)
				Sigmoid<<<(num_node_arr[i + 1] + 1023) / 1024, 1024, 0, stream[i]>>>(layer[i].a_next[j], num_node_arr[i + 1]);
			
			cudaMemcpyPeerAsync(layer[i + 1].a[j], i + 1, layer[i].a_next[j], i, num_node_arr[i + 1] * sizeof(double), stream[i]);
			cudaMemcpyPeerAsync(layer[i + 1].fw_ready + j, i + 1, layer[i].one , i, sizeof(int), stream[i]);
			SetForwardNotReady<<<1, 1, 0, stream[i]>>>(layer[i].fw_ready, j);
			PrintFw<<<1, 1, 0, stream[i]>>>(i, j);
		}
		for(int j = 0; j < i + 1; j++)
		{
			BackwardWait<<<1, 1, 0, stream[i]>>>(layer[i].bw_ready, i, j);
			MatrixMultiply(layer[i].weight[j], layer[i].delta_next[j], layer[i].delta[j], num_node_arr[i], num_node_arr[i + 1], 1, i);
			GetHiddenLayerDelta<<<(num_node_arr[i] + 1023) / 1024, 1024>>>(layer[i].delta[j], layer[i].a[j], layer[i].weight[j], layer[i].delta_next[j], num_node_arr[i]);
			cudaMemcpyPeerAsync(layer[i - 1].delta_next[j], i - 1, layer[i].delta[j], i, num_node_arr[i] * sizeof(double), stream[i]);
			cudaMemcpyPeerAsync(layer[i - 1].bw_ready + j, i - 1, layer[i].one , i, sizeof(int), stream[0]);
			SetBackwardNotReady<<<1, 1, 0, stream[i]>>>(layer[i].bw_ready, j);
			PrintBw<<<1, 1, 0, stream[i]>>>(i, j);
		}
	}

	
	// output layer
	cudaSetDevice(num_layer - 1);
	for(int j = 0; j < num_layer; j++)
	{
		ForwardWait<<<1, 1, 0, stream[num_layer - 1]>>>(layer[num_layer - 1].fw_ready, num_layer - 1, j);
		Exponential<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024>>>(layer[num_layer - 1].a[j], num_node_arr[num_layer - 1]);
		cublasDasum(handle[num_layer - 1], num_node_arr[num_layer - 1], layer[num_layer - 1].a[j], 1, &sum);
		Softmax<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024>>>(layer[num_layer - 1].a[j], sum, num_node_arr[num_layer - 1]);
		PrintFw<<<1, 1, 0, stream[num_layer - 1]>>>(num_layer - 1, j);
		PrintBw<<<1, 1, 0, stream[num_layer - 1]>>>(num_layer - 1, j);
		GetOutputLayerDelta<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024>>>(layer[num_layer - 1].a[j], layer[num_layer - 1].delta[j], result_d[j], num_node_arr[num_layer - 1]);
		cudaMemcpyPeerAsync(layer[num_layer - 2].delta_next[j], num_layer - 2, layer[num_layer - 1].delta[j], num_layer - 1, num_node_arr[num_layer - 1] * sizeof(double), stream[num_layer - 1]);
		SetForwardNotReady<<<1, 1, 0, stream[num_layer - 1]>>>(layer[num_layer - 1].fw_ready, j);
	}

	for(int i=0; i< num_layer; i++)
		cudaStreamSynchronize(stream[i]);
		*/

	/*
	// steady stage
	// input layer
	Setinput
	Forward
	BackwardWait
	Backward
	SetBackwardNotReady

	// hidden layer
	for(int i = 1; i < num_layer - 2; i++)
	{
		cudaSetDevice(i);
		ForwardWait<<<1, 1, stream[i]>>>(forward_ready[i], cur_fw[i]);
		Forward
		SetForwardNotReady
		BackwardWait
		Backward
		SetBackwardNotReady
	}

	// output layer
	ForwardWait<<<1, 1, stream[i]>>>(forward_ready[i], cur_fw[i]);
	Forward
	SetForwardNotReady
	GetOutputdelta
	Backward
	*/

	return; 
}

__global__ void ForwardWait(int *ready, int device, int index)
{
	while(ready[index] == 0) {
		//printf("device : %d , fw: %d\n", device, index);
	}
}

__global__ void BackwardWait(int *ready, int device, int index)
{
	while(ready[index] == 0) {
		//printf("device : %d , bw: %d\n", device, index);
	}
}

__global__ void SetForwardNotReady(int *ready, int index)
{
	ready[index] = 0;
}

__global__ void SetBackwardNotReady(int *ready, int index)
{
	ready[index] = 0;
}

/*
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
}*/

void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W, int i)
{
	const double alp = 1.0f;
	const double bet  = 0.0f;
		
	cublasDgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, B_W, A_H, A_W, &alp, d_B, B_W, d_A, A_W, &bet, d_C, B_W);
}


__global__ void GetOutputLayerDelta(double *output_a, double *output_delta, double result, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
	{
		if(i != result)
			output_delta[i] = 0.0 - output_a[i];
		else
			output_delta[i] = 1.0 - output_a[i];
	}
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
		cur_weight[i] = cur_weight[i] + lr * cur_a[i / W_W] * next_delta[i % W_W];
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
		printf(" %lf /", a[i]);
	printf("\n");
}

__global__ void Printint(int *a, int size)
{
	for(int i=0; i<size; i++)
		printf(" %d /", a[i]);
	printf("\n");
}

__global__ void PrintFw(int device, int index)
{
	printf("device : %d , fw : %d\n", device, index);
}

__global__ void PrintBw(int device, int index)
{
	printf("device : %d , fw : %d\n", device, index);
}


