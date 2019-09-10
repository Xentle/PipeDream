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
__global__ void Printint(int *a, int device, int size);
__global__ void Sigmoid(double *a, int size);
__global__ void Exponential(double *a, int size);
__global__ void Softmax(double *a, double sum, int size);
__global__ void PrintBw(int device, int index);
__global__ void PrintFw(int device, int index);
__global__ void WaituntilZero(int *ready, int index, int device, int fb);
__global__ void WaituntilOne(int *ready, int index, int device, int fb);
__global__ void dummy();
void* StartupStageOutputLayer(void *arg);
void* StartupStageHiddenLayer(void* index);
void* StartupStageInputLayer(void *arg);
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
	int *index_arr;
	int *fw_ready;
	int *bw_ready;
};

struct layer_info *layer;
cublasHandle_t *handle;
cudaStream_t *stream;
pthread_t *thread;

double *input_d, *input_host, *result_d, *result_host;
int *num_node_arr, *cur_fw, *cur_bw, *index_arr_h;

int num_layer = 1, num_data, epoch, thr_id;

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
	thread = (pthread_t *)malloc(num_layer * sizeof(pthread_t));
	index_arr_h = (int *)malloc(num_layer * sizeof(int));
	for(int i = 0; i < num_layer; i++)
		index_arr_h[i] = i;

	// build model
	layer = (struct layer_info *)malloc(num_layer * sizeof(struct layer_info));
	for(int i = 0; i < num_layer; i++)
		SetLayer(i);
	cur_fw = (int *)malloc(num_layer * sizeof(int));
	cur_bw = (int *)malloc(num_layer * sizeof(int));

	cudaDeviceSynchronize();

	// enable peer access
	for(int i = 0; i < num_layer - 1; i++)
	{
		cudaSetDevice(i);
		cudaDeviceEnablePeerAccess(i + 1, 0);
	}
	for(int i = num_layer - 1; i > 0; i--)
	{
		cudaSetDevice(i);
		cudaDeviceEnablePeerAccess(i - 1, 0);
	}
	
	// train and test
	printf("epoch : ");
	scanf(" %d", &epoch);
	train_model();

	return 0;
}

void SetLayer(int layer_index)
{
	int cur_node = num_node_arr[layer_index], next_node;
	double *weight_host;

	cudaSetDevice(layer_index);
	cudaStreamCreate(&stream[layer_index]);
	cublasCreate(&handle[layer_index]);
	cublasSetStream(handle[layer_index], stream[layer_index]);

	cudaMalloc((void**) &layer[layer_index].fw_ready, num_layer * sizeof(int));
	cudaMalloc((void**) &layer[layer_index].bw_ready, num_layer * sizeof(int));

	for(int i = 0; i < num_layer; i++) {
		cudaMemcpy(layer[layer_index].fw_ready + i, index_arr_h, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(layer[layer_index].bw_ready + i, index_arr_h, sizeof(int), cudaMemcpyHostToDevice);
	}

	cudaMalloc((void**) &layer[layer_index].index_arr, num_layer * sizeof(int));
	cudaMemcpy(layer[layer_index].index_arr, index_arr_h, num_layer * sizeof(int), cudaMemcpyHostToDevice);

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
			cudaDeviceSynchronize();
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
	// cudaStream_t dum;

	pFile = fopen("mnist_train.csv", "r");
	result_host = (double *)malloc(num_data * sizeof(double));
	input_host = (double *)malloc(num_data * num_node_arr[0] * sizeof(double));
	if(pFile != NULL)
    {   
		for(int r_index = 0, i_index = 0; ;)
		{
			fgets(str_tmp, num_node_arr[0] * 5, pFile);
			if (feof(pFile))
				break;
			
			// set result
			p = strtok(str_tmp, ",");
			if(atoi(p) == 0)
				result_host[r_index++] = num_node_arr[num_layer - 1] - 1;
			else
				result_host[r_index++] = atoi(p);

			// set input
			input_host[i_index++] = 1.0;
			for (int i = 1; i < num_node_arr[0]; i++)
			{
				p = strtok(NULL, ",");
				input_host[i_index++] = atof(p) / 255.0;
			}
        }       
	}
	cudaSetDevice(num_layer - 1);
	cudaMalloc((void**) &result_d, num_data * sizeof(int));
	cudaMemcpy(result_d, result_host, num_data * sizeof(int), cudaMemcpyHostToDevice);

	cudaSetDevice(0);
	cudaMalloc((void**) &input_d, num_data * num_node_arr[0] * sizeof(double));
	cudaMemcpy(input_d, input_host, num_data * num_node_arr[0] * sizeof(double), cudaMemcpyHostToDevice);

	result_index = input_index = num_layer;
	for(int i = 0; i < num_layer; i++)
		cur_fw[i] = 0;
	for(int i = 0; i < num_layer - 1; i++)
		cur_bw[i] = i + 1;
	cur_bw[num_layer - 1] = 0;

	// startup stage
	// input layer
	cudaSetDevice(0);
	for(int j = 0; j < num_layer; j++)
	{	
		cudaMemcpyAsync(layer[0].a[j], input_d + j * num_node_arr[0], num_node_arr[0] * sizeof(double), cudaMemcpyDeviceToDevice, stream[0]);
		MatrixMultiply(layer[0].a[j], layer[0].weight[j], layer[0].a_next[j], 1, num_node_arr[0], num_node_arr[1], 0);
		Sigmoid<<<(num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a_next[j], num_node_arr[1]);

		WaituntilZero<<<1, 1, 0, stream[0]>>>(layer[1].fw_ready, j, 1, 1);

		cudaMemcpyPeerAsync(layer[1].a[j], 1, layer[0].a_next[j], 0, num_node_arr[1] * sizeof(double), stream[0]);
		cudaMemcpyPeerAsync(layer[1].fw_ready + j, 1, layer[0].index_arr + 1, 0, sizeof(int), stream[0]);

		PrintFw<<<1, 1, 0, stream[0]>>>(0, j);
	} // Forward  X num_layer
	WaituntilOne<<<1, 1, 0, stream[0]>>>(layer[0].bw_ready, 0, 0, 0);

	UpdateWeight<<<(num_node_arr[0] * num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a[0], layer[0].weight[0], layer[0].delta_next[0], num_node_arr[1], num_node_arr[1] * num_node_arr[0], alpha);
	cudaMemcpyAsync(layer[0].bw_ready, layer[0].index_arr, sizeof(int), cudaMemcpyDeviceToDevice, stream[0]);

	PrintBw<<<1, 1, 0, stream[0]>>>(0, 0);

	// hidden layer
	for(int i = 1; i < num_layer - 1; i++)
	{
		cudaSetDevice(i);
		for(int j = 0; j < num_layer; j++)
		{
			WaituntilOne<<<1, 1, 0, stream[i]>>>(layer[i].fw_ready, j, i, 1);

			MatrixMultiply(layer[i].a[j], layer[i].weight[j], layer[i].a_next[j], 1, num_node_arr[i], num_node_arr[i + 1], i);
			if(i != num_layer - 2)
				Sigmoid<<<(num_node_arr[i + 1] + 1023) / 1024, 1024, 0, stream[i]>>>(layer[i].a_next[j], num_node_arr[i + 1]);
			
			WaituntilZero<<<1, 1, 0, stream[i]>>>(layer[i + 1].fw_ready, j, i + 1, 1);

			cudaMemcpyPeerAsync(layer[i + 1].a[j], i + 1, layer[i].a_next[j], i, num_node_arr[i + 1] * sizeof(double), stream[i]);
			cudaMemcpyAsync(layer[i].fw_ready + j, layer[i].index_arr, sizeof(int), cudaMemcpyDeviceToDevice, stream[i]);
			cudaMemcpyPeerAsync(layer[i + 1].fw_ready + j, i + 1, layer[i].index_arr + 1, i, sizeof(int), stream[i]);

			PrintFw<<<1, 1, 0, stream[i]>>>(i, j);
		}

		for(int j = 0; j < i + 1; j++)
		{
			WaituntilOne<<<1, 1, 0, stream[i]>>>(layer[i].bw_ready, j, i, 0);

			MatrixMultiply(layer[i].weight[j], layer[i].delta_next[j], layer[i].delta[j], num_node_arr[i], num_node_arr[i + 1], 1, i);
			GetHiddenLayerDelta<<<(num_node_arr[i] + 1023) / 1024, 1024, 0, stream[i]>>>(layer[i].delta[j], layer[i].a[j], layer[i].weight[j], layer[i].delta_next[j], num_node_arr[i]);
			WaituntilZero<<<1, 1, 0, stream[i]>>>(layer[i - 1].bw_ready, j, i - 1, 0);
			
			cudaMemcpyPeerAsync(layer[i - 1].delta_next[j], i - 1, layer[i].delta[j], i, num_node_arr[i - 1] * sizeof(double), stream[i]);
			cudaMemcpyAsync(layer[i].bw_ready + j, layer[i].index_arr, sizeof(int), cudaMemcpyDeviceToDevice, stream[i]);
			cudaMemcpyPeerAsync(layer[i - 1].bw_ready + j, i - 1, layer[i].index_arr + 1, i, sizeof(int), stream[i]);

			PrintBw<<<1, 1, 0, stream[i]>>>(i, j);
		}	
	}

	// output layer
	cudaSetDevice(num_layer - 1);
	for(int j = 0; j < num_layer; j++)
	{
		WaituntilOne<<<1, 1, 0, stream[num_layer - 1]>>>(layer[num_layer - 1].fw_ready, j, num_layer - 1, 1);

		Exponential<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[num_layer - 1]>>>(layer[num_layer - 1].a[j], num_node_arr[num_layer - 1]);
		cublasDasum(handle[num_layer - 1], num_node_arr[num_layer - 1], layer[num_layer - 1].a[j], 1, &sum);
		Softmax<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[num_layer - 1]>>>(layer[num_layer - 1].a[j], sum, num_node_arr[num_layer - 1]);

		PrintFw<<<1, 1, 0, stream[num_layer - 1]>>>(num_layer - 1, j);

		GetOutputLayerDelta<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[num_layer - 1]>>>(layer[num_layer - 1].a[j], layer[num_layer - 1].delta[j], result_d + j, num_node_arr[num_layer - 1]);

		WaituntilZero<<<1, 1, 0, stream[num_layer - 1]>>>(layer[num_layer - 2].bw_ready, j, num_layer - 2, 0);

		PrintBw<<<1, 1, 0, stream[num_layer - 1]>>>(num_layer - 1, j);

		cudaMemcpyPeerAsync(layer[num_layer - 2].delta_next[j], num_layer - 2, layer[num_layer - 1].delta[j], num_layer - 1, num_node_arr[num_layer - 1] * sizeof(double), stream[num_layer - 1]);
		cudaMemcpyAsync(layer[num_layer - 1].fw_ready + j, layer[num_layer - 1].index_arr, sizeof(int), cudaMemcpyDeviceToDevice, stream[num_layer - 1]);
		cudaMemcpyPeerAsync(layer[num_layer - 2].bw_ready + j, num_layer - 2, layer[num_layer - 1].index_arr + 1, num_layer - 1, sizeof(int), stream[num_layer - 1]);
	}

	for(int i = 0; i < num_layer; i++)
	{	
		cudaSetDevice(i);
		cudaStreamSynchronize(stream[i]);
	}

	printf("\nStart Steady stage\n");

	// steady stage
	while(1)
	{
		// printf("\ninput index : %d, result index : %d\n", input_index, result_index);
		if(input_index == num_data)
			break;

		// input layer
		cudaSetDevice(0);
		cudaMemcpyAsync(layer[0].a[cur_fw[0]], input_d + input_index++ * num_node_arr[0], num_node_arr[0] * sizeof(double), cudaMemcpyDeviceToDevice, stream[0]);
		MatrixMultiply(layer[0].a[cur_fw[0]], layer[0].weight[cur_fw[0]], layer[0].a_next[cur_fw[0]], 1, num_node_arr[0], num_node_arr[1], 0);
		Sigmoid<<<(num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a_next[cur_fw[0]], num_node_arr[1]);

		WaituntilZero<<<1, 1, 0, stream[0]>>>(layer[1].fw_ready, cur_fw[0], 1, 1);

		cudaMemcpyPeerAsync(layer[1].a[cur_fw[0]], 1, layer[0].a_next[cur_fw[0]], 0, num_node_arr[1] * sizeof(double), stream[0]);
		cudaMemcpyPeerAsync(layer[1].fw_ready + cur_fw[0], 1, layer[0].index_arr + 1, 0, sizeof(int), stream[0]);

		PrintFw<<<1, 1, 0, stream[0]>>>(0, cur_fw[0]);

		if(cur_fw[0] == num_layer - 1)
			cudaMemcpyAsync(cur_fw, layer[0].index_arr, sizeof(int), cudaMemcpyDeviceToHost, stream[0]);
		else
			cudaMemcpyAsync(cur_fw, layer[0].index_arr + cur_fw[0] + 1, sizeof(int), cudaMemcpyDeviceToHost, stream[0]);

		WaituntilOne<<<1, 1, 0, stream[0]>>>(layer[0].bw_ready, cur_bw[0], 0, 0);

		UpdateWeight<<<(num_node_arr[0] * num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a[cur_bw[0]], layer[0].weight[cur_bw[0]], layer[0].delta_next[cur_bw[0]], num_node_arr[1], num_node_arr[1] * num_node_arr[0], alpha);
		cudaMemcpyAsync(layer[0].bw_ready + cur_bw[0], layer[0].index_arr, sizeof(int), cudaMemcpyDeviceToDevice, stream[0]);

		PrintBw<<<1, 1, 0, stream[0]>>>(0, cur_bw[0]);

		if(cur_bw[0] == num_layer - 1)
			cudaMemcpyAsync(cur_bw, layer[0].index_arr, sizeof(int), cudaMemcpyDeviceToHost, stream[0]);
		else
			cudaMemcpyAsync(cur_bw, layer[0].index_arr + cur_bw[0] + 1, sizeof(int), cudaMemcpyDeviceToHost, stream[0]);

		// hidden layer
		for(int i = 1; i < num_layer - 1; i++)
		{
			cudaSetDevice(i);
			WaituntilOne<<<1, 1, 0, stream[i]>>>(layer[i].fw_ready, cur_fw[i], i, 1);

			MatrixMultiply(layer[i].a[cur_fw[i]], layer[i].weight[cur_fw[i]], layer[i].a_next[cur_fw[i]], 1, num_node_arr[i], num_node_arr[i + 1], i);
			if(i != num_layer - 2)
				Sigmoid<<<(num_node_arr[i + 1] + 1023) / 1024, 1024, 0, stream[i]>>>(layer[i].a_next[cur_fw[i]], num_node_arr[i + 1]);
			
			WaituntilZero<<<1, 1, 0, stream[i]>>>(layer[i + 1].fw_ready, cur_fw[i], i + 1, 1);

			cudaMemcpyPeerAsync(layer[i + 1].a[cur_fw[i]], i + 1, layer[i].a_next[cur_fw[i]], i, num_node_arr[i + 1] * sizeof(double), stream[i]);
			cudaMemcpyAsync(layer[i].fw_ready + cur_fw[i], layer[i].index_arr, sizeof(int), cudaMemcpyDeviceToDevice, stream[i]);
			cudaMemcpyPeerAsync(layer[i + 1].fw_ready + cur_fw[i], i + 1, layer[i].index_arr + 1, i, sizeof(int), stream[i]);

			PrintFw<<<1, 1, 0, stream[i]>>>(i, cur_fw[i]);

			if(cur_fw[i] == num_layer - 1)
				cudaMemcpyAsync(cur_fw + i, layer[i].index_arr, sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
			else
				cudaMemcpyAsync(cur_fw + i, layer[i].index_arr + cur_fw[i] + 1, sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
		
			WaituntilOne<<<1, 1, 0, stream[i]>>>(layer[i].bw_ready, cur_bw[i], i, 0);

			MatrixMultiply(layer[i].weight[cur_bw[i]], layer[i].delta_next[cur_bw[i]], layer[i].delta[cur_bw[i]], num_node_arr[i], num_node_arr[i + 1], 1, i);
			GetHiddenLayerDelta<<<(num_node_arr[i] + 1023) / 1024, 1024, 0, stream[i]>>>(layer[i].delta[cur_bw[i]], layer[i].a[cur_bw[i]], layer[i].weight[cur_bw[i]], layer[i].delta_next[cur_bw[i]], num_node_arr[i]);
			WaituntilZero<<<1, 1, 0, stream[i]>>>(layer[i - 1].bw_ready, cur_bw[i], i - 1, 0);
			
			cudaMemcpyPeerAsync(layer[i - 1].delta_next[cur_bw[i]], i - 1, layer[i].delta[cur_bw[i]], i, num_node_arr[i - 1] * sizeof(double), stream[i]);
			cudaMemcpyAsync(layer[i].bw_ready + cur_bw[i], layer[i].index_arr, sizeof(int), cudaMemcpyDeviceToDevice, stream[i]);
			cudaMemcpyPeerAsync(layer[i - 1].bw_ready + cur_bw[i], i - 1, layer[i].index_arr + 1, i, sizeof(int), stream[i]);

			PrintBw<<<1, 1, 0, stream[i]>>>(i, cur_bw[i]);

			if(cur_bw[i] == num_layer - 1)
				cudaMemcpyAsync(cur_bw + i, layer[i].index_arr, sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
			else
				cudaMemcpyAsync(cur_bw + i, layer[i].index_arr + cur_bw[i] + 1, sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
		}

		// output layer
		cudaSetDevice(num_layer - 1);

		WaituntilOne<<<1, 1, 0, stream[num_layer - 1]>>>(layer[num_layer - 1].fw_ready, cur_fw[num_layer - 1], num_layer - 1, 1);

		Exponential<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[num_layer - 1]>>>(layer[num_layer - 1].a[cur_fw[num_layer - 1]], num_node_arr[num_layer - 1]);
		cublasDasum(handle[num_layer - 1], num_node_arr[num_layer - 1], layer[num_layer - 1].a[cur_fw[num_layer - 1]], 1, &sum);
		Softmax<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[num_layer - 1]>>>(layer[num_layer - 1].a[cur_fw[num_layer - 1]], sum, num_node_arr[num_layer - 1]);

		PrintFw<<<1, 1, 0, stream[num_layer - 1]>>>(num_layer - 1, cur_fw[num_layer - 1]);

		GetOutputLayerDelta<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[num_layer - 1]>>>(layer[num_layer - 1].a[cur_fw[num_layer - 1]], layer[num_layer - 1].delta[cur_bw[num_layer - 1]], result_d + result_index++, num_node_arr[num_layer - 1]);

		WaituntilZero<<<1, 1, 0, stream[num_layer - 1]>>>(layer[num_layer - 2].bw_ready, cur_fw[num_layer - 1], num_layer - 2, 0);

		cudaMemcpyPeerAsync(layer[num_layer - 2].delta_next[cur_fw[num_layer - 1]], num_layer - 2, layer[num_layer - 1].delta[cur_fw[num_layer - 1]], num_layer - 1, num_node_arr[num_layer - 1] * sizeof(double), stream[num_layer - 1]);
		cudaMemcpyAsync(layer[num_layer - 1].fw_ready + cur_fw[num_layer - 1], layer[num_layer - 1].index_arr, sizeof(int), cudaMemcpyDeviceToDevice, stream[num_layer - 1]);
		cudaMemcpyPeerAsync(layer[num_layer - 2].bw_ready + cur_fw[num_layer - 1], num_layer - 2, layer[num_layer - 1].index_arr + 1, num_layer - 1, sizeof(int), stream[num_layer - 1]);

		PrintBw<<<1, 1, 0, stream[num_layer - 1]>>>(num_layer - 1, cur_fw[num_layer - 1]);

		if(cur_fw[num_layer - 1] == num_layer - 1)
			cudaMemcpyAsync(cur_fw + num_layer - 1, layer[num_layer - 1].index_arr, sizeof(int), cudaMemcpyDeviceToHost, stream[num_layer - 1]);
		else				
			cudaMemcpyAsync(cur_fw + num_layer - 1, layer[num_layer - 1].index_arr + cur_fw[num_layer - 1] + 1, sizeof(int), cudaMemcpyDeviceToHost, stream[num_layer - 1]);
	}

	
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

__global__ void GetOutputLayerDelta(double *output_a, double *output_delta, double *result, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
	{
		if(i != result[0])
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

__global__ void Printint(int *a, int device, int size)
{
	printf("device : %d , ", device);
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
	printf("device : %d , bw : %d\n", device, index);
}

__global__ void WaituntilZero(int *ready, int index, int device, int fb)
{
	while(ready[index] == 1) {
		// for(int i = 0; i < 100; i++)
		// { }
		printf("device : %d, ", device);
		if(fb == 1)
			printf("direction : fw, ");
		else
			printf("direction : bw, ");
		printf("%d\n", ready[index]);
	}
}

__global__ void WaituntilOne(int *ready, int index, int device, int fb)
{
	
	while(ready[index] == 0) { 
		// for(int i = 0; i < 10000000; i++)
		// { }
		printf("device : %d, ", device);
		if(fb == 1)
			printf("direction : fw, ");
		else
			printf("direction : bw, ");
		printf("%d\n", ready[index]);
	}
}

__global__ void dummy()
{
	while(1) { }
}