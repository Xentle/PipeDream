#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cublas_v2.h>

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
__global__ void WaituntilZero(int *ready, int index);
__global__ void WaituntilOne(int *ready, int index);
__global__ void SetFlag(int *ready, int index);
__global__ void dummy();
void* StartupStageOutputLayer(void *arg);
void* StartupStageHiddenLayer(void* index);
void* StartupStageInputLayer(void *arg);
void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W, int i);
void SetLayer(int layer_index);
void train_model();
void test_accuracy();

struct layer_info {
	double *weight;
	double **a;
	double **a_next;
	double **delta;
	double **delta_next;
	int *is_fw_input_ready;
	int *is_bw_input_ready;
	int *is_fw_act_ready;
	int *is_bw_delta_ready;
	int *cur_fw_cm;
	int *cur_bw_cm;
	int *cur_fw_cp;
	int *cur_bw_cp;
};

struct layer_info *layer;
cublasHandle_t *handle;
cudaStream_t *stream;

double *input_d, *input_host, *result_d, *result_host, *sum;
int *num_node_arr;

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
	stream = (cudaStream_t *)malloc(2 * num_layer * sizeof(cudaStream_t));
	sum = (double *)malloc(2 * sizeof(double));

	// build model
	layer = (struct layer_info *)malloc(num_layer * sizeof(struct layer_info));
	for(int i = 0; i < num_layer; i++)
		SetLayer(i);

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
	cudaStreamCreate(&stream[2 * layer_index]);
	cudaStreamCreate(&stream[2 * layer_index + 1]);
	cublasCreate(&handle[layer_index]);
	cublasSetStream(handle[layer_index], stream[2 * layer_index]);

	cudaMallocManaged((void**) &layer[layer_index].is_fw_act_ready, 2 * sizeof(int));
	cudaMallocManaged((void**) &layer[layer_index].is_bw_delta_ready, 2 * sizeof(int));
	layer[layer_index].is_fw_act_ready[0] = layer[layer_index].is_fw_act_ready[1] = 0;
	layer[layer_index].is_bw_delta_ready[0] = layer[layer_index].is_bw_delta_ready[0] = 0;

	cudaMallocManaged((void**) &layer[layer_index].is_fw_input_ready, 2 * sizeof(int));
	cudaMallocManaged((void**) &layer[layer_index].is_bw_input_ready, 2 * sizeof(int));
	layer[layer_index].is_fw_input_ready[0] = layer[layer_index].is_fw_input_ready[1] = 0;
	layer[layer_index].is_bw_input_ready[0] = layer[layer_index].is_bw_input_ready[0] = 0;
	
	cudaMallocManaged((void**) &layer[layer_index].cur_fw_cm, sizeof(int));
	cudaMallocManaged((void**) &layer[layer_index].cur_bw_cm, sizeof(int));
	layer[layer_index].cur_fw_cm = layer[layer_index].cur_bw_cm = 0;

	cudaMallocManaged((void**) &layer[layer_index].cur_fw_cp, sizeof(int));
	cudaMallocManaged((void**) &layer[layer_index].cur_bw_cp, sizeof(int));
	layer[layer_index].cur_fw_cp = layer[layer_index].cur_bw_cp = 0;

	layer[layer_index].a = (double **)malloc(2 * sizeof(double *));
	for(int i = 0; i < 2; i++)
		cudaMalloc((void**) &layer[layer_index].a[i], num_node_arr[layer_index] * sizeof(double));

	// except input layer
	if(layer_index != 0)
	{
		layer[layer_index].delta = (double **)malloc(2 * sizeof(double *));
		for(int i = 0; i < 2; i++)
			cudaMalloc((void**) &layer[layer_index].delta, num_node_arr[layer_index] * sizeof(double));
	}

	// except output layer
	if(layer_index < num_layer - 1)
	{
		next_node = num_node_arr[layer_index + 1];

		layer[layer_index].a_next = (double **)malloc(2 * sizeof(double *));
		for(int i = 0; i < 2; i++)
			cudaMalloc((void**) &layer[layer_index].a_next[i], next_node * sizeof(double));

		layer[layer_index].delta_next = (double **)malloc(2 * sizeof(double *));
		for(int i = 0; i < 2; i++)
			cudaMalloc((void**) &layer[layer_index].delta_next[i], next_node * sizeof(double));

		weight_host = (double *)malloc(cur_node * next_node * sizeof(double));
		for (int j = 0; j < cur_node * next_node; j++)
			weight_host[j] = sqrt(6.0 / (cur_node + next_node)) * (rand() / (double)RAND_MAX * 2.0 - 1.0);
		cudaMalloc((void**) &layer[layer_index].weight, cur_node * next_node * sizeof(double));
		cudaMemcpy(layer[layer_index].weight, weight_host, cur_node * next_node * sizeof(double), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		free(weight_host);
	}
}

void train_model() 
{
	int result_index = 0, input_index = 0;
	FILE* pFile = NULL;
	char str_tmp[num_node_arr[0] * 3], *p;

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
	
	///////////////////////////////////////////////////////////////////
	// startup stage //////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////////
	// input layer ////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

	cudaSetDevice(0);
	for(int j = 0; j < num_layer; j++)
	{	// forward X num_layer
		// copy input data
		cudaMemcpyAsync(layer[0].a, input_d + j * num_node_arr[0], num_node_arr[0] * sizeof(double), cudaMemcpyDeviceToDevice, stream[0]);

		// wait for current layer's activation buffer is empty
		WaituntilZero<<<1, 1, 0, stream[0]>>>(layer[0].is_fw_act_ready, layer[0].cur_fw_cp);

		// compute activation
		MatrixMultiply(layer[0].a[layer[0].cur_fw_cp], layer[0].weight, layer[0].a_next[layer[0].cur_fw_cp], 1, num_node_arr[0], num_node_arr[1], 0);
		Sigmoid<<<(num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a_next[layer[0].cur_fw_cp], num_node_arr[1]);

		// current layer's activation buffer is full
		SetFlag<<<1, 1, 0, stream[0]>>>(layer[0].is_fw_act_ready, layer[0].cur_fw_cp);

		// buffer exchange
		SetBuffer<<<1, 1, 0, stream[0]>>>(layer[0].cur_fw_cp);

		// wait for current layer's activation buffer is full
		WaituntilOne<<<1, 1, 0, stream[1]>>>(layer[0].is_fw_act_ready, layer[0].cur_fw_cm);

		// wait for next layer's forward input buffer is empty
		WaituntilZero<<<1, 1, 0, stream[1]>>>(layer[1].is_fw_input_ready, layer[0].cur_fw_cm);

		// copy activation to next layer
		cudaMemcpyPeerAsync(layer[1].a[layer[0].cur_fw_cm], 1, layer[0].a_next[layer[0].cur_fw_cm], 0, num_node_arr[1] * sizeof(double), stream[1]);

		// next layer's forward input buffer is full
		SetFlag<<<1, 1, 0, stream[1]>>>(layer[1].is_fw_input_ready, layer[0].cur_fw_cm);

		// current layer's activation buffer is empty
		SetFlag<<<1, 1, 0, stream[1]>>>(layer[0].is_fw_act_ready, layer[0].cur_fw_cm);

		// buffer exchange
		SetBuffer<<<1, 1, 0, stream[1]>>>(layer[0].cur_fw_cm);
	}

	// Backward
	// wait for current layer's backward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[0]>>>(layer[0].is_bw_input_ready, layer[0].cur_bw_cp);

	// update weight
	UpdateWeight<<<(num_node_arr[0] * num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a[layer[0].cur_bw_cp], layer[0].weight, layer[0].delta_next[layer[0].cur_bw_cp], num_node_arr[1], num_node_arr[1] * num_node_arr[0], alpha);

	// current layer's backward input buffer is empty
	SetFlag<<<1, 1, 0, stream[0]>>>(layer[0].is_bw_delta_ready, layer[0].cur_bw_cp);

	// buffer exchange
	SetBuffer<<<1, 1, 0, stream[0]>>>(layer[0].cur_bw_cp);

	///////////////////////////////////////////////////////////////////
	// hidden layer ///////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

	for(int i = 1; i < num_layer - 1; i++)
	{
		cudaSetDevice(i);
		for(int j = 0; j < num_layer; j++)
		{	// forward X num_layer
			// wait for current layer's forward input buffer is full
			WaituntilOne<<<1, 1, 0, stream[2 * i]>>>(layer[i].is_fw_input_ready, layer[i].cur_fw_cp);

			// wait for current layer's activation buffer is empty
			WaituntilZero<<<1, 1, 0, stream[2 * i]>>>(layer[i].is_fw_act_ready, layer[i].cur_fw_cp);

			// compute activation
			MatrixMultiply(layer[i].a[layer[i].cur_fw_cp], layer[i].weight, layer[i].a_next[layer[i].cur_fw_cp], 1, num_node_arr[i], num_node_arr[i + 1], i);
			if(i != num_layer - 2)
				Sigmoid<<<(num_node_arr[i + 1] + 1023) / 1024, 1024, 0, stream[2 * i]>>>(layer[i].a_next[layer[i].cur_fw_cp], num_node_arr[i + 1]);

			// current layer's activation buffer is full
			SetFlag<<<1, 1, 0, stream[2 * i]>>>(layer[i].is_fw_act_ready, layer[i].cur_fw_cp);

			// current layer's forward input buffer is empty
			SetFlag<<<1, 1, 0, stream[2 * i]>>>(layer[i].is_fw_input_ready, layer[i].cur_fw_cp);

			// buffer exchange
			SetBuffer<<<1, 1, 0, stream[2 * i]>>>(layer[i].cur_fw_cp);

			// wait for current layer's activation buffer is full
			WaituntilOne<<<1, 1, 0, stream[2 * i + 1]>>>(layer[i].is_fw_act_ready, layer[i].cur_fw_cm);
				
			// wait for next layer's forward input buffer is empty
			WaituntilZero<<<1, 1, 0, stream[2 * i + 1]>>>(layer[i + 1].is_fw_input_ready, layer[i].cur_fw_cm);

			// copy activation to next layer
			cudaMemcpyPeerAsync(layer[i + 1].a[layer[i].cur_fw_cm], i + 1, layer[i].a_next[layer[i].cur_fw_cm], i, num_node_arr[i + 1] * sizeof(double), stream[2 * i + 1]);

			// next layer's forward input buffer is full
			SetFlag<<<1, 1, 0, stream[2 * i + 1]>>>(layer[i + 1].is_fw_input_ready, layer[i].cur_fw_cm);

			// current layer's activation buffer is empty
			SetFlag<<<1, 1, 0, stream[2 * i + 1]>>>(layer[i].is_fw_act_ready, layer[i].cur_fw_cm);

			// buffer exchange
			SetBuffer<<<1, 1, 0, stream[2 * i + 1]>>>(layer[i].cur_fw_cm);
		}

		for(int j = 0; j < i + 1; j++)
		{	// backward X (layer_index + 1)
			// wait for current layer's backward input buffer is full
			WaituntilOne<<<1, 1, 0, stream[2 * i]>>>(layer[i].is_bw_input_ready, layer[i].cur_bw_cp);

			// wait for current layer's delta buffer is empty
			WaituntilZero<<<1, 1, 0, stream[2 * i]>>>(layer[i].is_bw_delta_ready, layer[i].cur_bw_cp);

			// compute delta
			MatrixMultiply(layer[i].weight, layer[i].delta_next[layer[i].cur_bw_cp] + sizeof(double), layer[i].delta[layer[i].cur_bw_cp], num_node_arr[i], num_node_arr[i + 1] - 1, 1, i);
			//GetHiddenLayerDelta<<<(num_node_arr[i] + 1023) / 1024, 1024, 0, stream[i]>>>(layer[i].delta[j], layer[i].a[j], layer[i].weight[j], layer[i].delta_next[j], num_node_arr[i]);

			// current layer's delta buffer is full
			SetFlag<<<1, 1, 0, stream[2 * i]>>>(layer[i].is_bw_delta_ready, layer[i].cur_bw_cp);

			// update weight
			UpdateWeight<<<(num_node_arr[i] * num_node_arr[i + 1] + 1023) / 1024, 1024, 0, stream[2 * i]>>>(layer[i].a[layer[i].cur_bw_cp], layer[i].weight, layer[i].delta_next[layer[i].cur_bw_cp], num_node_arr[i + 1], num_node_arr[i + 1] * num_node_arr[i], alpha);

			// curren layer's backward input buffer is empty
			SetFlag<<<1, 1, 0, stream[2 * i]>>>(layer[i].is_bw_input_ready, layer[i].cur_bw_cp);

			//buffer exchange
			SetBuffer<<<1, 1, 0, stream[2 * i]>>>(layer[i].cur_bw_cp);

			// wait for current layer's delta buffer is full
			WaituntilOne<<<1, 1, 0, stream[2 * i + 1]>>>(layer[i].is_bw_delta_ready, layer[i].cur_bw_cm);

			// wait for previous layer's backward input buffer is empty
			WaituntilZero<<<1, 1, 0, stream[2 * i + 1]>>>(layer[i - 1].is_bw_input_ready, layer[i].cur_bw_cm);

			// copy delta to preious layer
			cudaMemcpyPeerAsync(layer[i - 1].delta_next[layer[i].cur_bw_cm], i - 1, layer[i].delta[layer[i].cur_bw_cm], i, num_node_arr[i - 1] * sizeof(double), stream[2 * i + 1]);

			// previous backward input buffer is full
			SetFlag<<<1, 1, 0, stream[2 * i + 1]>>>(layer[i - 1].is_bw_input_ready, layer[i].cur_bw_cm);

			// current layer's delta buffer is empty
			SetFlag<<<1, 1, 0, stream[2 * i + 1]>>>(layer[i].is_bw_delta_ready, layer[i].cur_bw_cm);

			// buffer exchange
			SetBuffer<<<1, 1, 0, stream[2 * i + 1]>>>(layer[i].cur_bw_cm);
		}	
	}

	///////////////////////////////////////////////////////////////////
	// output layer ///////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

	cudaSetDevice(num_layer - 1);
	for(int j = 0; j < num_layer; j++)
	{	// (Forward + Backward) X num_layer
		// wait for current layer's forward input buffer is full
		WaituntilOne<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].is_fw_input_ready, layer[num_layer - 1].cur_fw_cp);
		
		// softmax
		Exponential<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[layer[num_layer - 1].cur_fw_cp], num_node_arr[num_layer - 1]);
		cublasDasum(handle[num_layer - 1], num_node_arr[num_layer - 1], layer[num_layer - 1].a[layer[num_layer - 1].cur_fw_cp], 1, &sum[layer[num_layer - 1].cur_fw_cp]);
		Softmax<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[layer[num_layer - 1].cur_fw_cp], sum[layer[num_layer - 1].cur_fw_cp], num_node_arr[num_layer - 1]);

		// wait for current layer's delta buffer is empty
		WaituntilZero<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].is_bw_delta_ready, layer[num_layer - 1].cur_bw_cp);

		// compute delta
		GetOutputLayerDelta<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[layer[num_layer - 1].cur_bw_cp], layer[num_layer - 1].delta[layer[num_layer - 1].cur_bw_cp], result_d + j, num_node_arr[num_layer - 1]);

		// current layer's delta buffer is full
		SetFlag<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].is_bw_delta_ready, layer[num_layer - 1].cur_bw_cp);

		// current layer's forward input buffer is empty
		SetFlag<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].is_fw_input_ready, layer[num_layer - 1].cur_fw_cp);

		// buffer exchange
		SetBuffer<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].cur_fw_cp);
		SetBuffer<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].cur_bw_cp);

		// wait for previous layer's backward input buffer is empty
		WaituntilZero<<<1, 1, 0, stream[2 * num_layer - 1]>>>(layer[num_layer - 2].is_bw_input_ready, layer[num_layer - 1].cur_bw_cm);

		// wait for current layer's delta buffer is full
		WaituntilOne<<<1, 1, 0, stream[2 * num_layer - 1]>>>(layer[num_layer - 1].is_bw_delta_ready, layer[num_layer - 1].cur_bw_cm);

		// copy delta to previous layer
		cudaMemcpyPeerAsync(layer[num_layer - 2].delta_next[layer[num_layer - 1].cur_bw_cm], num_layer - 2, layer[num_layer - 1].delta[layer[num_layer - 1].cur_bw_cm], num_layer - 1, num_node_arr[num_layer - 1] * sizeof(double), stream[2 * num_layer -1]);

		// previous layer's backward input buffer is full
		SetFlag<<<1, 1, 0, stream[2 * num_layer - 1]>>>(layer[num_layer - 2].is_bw_input_ready, layer[num_layer - 1].cur_bw_cm);

		// current layer's delta buffer is empty
		SetFlag<<<1, 1, 0, stream[2 * num_layer - 1]>>>(layer[num_layer - 1].is_bw_delta_ready, layer[num_layer - 1].cur_bw_cm);

		// buffer exchange
		SetBuffer<<<1, 1, 0, stream[2 * num_layer - 1]>>>(layer[num_layer - 1].cur_bw_cm);
	}

	///////////////////////////////////////////////////////////////////
	// steady stage ///////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

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

__global__ void WaituntilZero(int *ready, int index)
{
	while(ready[index] == 1) {
		// for(int i = 0; i < 100; i++)
		// { }
		/*
		printf("device : %d, ", device);
		if(fb == 1)
			printf("direction : fw, ");
		else
			printf("direction : bw, ");
		printf("%d\n", ready[index]);
		*/
	}
}

__global__ void WaituntilOne(int *ready, int index)
{
	
	while(ready[index] == 0) { 
		// for(int i = 0; i < 10000000; i++)
		// { }
		/*
		printf("device : %d, ", device);
		if(fb == 1)
			printf("direction : fw, ");
		else
			printf("direction : bw, ");
		printf("%d\n", ready[index]);
		*/
	}
}

__global__ void SetFlag(int *ready, int index)
{
	ready[index] = 1 - ready[index];
}

__global__ void SetBuffer(int *index)
{
	index = 1 - index;
}

__global__ void dummy()
{
	while(1) { }
}
