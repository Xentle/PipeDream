#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cublas_v2.h>

#define alpha 0.001
#define lambda 0.001

// Debug
__global__ void PrintInt(int *arr, int size);
__global__ void PrintDouble(double *arr, int size);
__global__ void PrintFw(int device, int index, int e);
__global__ void PrintBw(int device, int index, int e);

// Computation
__global__ void GetOutputLayerDelta(double *output_a, double *output_delta, double *result, int index, int size);
__global__ void GetHiddenLayerDelta(double *cur_delta, double *cur_a, double *cur_weight, double *prev_delta, int size);
__global__ void UpdateWeight(double *cur_a, double *cur_weight, double *next_weight, double *delayed_weight, double *next_delta, int W_W, int size, double lr);
__global__ void Sigmoid(double *a, int size);
__global__ void Exponential(double *a, int size);
__global__ void Softmax(double *a, double* sum, int size);
__global__ void CheckCorrect(double *test_result_d, int fw_index, int *max_index, double *num_correct);
__global__ void GetMaxIndex(double *a, int num_node, int *index);
__global__ void GetSum(double *a, int num_node, double *s);
void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W, int i);

// Scheduling
void InputForwardComputation(int index);
void InputForwardCommunication(int index, int e);
void HiddenForwardComputation(int device, int index);
void HiddenForwardCommunication(int device, int index, int e);
void InputForwardBackwardComputation(int fw_index, int bw_index, int e);
void HiddenForwardBackwardComputation(int device, int fw_index, int bw_index, int e);
void HiddenForwardBackwardCommunication(int device, int fw_index, int bw_index, int e);
void OutputForwardComputation(int fw_index);
void OutputForwardBackwardComputation(int fw_index, int bw_index, int e);
void OutputForwardBackwardCommunication(int fw_index, int bw_index, int e);

// Synchronization
__global__ void WaituntilZero(int *ready, int index, int line, int x, int e, int d);
__global__ void WaituntilOne(int *ready, int index, int line, int x, int e, int d);
__global__ void SetFlag(int *ready, int index);

// Build Model
void GetResultAndInput();
void SetLayer(int layer_index);

// Main
void train_model();
void test_accuracy();

struct layer_info {
	double **weight;
	double **a;
	double **a_next;
	double **delta;
	double **delta_next;
	int *is_fw_input_ready;
	int *is_bw_input_ready;
	int *is_fw_output_ready;
	int *is_bw_output_ready;
	int *is_fw_next_input_ready;
	int *is_bw_prev_input_ready;
};

struct layer_info *layer;
cublasHandle_t *handle;
cudaStream_t *stream;

double *input_d, *input_host, *result_d, *result_host, *sum;
double *test_input_d, *test_input_host, *test_result_d, *test_result_host, *d_num_correct, *zero, *d_train_correct;
int *num_node_arr, *cur_fw, *cur_bw, *e, *max_index, *d_max_index;

int num_layer = 0, num_data = 0, epoch = 0, test_num_data = 0;
double num_correct = 0;

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

	printf("number of test data : ");
	scanf(" %d", &test_num_data);

	printf("epoch : ");
	scanf(" %d", &epoch);

	// make tool
	handle = (cublasHandle_t *)malloc(num_layer * sizeof(cublasHandle_t));
	stream = (cudaStream_t *)malloc(2 * num_layer * sizeof(cudaStream_t));

	cur_fw = (int *)malloc(num_layer * sizeof(int));
	cur_bw = (int *)malloc(num_layer * sizeof(int));
	for(int i = 0; i < num_layer; i++)
		cur_fw[i] = cur_bw[i] = 0;

	e = (int *)malloc(num_layer * sizeof(int));
	for(int i = 0; i < num_layer; i++)
		e[i] = 0;
	
	zero = (double *)malloc(sizeof(double));
	zero[0] = 0.0;

	// build model
	GetResultAndInput();
	layer = (struct layer_info *)malloc(num_layer * sizeof(struct layer_info));
	for(int i = 0; i < num_layer; i++)
		SetLayer(i);

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
	for(int i = 0; i < num_layer; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
	
	start = clock();
	
	// train and test
	train_model();
	for(int i = 0; i < num_layer; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

	printf("%fs\n", (double)(clock() - start)/CLOCKS_PER_SEC);

	// test accuracy
	test_accuracy();

	for(int i = 0; i < num_layer; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

	printf("finished\n");

	return 0;
}

void SetLayer(int layer_index)
{
	int cur_node = num_node_arr[layer_index], next_node, num_buffer = (num_layer - layer_index) * 2 - 1;
	double *weight_host;

	cudaSetDevice(layer_index);
	cudaStreamCreate(&stream[2 * layer_index]);
	cudaStreamCreate(&stream[2 * layer_index + 1]);
	cublasCreate(&handle[layer_index]);
	cublasSetStream(handle[layer_index], stream[2 * layer_index]);

    layer[layer_index].a = (double **)malloc(num_buffer * sizeof(double *));
	for(int i = 0; i < num_buffer; i++)
        cudaMalloc((void**) &layer[layer_index].a[i], num_node_arr[layer_index] * sizeof(double));
    cudaMalloc((void**) &layer[layer_index].is_fw_input_ready, num_buffer * sizeof(int));

    // except input layer
    if(layer_index != 0)
    {
        cudaMalloc((void**) &layer[layer_index].is_bw_output_ready, num_buffer * sizeof(int));
        cudaMalloc((void**) &layer[layer_index].is_bw_prev_input_ready, (num_buffer + 2) * sizeof(int));

        layer[layer_index].delta = (double **)malloc(num_buffer * sizeof(double *));
		for(int i = 0; i < num_buffer; i++)
			cudaMalloc((void**) &layer[layer_index].delta[i], num_node_arr[layer_index] * sizeof(double));
    }

	// except output layer
	if(layer_index < num_layer - 1)
	{
        next_node = num_node_arr[layer_index + 1];
        cudaMalloc((void**) &layer[layer_index].is_bw_input_ready, num_buffer * sizeof(int));
        cudaMalloc((void**) &layer[layer_index].is_fw_output_ready, num_buffer * sizeof(int));
        cudaMalloc((void**) &layer[layer_index].is_fw_next_input_ready, (num_buffer - 2) * sizeof(int));

		layer[layer_index].a_next = (double **)malloc(num_buffer * sizeof(double *));
		for(int i = 0; i < num_buffer; i++)
			cudaMalloc((void**) &layer[layer_index].a_next[i], next_node * sizeof(double));

		layer[layer_index].delta_next = (double **)malloc(num_buffer * sizeof(double *));
		for(int i = 0; i < num_buffer; i++)
			cudaMalloc((void**) &layer[layer_index].delta_next[i], next_node * sizeof(double));

        weight_host = (double *)malloc(cur_node * next_node * sizeof(double));
        layer[layer_index].weight = (double **)malloc(num_buffer * sizeof(double *));
		for (int j = 0; j < cur_node * next_node; j++)
            weight_host[j] = sqrt(6.0 / (cur_node + next_node)) * (rand() / (double)RAND_MAX * 2.0 - 1.0);
        for(int i = 0; i < num_buffer; i++)
        {
            cudaMalloc((void**) &layer[layer_index].weight[i], cur_node * next_node * sizeof(double));
            cudaMemcpy(layer[layer_index].weight[i], weight_host, cur_node * next_node * sizeof(double), cudaMemcpyHostToDevice);
        }
		cudaDeviceSynchronize();
		free(weight_host);
	}
}

void GetResultAndInput()
{
	FILE* pFile = NULL;
	char str_tmp[num_node_arr[0] * 3], *p;

	pFile = fopen("mnist_train.csv", "r");
	result_host = (double *)malloc(num_data * sizeof(double));
	input_host = (double *)malloc(num_data * num_node_arr[0] * sizeof(double));
	if(pFile != NULL)
    {   
		for(int r_index = 0, i_index = 0; r_index < num_data;)
		{
			fgets(str_tmp, num_node_arr[0] * 3, pFile);

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
	cudaMalloc((void**) &result_d, num_data * sizeof(double));
	cudaMemcpy(result_d, result_host, num_data * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &sum, sizeof(double));

	cudaMalloc((void**) &d_num_correct, sizeof(double));
	cudaMalloc((void**) &d_train_correct, sizeof(double));
	cudaMalloc((void**) &d_max_index, sizeof(int)); max_index = (int *)malloc(sizeof(int));

	cudaSetDevice(0);
	cudaMalloc((void**) &input_d, num_data * num_node_arr[0] * sizeof(double));
	cudaMemcpy(input_d, input_host, num_data * num_node_arr[0] * sizeof(double), cudaMemcpyHostToDevice);
}

void train_model() 
{
	//start stage
	for(int i = 0; i < num_layer - 1; i++)
	{
		if(i == 0)
		{
			while(cur_fw[i] < 2 * (num_layer - 1))
			{
				// printf("%d %d\n", i, cur_fw[i]);
				InputForwardComputation(cur_fw[i]);
				InputForwardCommunication(cur_fw[i]++, 0);
			}
		}
		else
		{
			while(cur_fw[i] < 2 * (num_layer  - 1 - i))
			{
				// printf("%d %d\n", i, cur_fw[i]);
				HiddenForwardComputation(i, cur_fw[i]);
				HiddenForwardCommunication(i, cur_fw[i]++, 0);
			}
		}
	}
	for(int i = 1; i < num_layer; i++)
	{
		if(i == num_layer - 1)
		{
			while(cur_fw[i] < i)
			{
				// printf("%d\t%d, %d\n", i, cur_fw[i], cur_bw[i]);
				OutputForwardBackwardComputation(cur_fw[i], cur_bw[i], 0);
				OutputForwardBackwardCommunication(cur_fw[i]++, cur_bw[i]++, 0);
			}
		}
		else
		{
			while(cur_fw[i] < i + 2 * (num_layer  - 1 - i))
			{
				// printf("%d\t%d, %d\n", i, cur_fw[i], cur_bw[i]);
				HiddenForwardBackwardComputation(i, cur_fw[i], cur_bw[i], 0);
				HiddenForwardBackwardCommunication(i, cur_fw[i]++, cur_bw[i]++, 0);
			}
		}
	}
	
	for(int i = 0; i < num_layer; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
	
	// steady stage		if (data 개수 홀수) : epoch 홀수 / 짝수일 때 시작 buffer remain different buffer
	while(e[0] < epoch)
	{
		for(int i = 0; i < num_layer; i++)
		{
			if(i == 0)
			{
				InputForwardBackwardComputation(cur_fw[i], cur_bw[i]++, e[i]);
				InputForwardCommunication(cur_fw[i]++, e[i]);
			}
			else if(i == num_layer - 1)
			{
				OutputForwardBackwardComputation(cur_fw[i], cur_bw[i], e[i]);
				OutputForwardBackwardCommunication(cur_fw[i]++, cur_bw[i]++, e[i]);
			}
			else
			{
				HiddenForwardBackwardComputation(i, cur_fw[i], cur_bw[i], e[i]);
				HiddenForwardBackwardCommunication(i, cur_fw[i]++, cur_bw[i]++, e[i]);
			}
		}

		for(int i = 0; i < num_layer; i++)
		{
			if(cur_fw[i] == num_data)
				cur_fw[i] = 0;
			if(cur_bw[i] == num_data)
			{
				cur_bw[i] = 0;
				e[i]++;
			}
		}
	}

	// end stage
	// while(cur_bw[0] < num_data)
	// {
	// 	for(int i = 0; i < num_layer; i++)
	// 	{
	// 		if(i == 0)
	// 		{
	// 			InputBackwardComputation(cur_bw[i]++);
	// 		}
	// 		else if(i != num_layer - 1)
	// 		{
	// 			HiddenBackwardComputation(cur_bw[i]);
	// 			HiddenBackwardCommunication(cur_bw[i]++);
	// 		}
	// 	}
	// }
	
//	print device flag
//	for(int i = 0; i < num_layer; i++)
//	{
//		cudaSetDevice(i);
//		PrintInt<<<1, 1>>>(layer[i].is_fw_input_ready, 0, __LINE__);
//		PrintInt<<<1, 1>>>(layer[i].is_fw_input_ready, 1, __LINE__);
//
//		PrintInt<<<1, 1>>>(layer[i].is_bw_input_ready, 0, __LINE__);
//		PrintInt<<<1, 1>>>(layer[i].is_bw_input_ready, 1, __LINE__);
//
//		PrintInt<<<1, 1>>>(layer[i].is_fw_output_ready, 0, __LINE__);
//		PrintInt<<<1, 1>>>(layer[i].is_fw_output_ready, 1, __LINE__);
//
//		PrintInt<<<1, 1>>>(layer[i].is_bw_output_ready, 0, __LINE__);
//		PrintInt<<<1, 1>>>(layer[i].is_bw_output_ready, 1, __LINE__);
//
//		PrintInt<<<1, 1>>>(layer[i].is_fw_next_input_ready, 0, __LINE__);
//		PrintInt<<<1, 1>>>(layer[i].is_fw_next_input_ready, 1, __LINE__);
//
//		PrintInt<<<1, 1>>>(layer[i].is_bw_prev_input_ready, 0, __LINE__);
//		PrintInt<<<1, 1>>>(layer[i].is_bw_prev_input_ready, 1, __LINE__);
//		PrintEnter<<<1, 1>>>();
//
//		cudaDeviceSynchronize();
//	}
}

void test_accuracy() 
{	
	// Set test data
	FILE* pFile = NULL;
	char str_tmp[num_node_arr[0] * 3], *p;

	pFile = fopen("mnist_test.csv", "r");
	if(pFile != NULL)
	{
		for(int r_index = 0, i_index = 0; r_index < test_num_data;)
		{
			fgets(str_tmp, num_node_arr[0] * 3, pFile);

			// set test_result
			p = strtok(str_tmp, ",");
			if(atoi(p) == 0)
				result_host[r_index++] = num_node_arr[num_layer - 1] - 1;
			else
				result_host[r_index++] = atoi(p);

			// set test_input
			input_host[i_index++] = 1.0;
			for (int i = 1; i < num_node_arr[0]; i++)
			{
				p = strtok(NULL, ",");
				input_host[i_index++] = atof(p) / 255.0;
			}
	    }
	}

	cudaSetDevice(0);
	cudaMemcpy(input_d, input_host, test_num_data * num_node_arr[0] * sizeof(double), cudaMemcpyHostToDevice);

	cudaSetDevice(num_layer - 1);
	cudaMemcpy(result_d, result_host, test_num_data * sizeof(double), cudaMemcpyHostToDevice);

	for(int i = 0; i < num_layer; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

	// reset flags
	for(int i = 0 ; i < num_layer - 1; i++)
		cur_fw[i] = i;
	cur_fw[num_layer - 1] = 0;

	
	for(int i = 0; i < num_layer; i++)
	{
		cudaSetDevice(i);
		for(int j = 0; j < (num_layer - j) * 2 - 1; j++)
		{
			cudaMemset(&layer[i].is_fw_input_ready[j], 0, sizeof(int));

			if(i != num_layer - 1)
			{
				cudaMemset(&layer[i].is_fw_output_ready[j], 0, sizeof(int));
				cudaMemset(&layer[i].is_bw_input_ready[j], 0, sizeof(int));
			}

			if(i != 0)
				cudaMemset(&layer[i].is_bw_output_ready[j], 0, sizeof(int));
		}

		if(i != num_layer - 1)
			for(int j = 0; j < (num_layer - j) * 2 - 3; j++)
				cudaMemset(&layer[i].is_fw_next_input_ready[j], 0, sizeof(int));

		if(i != 0)
			for(int j = 0; j < (num_layer - j) * 2 + 1; j++)
				cudaMemset(&layer[i].is_bw_prev_input_ready[j], 0, sizeof(int));
	}

	for(int i = 0; i < num_layer; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

	while(cur_fw[num_layer - 1] < test_num_data)
	{
		for(int i = 0; i < num_layer; i++)
		{
			if(i == 0)
			{
				InputForwardComputation(cur_fw[i]);
				InputForwardCommunication(cur_fw[i]++, -1);
			}
			else if(i == num_layer - 1)
				OutputForwardComputation(cur_fw[i]++);
			else
			{
				HiddenForwardComputation(i, cur_fw[i]);
				HiddenForwardCommunication(i, cur_fw[i], -1);
			}
		}
	}

	for(int i = 0; i < num_layer; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(&num_correct, d_num_correct, sizeof(double), cudaMemcpyDeviceToHost);
	for(int i = 0; i < num_layer; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

	printf("%lf%%\n", num_correct / test_num_data * 100);
}

void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W, int i)
{
	const double alp = 1.0f;
	const double bet  = 0.0f;
		
	cublasDgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, B_W, A_H, A_W, &alp, d_B, B_W, d_A, A_W, &bet, d_C, B_W);
}

void InputForwardComputation(int index)
{
    int num_buffer = num_layer * 2 - 1;
    int buffer = index % num_buffer;
    cudaSetDevice(0);

    // wait for current layer's forward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[0]>>>(layer[0].is_fw_input_ready, buffer, __LINE__, index, 0, 0);
	
	// copy input data
	cudaMemcpyAsync(layer[0].a[buffer], input_d + index * num_node_arr[0], num_node_arr[0] * sizeof(double), cudaMemcpyDeviceToDevice, stream[0]);
	
	// wait for current layer's forward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[0]>>>(layer[0].is_fw_output_ready, buffer, __LINE__, index, 0, 0);
	
	// compute activation
	MatrixMultiply(layer[0].a[buffer], layer[0].weight[buffer], layer[0].a_next[buffer], 1, num_node_arr[0], num_node_arr[1], 0);
	Sigmoid<<<(num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a_next[buffer], num_node_arr[1]);
	
	// current layer's forward output buffer is full
	cudaMemsetAsync(&layer[0].is_fw_output_ready[buffer], 1, sizeof(int), stream[0]);
}

void InputForwardCommunication(int index, int e)
{
    int num_buffer = num_layer * 2 - 1;
	int buffer = index % num_buffer;
	int next_buffer;
	if(e == -1)
		next_buffer = 1;
	else
    	next_buffer = index % (num_buffer - 2);
	cudaSetDevice(0);

	// wait for current layer's forward output buffer is full
	WaituntilOne<<<1, 1, 0, stream[1]>>>(layer[0].is_fw_output_ready, buffer, __LINE__, index, e, 0);
	
	// wait for next layer's forward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[1]>>>(layer[0].is_fw_next_input_ready, next_buffer, __LINE__, index, e, 0);

	// copy foward output (activation) to next layer
	cudaMemcpyPeerAsync(layer[1].a[next_buffer], 1, layer[0].a_next[buffer], 0, num_node_arr[1] * sizeof(double), stream[1]);
	
	// current layer's forward output buffer is empty
	cudaMemsetAsync(&layer[0].is_fw_output_ready[buffer], 0, sizeof(int), stream[1]);

	// next layer's forward input buffer is full
	cudaMemsetAsync(&layer[0].is_fw_next_input_ready[next_buffer], 1, sizeof(int), stream[1]);
	cudaMemcpyPeerAsync(&layer[1].is_fw_input_ready[next_buffer], 1, &layer[0].is_fw_next_input_ready[next_buffer], 0, sizeof(int), stream[1]);
}

void InputForwardBackwardComputation(int fw_index, int bw_index, int e)
{
    int num_buffer = num_layer * 2 - 1;
    int fw_buffer = fw_index % num_buffer;
    int bw_buffer = bw_index % num_buffer;
    cudaSetDevice(0);

    // wait for current layer's forward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[0]>>>(layer[0].is_fw_input_ready, fw_buffer, __LINE__, fw_index, 0, 0);

	// copy input data
	cudaMemcpyAsync(layer[0].a[fw_buffer], input_d + fw_index * num_node_arr[0], num_node_arr[0] * sizeof(double), cudaMemcpyDeviceToDevice, stream[0]);
	
	// wait for current layer's forward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[0]>>>(layer[0].is_fw_output_ready, fw_buffer, __LINE__, fw_index, e, 0);
	
	// compute activation
	MatrixMultiply(layer[0].a[fw_buffer], layer[0].weight[fw_buffer], layer[0].a_next[fw_buffer], 1, num_node_arr[0], num_node_arr[1], 0);
	Sigmoid<<<(num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a_next[fw_buffer], num_node_arr[1]);
	
	// current layer's forward output buffer is full
	cudaMemsetAsync(&layer[0].is_fw_output_ready[fw_buffer], 1, sizeof(int), stream[0]);

	// wait for current layer's backward input buffer is full
    WaituntilOne<<<1, 1, 0, stream[0]>>>(layer[0].is_bw_input_ready, bw_buffer, __LINE__, bw_index, e, 0);

	// update weight
    UpdateWeight<<<(num_node_arr[0] * num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a[bw_buffer], layer[0].weight[fw_buffer], layer[0].weight[(fw_buffer + 1) % num_buffer], layer[0].weight[bw_buffer], layer[0].delta_next[bw_buffer], num_node_arr[1], num_node_arr[1] * num_node_arr[0], alpha);

    // current layer's forward input buffer is empty
    cudaMemsetAsync(&layer[0].is_fw_input_ready[bw_buffer], 0, sizeof(int), stream[0]);

	// current layer's backward input buffer is empty
	cudaMemsetAsync(&layer[0].is_bw_input_ready[bw_buffer], 0, sizeof(int), stream[0]);
	cudaMemcpyPeerAsync(&layer[1].is_bw_prev_input_ready[bw_buffer], 1, &layer[0].is_bw_input_ready[bw_buffer], 0, sizeof(int), stream[0]);
}

void HiddenForwardComputation(int device, int index)
{
	int num_buffer = (num_layer - device) * 2 - 1;
    int buffer = index % num_buffer;
	cudaSetDevice(device);

	// wait for current layer's forward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_fw_input_ready, buffer, __LINE__, index, 0, device);

	// wait for current layer's forward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_fw_output_ready, buffer, __LINE__, index, 0, device);

	// compute activation
	MatrixMultiply(layer[device].a[buffer], layer[device].weight[buffer], layer[device].a_next[buffer], 1, num_node_arr[device], num_node_arr[device + 1], device);
	if(device != num_layer - 2)
		Sigmoid<<<(num_node_arr[device + 1] + 1023) / 1024, 1024, 0, stream[2 * device]>>>(layer[device].a_next[buffer], num_node_arr[device + 1]);

	// current layer's forward output buffer is full
	cudaMemsetAsync(&layer[device].is_fw_output_ready[buffer], 1, sizeof(int), stream[2 * device]);

	// current layer's forward input buffer is empty
	cudaMemsetAsync(&layer[device].is_fw_input_ready[buffer], 0, sizeof(int), stream[2 * device]);
	cudaMemcpyPeerAsync(&layer[device - 1].is_fw_next_input_ready[buffer], device - 1, &layer[device].is_fw_input_ready[buffer], device, sizeof(int), stream[2 * device]);
}

void HiddenForwardCommunication(int device, int index, int e)
{
	int num_buffer = (num_layer - device) * 2 - 1;
	int buffer = index % num_buffer;
	int next_buffer;
	if(e == -1)
		next_buffer = (index + 1) % (num_buffer- 2);
	else
		next_buffer = index % (num_buffer - 2);
	cudaSetDevice(device);
	
	// wait for current layer's forward output buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_fw_output_ready, buffer, __LINE__, index, 0, device);
				
	// wait for next layer's forward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_fw_next_input_ready, next_buffer, __LINE__, index, 0, device);

	// copy forward output (activation) to next layer
	cudaMemcpyPeerAsync(layer[device + 1].a[next_buffer], device + 1, layer[device].a_next[buffer], device, num_node_arr[device + 1] * sizeof(double), stream[2 * device + 1]);

	// current layer's forward output buffer is empty
	cudaMemsetAsync(&layer[device].is_fw_output_ready[buffer], 0, sizeof(int), stream[2 * device + 1]);

	// next layer's forward input buffer is full
	cudaMemsetAsync(&layer[device].is_fw_next_input_ready[next_buffer], 1, sizeof(int), stream[2 * device + 1]);
	cudaMemcpyPeerAsync(&layer[device + 1].is_fw_input_ready[next_buffer], device + 1, &layer[device].is_fw_next_input_ready[next_buffer], device, sizeof(int), stream[2 * device + 1]);
}

void HiddenForwardBackwardComputation(int device, int fw_index, int bw_index, int e)
{
	int num_buffer = (num_layer - device) * 2 - 1;
    int fw_buffer = fw_index % num_buffer;
	int bw_buffer = bw_index % num_buffer;

	cudaSetDevice(device);

	// wait for current layer's forward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_fw_input_ready, fw_buffer, __LINE__, fw_index, e, device);

	// wait for current layer's forward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_fw_output_ready, fw_buffer, __LINE__, fw_index, e, device);

	// compute activation
	MatrixMultiply(layer[device].a[fw_buffer], layer[device].weight[fw_buffer], layer[device].a_next[fw_buffer], 1, num_node_arr[device], num_node_arr[device + 1], device);
	if(device != num_layer - 2)
		Sigmoid<<<(num_node_arr[device + 1] + 1023) / 1024, 1024, 0, stream[2 * device]>>>(layer[device].a_next[fw_buffer], num_node_arr[device + 1]);

	// current layer's forward output buffer is full
	cudaMemsetAsync(&layer[device].is_fw_output_ready[fw_buffer], 1, sizeof(int), stream[2 * device]);

	// wait for current layer's backward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_bw_input_ready, bw_buffer, __LINE__, bw_index, e, device);

	// wait for current layer's backward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_bw_output_ready, bw_buffer, __LINE__, bw_index, e, device);

	// compute backward output (delta)
	MatrixMultiply(layer[device].weight[bw_buffer], layer[device].delta_next[bw_buffer], layer[device].delta[bw_buffer], num_node_arr[device], num_node_arr[device + 1], 1, device);
	GetHiddenLayerDelta<<<(num_node_arr[device] + 1023) / 1024, 1024, 0, stream[2 * device]>>>(layer[device].delta[bw_buffer], layer[device].a[bw_buffer], layer[device].weight[bw_buffer], layer[device].delta_next[bw_buffer], num_node_arr[device]);

	// current layer's backward output buffer is full
	cudaMemsetAsync(&layer[device].is_bw_output_ready[bw_buffer], 1, sizeof(int), stream[2 * device]);

	// update weight
	UpdateWeight<<<(num_node_arr[device] * num_node_arr[device + 1] + 1023) / 1024, 1024, 0, stream[2 * device]>>>(layer[device].a[bw_buffer], layer[device].weight[fw_buffer], layer[device].weight[(fw_buffer + 1) % num_buffer], layer[device].weight[bw_buffer], layer[device].delta_next[bw_buffer], num_node_arr[device + 1], num_node_arr[device + 1] * num_node_arr[device], alpha);

	// current layer's forward/backward input buffer is empty
	cudaMemsetAsync(&layer[device].is_fw_input_ready[fw_buffer], 0, sizeof(int), stream[2 * device]);
	cudaMemsetAsync(&layer[device].is_bw_input_ready[bw_buffer], 0, sizeof(int), stream[2 * device]);
	cudaMemcpyPeerAsync(&layer[device - 1].is_fw_next_input_ready[fw_buffer], device - 1, &layer[device].is_fw_input_ready[fw_buffer], device, sizeof(int), stream[2 * device]);
	cudaMemcpyPeerAsync(&layer[device + 1].is_bw_prev_input_ready[bw_buffer], device + 1, &layer[device].is_bw_input_ready[bw_buffer], device, sizeof(int), stream[2 * device]);
}

void HiddenForwardBackwardCommunication(int device, int fw_index, int bw_index, int e)
{
	int num_buffer = (num_layer - device) * 2 - 1;
	int fw_buffer = fw_index % num_buffer;
	int next_buffer = fw_index % (num_buffer - 2);
	int bw_buffer = bw_index % num_buffer;
	int prev_buffer = bw_index % (num_buffer + 2);
	cudaSetDevice(device);

	// wait for current layer's forward output buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_fw_output_ready, fw_buffer, __LINE__, fw_index, e, device);
				
	// wait for next layer's forward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_fw_next_input_ready, next_buffer, __LINE__, fw_index, e, device);

	// copy forward output (activation) to next layer
	cudaMemcpyPeerAsync(layer[device + 1].a[next_buffer], device + 1, layer[device].a_next[fw_buffer], device, num_node_arr[device + 1] * sizeof(double), stream[2 * device + 1]);

	// current layer's forward output buffer is empty
	cudaMemsetAsync(&layer[device].is_fw_output_ready[fw_buffer], 0, sizeof(int), stream[2 * device + 1]);

	// next layer's forward input buffer is full
	cudaMemsetAsync(&layer[device].is_fw_next_input_ready[next_buffer], 1, sizeof(int), stream[2 * device + 1]);
	cudaMemcpyPeerAsync(&layer[device + 1].is_fw_input_ready[next_buffer], device + 1, &layer[device].is_fw_next_input_ready[next_buffer], device, sizeof(int), stream[2 * device + 1]);
	//cudaMemsetAsync(&layer[device + 1].is_fw_input_ready[next_buffer], 1, sizeof(int), stream[2 * device + 1]);

	// wait for current layer's backward output buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_bw_output_ready, bw_buffer, __LINE__, bw_index, e, device);

	// wait for previous layer's backward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_bw_prev_input_ready, prev_buffer, __LINE__, bw_index, e, device);

	// copy backward output (delta) to preious layer
	cudaMemcpyPeerAsync(layer[device - 1].delta_next[prev_buffer], device - 1, layer[device].delta[bw_buffer], device, num_node_arr[device - 1] * sizeof(double), stream[2 * device + 1]);

	// current layer's backward output buffer is empty
	cudaMemsetAsync(&layer[device].is_bw_output_ready[bw_buffer], 0, sizeof(int), stream[2 * device + 1]);

	// previous layer's backward input buffer is full
	cudaMemsetAsync(&layer[device].is_bw_prev_input_ready[prev_buffer], 1, sizeof(int), stream[2 * device + 1]);
	cudaMemcpyPeerAsync(&layer[device - 1].is_bw_input_ready[prev_buffer], device - 1, &layer[device].is_bw_prev_input_ready[prev_buffer], device, sizeof(int), stream[2 * device + 1]);
}

void OutputForwardComputation(int fw_index)
{
	int fw_buffer = 0;
	cudaSetDevice(num_layer - 1);

	// wait for current layer's forward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].is_fw_input_ready, fw_buffer, __LINE__, fw_index, 0, num_layer - 1);

	// get estimated class
	GetMaxIndex<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[fw_buffer], num_node_arr[num_layer - 1], d_max_index);

	// compare with real class
	CheckCorrect<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(result_d, fw_index, d_max_index, d_num_correct);

	// current layer's forward input buffer is empty
	cudaMemsetAsync(&layer[num_layer - 1].is_fw_input_ready[fw_buffer], 0, sizeof(int), stream[2 * (num_layer - 1)]);
	cudaMemcpyPeerAsync(&layer[num_layer - 2].is_fw_next_input_ready[fw_buffer], num_layer - 2, &layer[num_layer - 1].is_fw_input_ready[fw_buffer], num_layer - 1, sizeof(int), stream[2 * (num_layer - 1)]);
}

void OutputForwardBackwardComputation(int fw_index, int bw_index, int e)
{
	int fw_buffer = 0;
	int bw_buffer = 0;
	cudaSetDevice(num_layer - 1);

	// wait for current layer's forward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].is_fw_input_ready, fw_buffer, __LINE__, fw_index, e, num_layer - 1);

	// softmax
	Exponential<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[fw_buffer], num_node_arr[num_layer - 1]);
	GetSum<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[fw_buffer], num_node_arr[num_layer - 1], sum);
	Softmax<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[fw_buffer], sum, num_node_arr[num_layer - 1]);

	// get estimated class
	GetMaxIndex<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[fw_buffer], num_node_arr[num_layer - 1], d_max_index);

	// compare with real class
	CheckCorrect<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(result_d, fw_index, d_max_index, d_train_correct);

	// wait for current layer's backward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].is_bw_output_ready, bw_buffer, __LINE__, bw_index, e, num_layer - 1);
		
	// compute delta
	GetOutputLayerDelta<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[bw_buffer], layer[num_layer - 1].delta[bw_buffer], result_d, bw_index, num_node_arr[num_layer - 1]);

	// current layer's backward output buffer is full
	cudaMemsetAsync(&layer[num_layer - 1].is_bw_output_ready[bw_buffer], 1, sizeof(int), stream[2 * (num_layer - 1)]);

	// current layer's forward input buffer is empty
	cudaMemsetAsync(&layer[num_layer - 1].is_fw_input_ready[fw_buffer], 0, sizeof(int), stream[2 * (num_layer - 1)]);
	cudaMemcpyPeerAsync(&layer[num_layer - 2].is_fw_next_input_ready[fw_buffer], num_layer - 2, &layer[num_layer - 1].is_fw_input_ready[fw_buffer], num_layer - 1, sizeof(int), stream[2 * (num_layer - 1)]);
}

void OutputForwardBackwardCommunication(int fw_index, int bw_index, int e)
{
	int bw_buffer = 0;
	int prev_buffer = bw_index % 3;
	cudaSetDevice(num_layer - 1);

	// wait for previous layer's backward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * num_layer - 1]>>>(layer[num_layer - 1].is_bw_prev_input_ready, prev_buffer, __LINE__, bw_index, e, num_layer - 1);

	// wait for current layer's backward output buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * num_layer - 1]>>>(layer[num_layer - 1].is_bw_output_ready, bw_buffer, __LINE__, bw_index, e, num_layer - 1);

	// copy backward output (delta) to previous layer
	cudaMemcpyPeerAsync(layer[num_layer - 2].delta_next[prev_buffer], num_layer - 2, layer[num_layer - 1].delta[bw_buffer], num_layer - 1, num_node_arr[num_layer - 1] * sizeof(double), stream[2 * num_layer -1]);

	// current layer's backward output buffer is empty
	cudaMemsetAsync(&layer[num_layer - 1].is_bw_output_ready[bw_buffer], 0, sizeof(int), stream[2 * num_layer - 1]);

	// previous layer's backward input buffer is full
	cudaMemsetAsync(&layer[num_layer - 1].is_bw_prev_input_ready[prev_buffer], 1, sizeof(int), stream[2 * num_layer - 1]);
	cudaMemcpyPeerAsync(&layer[num_layer - 2].is_bw_input_ready[prev_buffer], num_layer - 2, &layer[num_layer - 1].is_bw_prev_input_ready[prev_buffer], num_layer - 1, sizeof(int), stream[2 * num_layer - 1]);
}

__global__ void GetSum(double *a, int num_node, double *s)
{
	s[0] = 0.0;
	for(int i = 1; i < num_node; i++)
		s[0] += a[i];
}

__global__ void GetMaxIndex(double *a, int num_node, int *index)
{
	double max = a[1];
	index[0] = 1;
	printf("%f ", a[1]);
	for(int i = 2; i < num_node; i++)
	{
		printf("%f ", a[i]);
		if(a[i] > max)
		{
			max = a[i];
			index[0] = i;
		}
	}
	printf("\n", index[0]);
}
__global__ void CheckCorrect(double *test_result_d, int fw_index, int *max_index, double *num_correct)
{
	printf("%d %d\n", (int)test_result_d[fw_index] ,max_index[0]);
	if((int)test_result_d[fw_index] == max_index[0])
			num_correct[0]++;
}

__global__ void GetOutputLayerDelta(double *output_a, double *output_delta, double *result, int index, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
	{
		if(i != result[index])
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

__global__ void UpdateWeight(double *cur_a, double *cur_weight, double *next_weight, double *delayed_weight, double *next_delta, int W_W, int size, double lr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double DC;
    if(i < size)
    {
        DC = cur_a[i / W_W] * next_delta[i % W_W];
        next_weight[i] = cur_weight[i] + lr * (DC + 0.001 * DC * DC * (cur_weight[i] - delayed_weight[i]));
    }
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

__global__ void Softmax(double *a, double* sum, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		a[i] /= sum[0];
}

__global__ void WaituntilZero(int *ready, int index, int line, int x, int e, int d)
{
	while(ready[index] != 0)
	{
		// printf("line: %d\tindex: %d\tepoch: %d\tdevice: %d\n", line, x, e, d);
	}
}

__global__ void WaituntilOne(int *ready, int index, int line, int x, int e, int d)
{
	while(ready[index] == 0)
	{
		// printf("line: %d\tindex: %d\tepoch: %d\tdevice: %d\n", line, x, e, d);
	}
}

__global__ void SetFlag(int *ready, int index)
{
	ready[index] = 1 - ready[index];
}

__global__ void PrintFw(int device, int index, int e)
{
	printf("device #%d\t fw -> %d\tepoch : %d\n", device, index, e);
}

__global__ void PrintBw(int device, int index, int e)
{
	printf("device #%d\t bw -> %d\tepoch : %d\n", device, index, e);
}

__global__ void PrintInt(int *arr, int size)
{
	for(int i = 0; i < size; i++)
		printf("%d ", arr[i]);
	printf("\n");
}

__global__ void PrintDouble(double *arr, int size)
{
	for(int i = 0; i < size; i++)
		printf("%lf ", arr[i]);
	printf("\n");
}
