#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cublas_v2.h>

#define alpha 0.001

// Debug
__global__ void PrintInt(int *flag, int index, int line);
__global__ void PrintDouble(double *flag, int index, int size, int line);
__global__ void PrintFw(int device, int index);
__global__ void PrintBw(int device, int index);

// Computation
__global__ void GetOutputLayerDelta(double *output_a, double *output_delta, double *result, int index, int size);
__global__ void GetHiddenLayerDelta(double *cur_delta, double *cur_a, double *cur_weight, double *prev_delta, int size);
__global__ void UpdateWeight(double *cur_a, double *cur_weight, double *next_delta, int W_W, int size, double lr);
__global__ void Sigmoid(double *a, int size);
__global__ void Exponential(double *a, int size);
__global__ void Softmax(double *a, double sum, int size);
void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W, int i);

// Scheduling
void InputForwardComputation(int index);
void InputForwardCommunication(int index);
void HiddenForwardComputation(int device, int index);
void HiddenForwardCommunication(int device, int index);
void InputForwardBackwardComputation(int fw_index, int bw_index);
void InputForwardBackwardCommunication(int fw_index, int bw_index);
void HiddenForwardBackwardComputation(int device, int fw_index, int bw_index);
void HiddenForwardBackwardCommunication(int device, int fw_index, int bw_index);
void OutputForwardBackwardComputation(int fw_index, int bw_index);
void OutputForwardBackwardCommunication(int fw_index, int bw_index);

// Synchronization
__global__ void WaituntilZero(int *ready, int index, int line, int x);
__global__ void WaituntilOne(int *ready, int index, int line, int x);
__global__ void SetFlag(int *ready, int index);

// Build Model
void GetResultAndInput();
void SetLayer(int layer_index);

// Main
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
	int *is_fw_output_ready;
	int *is_bw_output_ready;
	int *is_fw_next_input_ready;
	int *is_bw_prev_input_ready;
};

struct layer_info *layer;
cublasHandle_t *handle;
cudaStream_t *stream;

double *input_d, *input_host, *result_d, *result_host, *sum;
int *num_node_arr, *cur_fw, *cur_bw, *e;

int num_layer = 0, num_data = 0, epoch = 0;

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

	sum = (double *)malloc(2 * sizeof(double));
	sum[0] = sum[1] = 0;

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
	
	// train and test
	train_model();

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
	int cur_node = num_node_arr[layer_index], next_node;
	double *weight_host;

	cudaSetDevice(layer_index);
	cudaStreamCreate(&stream[2 * layer_index]);
	cudaStreamCreate(&stream[2 * layer_index + 1]);
	cublasCreate(&handle[layer_index]);
	cublasSetStream(handle[layer_index], stream[2 * layer_index]);

	cudaMallocManaged((void**) &layer[layer_index].is_fw_output_ready, 2 * sizeof(int));
	cudaMallocManaged((void**) &layer[layer_index].is_bw_output_ready, 2 * sizeof(int));
	layer[layer_index].is_fw_output_ready[0] = layer[layer_index].is_fw_output_ready[1] = 0;
	layer[layer_index].is_bw_output_ready[0] = layer[layer_index].is_bw_output_ready[1] = 0;

	cudaMallocManaged((void**) &layer[layer_index].is_fw_input_ready, 2 * sizeof(int));
	cudaMallocManaged((void**) &layer[layer_index].is_bw_input_ready, 2 * sizeof(int));
	layer[layer_index].is_fw_input_ready[0] = layer[layer_index].is_fw_input_ready[1] = 0;
	layer[layer_index].is_bw_input_ready[0] = layer[layer_index].is_bw_input_ready[1] = 0;

	cudaMallocManaged((void**) &layer[layer_index].is_fw_next_input_ready, 2 * sizeof(int));
	cudaMallocManaged((void**) &layer[layer_index].is_bw_prev_input_ready, 2 * sizeof(int));
	layer[layer_index].is_fw_next_input_ready[0] = layer[layer_index].is_fw_next_input_ready[1] = 0;
	layer[layer_index].is_bw_prev_input_ready[0] = layer[layer_index].is_bw_prev_input_ready[1] = 0;
	
	layer[layer_index].a = (double **)malloc(2 * sizeof(double *));
	for(int i = 0; i < 2; i++)
		cudaMalloc((void**) &layer[layer_index].a[i], num_node_arr[layer_index] * sizeof(double));

	// except input layer
	if(layer_index != 0)
	{
		layer[layer_index].delta = (double **)malloc(2 * sizeof(double *));
		for(int i = 0; i < 2; i++)
			cudaMalloc((void**) &layer[layer_index].delta[i], num_node_arr[layer_index] * sizeof(double));
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
	cudaMalloc((void**) &result_d, num_data * sizeof(int));
	cudaMemcpy(result_d, result_host, num_data * sizeof(int), cudaMemcpyHostToDevice);

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
				InputForwardComputation(cur_fw[i]);
				InputForwardCommunication(cur_fw[i]++);
			}
		}
		else
		{
			while(cur_fw[i] < 2 * (num_layer  - 1 - i))
			{
				HiddenForwardComputation(i, cur_fw[i]);
				HiddenForwardCommunication(i, cur_fw[i]++);
			}
		}
	}

	
	for(int i = 1; i < num_layer; i++)
	{
		if(i == num_layer - 1)
		{
			while(cur_fw[i] < i)
			{
				OutputForwardBackwardComputation(cur_fw[i], cur_bw[i]);
				OutputForwardBackwardCommunication(cur_fw[i]++, cur_bw[i]++);
			}
		}
		else
		{
			while(cur_fw[i] < i + 2 * (num_layer  - 1 - i))
			{
				HiddenForwardBackwardComputation(i, cur_fw[i], cur_bw[i]);
				HiddenForwardBackwardCommunication(i, cur_fw[i]++, cur_bw[i]++);
			}
		}
	}
	
	for(int i = 0; i < num_layer; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
	
	// steady stage		epoch 홀수 or 짝수 -> cur_Bf = index % 2 or 1 - (index % 2)
	for(int i = 0; i < num_layer; i++)
	{
		if(i == 0)
		{
			while(e[i] < epoch - 1)
			{
				InputForwardBackwardComputation(cur_fw[i], cur_bw[i]++);
				InputForwardCommunication(cur_fw[i]++);
				if(cur_fw[i] == num_data)
					cur_fw[i] = 0;
				if(cur_bw[i] == num_data)
				{
					e[i]++;
					cur_bw[i] = 0;
				}
			}

			// while(cur_bw[i] < num_data - 2)
			// {
			// 	InputForwardBackwardComputation(cur_fw[i], cur_bw[i]++);
			// 	InputForwardCommunication(cur_fw[i]++);
			// 	if(cur_fw[i] == num_data)
			// 		cur_fw[i] = 0;
			// }
		}
		else if(i == num_layer - 1)
		{
			while(e[i] < epoch - 1)
			// while(e[i] < epoch)
			{
				OutputForwardBackwardComputation(cur_fw[i], cur_bw[i]);
				OutputForwardBackwardCommunication(cur_fw[i]++, cur_bw[i]++);
				if(cur_fw[i] == num_data)
					cur_fw[i] = 0;
				if(cur_bw[i] == num_data)
				{
					e[i]++;
					cur_bw[i] = 0;
				}
			}
		}
		else
		{
			while(e[i] < epoch - 1)
			{
				HiddenForwardBackwardComputation(i, cur_fw[i], cur_bw[i]);
				HiddenForwardBackwardCommunication(i, cur_fw[i]++, cur_bw[i]++);
				if(cur_fw[i] == num_data)
					cur_fw[i] = 0;
				if(cur_bw[i] == num_data)
				{
					e[i]++;
					cur_bw[i] = 0;
				}
			}

			// while(cur_bw[i] < num_data - 2)
			// {
			// 	HiddenForwardBackwardComputation(i, cur_fw[i], cur_bw[i]);
			// 	HiddenForwardBackwardCommunication(i, cur_fw[i]++, cur_bw[i]++);
			// 	if(cur_fw[i] == num_data)
			// 		cur_fw[i] = 0;
			// }
		}
	}

	/*
	// end stage
	for(int i = 0; i < num_layer; i++)
	{
		if(i == 0)
		{
			while(cur_bw[i] <= num_data)
			{
				InputBackwardComputation(cur_fw[i], cur_bw[i]);
				InputBackwardCommunication(cur_fw[i]++, cur_bw[i]++);
			}
		}
		else if(i == num_layer - 1)
		{
			while(cur_bw[i] <= num_data)
			{
				OutputBackwardComputation(cur_fw[i], cur_bw[i]);
				OutputBackwardCommunication(cur_fw[i]++, cur_bw[i]++);
			}
		}
		else
		{
			while(cur_bw[i] <= num_data)
			{
				HiddenBackwardComputation(cur_fw[i], cur_bw[i]);
				HiddenBackwardCommunication(cur_fw[i]++, cur_bw[i]++);
			}
		}
	}
	*/
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

void InputForwardComputation(int index)
{
	int buffer = index % 2;
	cudaSetDevice(0);
	
	// copy input data
	cudaMemcpyAsync(layer[0].a[buffer], input_d + (index * num_node_arr[0]) * sizeof(int), num_node_arr[0] * sizeof(double), cudaMemcpyDeviceToDevice, stream[0]);
	
	// wait for current layer's forward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[0]>>>(layer[0].is_fw_output_ready, buffer, __LINE__, index);
	
	// compute activation
	MatrixMultiply(layer[0].a[buffer], layer[0].weight, layer[0].a_next[buffer], 1, num_node_arr[0], num_node_arr[1], 0);
	Sigmoid<<<(num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a_next[buffer], num_node_arr[1]);
	
	// current layer's forward output buffer is full
	cudaMemsetAsync(&layer[0].is_fw_output_ready[buffer], 1, sizeof(int), stream[0]);
}

void InputForwardCommunication(int index)
{
	int buffer = index % 2;
	cudaSetDevice(0);

	// wait for current layer's forward output buffer is full
	WaituntilOne<<<1, 1, 0, stream[1]>>>(layer[0].is_fw_output_ready, buffer, __LINE__, index);
	
	// wait for next layer's forward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[1]>>>(layer[0].is_fw_next_input_ready, buffer, __LINE__, index);

	// copy foward output (activation) to next layer
	cudaMemcpyPeerAsync(layer[1].a[buffer], 1, layer[0].a_next[buffer], 0, num_node_arr[1] * sizeof(double), stream[1]);
	
	// next layer's forward input buffer is full
	cudaMemsetAsync(&layer[1].is_fw_input_ready[buffer], 1, sizeof(int), stream[1]);
	cudaMemsetAsync(&layer[0].is_fw_next_input_ready[buffer], 1, sizeof(int), stream[1]);

	// current layer's forward output buffer is empty
	cudaMemsetAsync(&layer[0].is_fw_output_ready[buffer], 0, sizeof(int), stream[1]);
}

void InputForwardBackwardComputation(int fw_index, int bw_index)
{
	int fw_buffer = fw_index % 2;
	int bw_buffer = bw_index % 2;
	cudaSetDevice(0);

	// copy input data
	cudaMemcpyAsync(layer[0].a[fw_buffer], input_d + (fw_index * num_node_arr[0]) * sizeof(int), num_node_arr[0] * sizeof(double), cudaMemcpyDeviceToDevice, stream[0]);
	
	// wait for current layer's forward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[0]>>>(layer[0].is_fw_output_ready, fw_buffer, __LINE__, fw_index);
	
	// compute activation
	MatrixMultiply(layer[0].a[fw_buffer], layer[0].weight, layer[0].a_next[fw_buffer], 1, num_node_arr[0], num_node_arr[1], 0);
	Sigmoid<<<(num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a_next[fw_buffer], num_node_arr[1]);
	
	// current layer's forward output buffer is full
	cudaMemsetAsync(&layer[0].is_fw_output_ready[fw_buffer], 1, sizeof(int), stream[0]);

	// wait for current layer's backward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[0]>>>(layer[0].is_bw_input_ready, bw_buffer, __LINE__, bw_index);

	// update weight
	UpdateWeight<<<(num_node_arr[0] * num_node_arr[1] + 1023) / 1024, 1024, 0, stream[0]>>>(layer[0].a[bw_buffer], layer[0].weight, layer[0].delta_next[bw_buffer], num_node_arr[1], num_node_arr[1] * num_node_arr[0], alpha);

	// current layer's backward input buffer is empty
	cudaMemsetAsync(&layer[0].is_bw_input_ready[bw_buffer], 0, sizeof(int), stream[0]);
	cudaMemsetAsync(&layer[1].is_bw_prev_input_ready[bw_buffer], 0, sizeof(int), stream[0]);
}

void HiddenForwardComputation(int device, int index)
{
	int buffer = index % 2;
	cudaSetDevice(device);

	// wait for current layer's forward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_fw_input_ready, buffer, __LINE__, index);

	// wait for current layer's forward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_fw_output_ready, buffer, __LINE__, index);

	// compute activation
	MatrixMultiply(layer[device].a[buffer], layer[device].weight, layer[device].a_next[buffer], 1, num_node_arr[device], num_node_arr[device + 1], device);
	if(device != num_layer - 2)
		Sigmoid<<<(num_node_arr[device + 1] + 1023) / 1024, 1024, 0, stream[2 * device]>>>(layer[device].a_next[buffer], num_node_arr[device + 1]);

	// current layer's forward output buffer is full
	cudaMemsetAsync(&layer[device].is_fw_output_ready[buffer], 1, sizeof(int), stream[2 * device]);

	// current layer's forward input buffer is empty
	cudaMemsetAsync(&layer[device].is_fw_input_ready[buffer], 0, sizeof(int), stream[2 * device]);
	cudaMemsetAsync(&layer[device - 1].is_fw_next_input_ready[buffer], 0, sizeof(int), stream[2 * device]);
}

void HiddenForwardCommunication(int device, int index)
{
	int buffer = index % 2;
	cudaSetDevice(device);
	
	// wait for current layer's forward output buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_fw_output_ready, buffer, __LINE__, index);
				
	// wait for next layer's forward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_fw_next_input_ready, buffer, __LINE__, index);

	// copy forward output (activation) to next layer
	cudaMemcpyPeerAsync(layer[device + 1].a[buffer], device + 1, layer[device].a_next[buffer], device, num_node_arr[device + 1] * sizeof(double), stream[2 * device + 1]);

	// next layer's forward input buffer is full
	cudaMemsetAsync(&layer[device + 1].is_fw_input_ready[buffer], 1, sizeof(int), stream[2 * device + 1]);
	cudaMemsetAsync(&layer[device].is_fw_next_input_ready[buffer], 1, sizeof(int), stream[2 * device + 1]);

	// current layer's forward output buffer is empty
	cudaMemsetAsync(&layer[device].is_fw_output_ready[buffer], 0, sizeof(int), stream[2 * device + 1]);
}

void HiddenForwardBackwardComputation(int device, int fw_index, int bw_index)
{
	int fw_buffer = fw_index % 2;
	int bw_buffer = bw_index % 2;
	cudaSetDevice(device);

	// wait for current layer's forward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_fw_input_ready, fw_buffer, __LINE__, fw_index);

	// wait for current layer's forward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_fw_output_ready, fw_buffer, __LINE__, fw_index);

	// compute activation
	MatrixMultiply(layer[device].a[fw_buffer], layer[device].weight, layer[device].a_next[fw_buffer], 1, num_node_arr[device], num_node_arr[device + 1], device);
	if(device != num_layer - 2)
		Sigmoid<<<(num_node_arr[device + 1] + 1023) / 1024, 1024, 0, stream[2 * device]>>>(layer[device].a_next[fw_buffer], num_node_arr[device + 1]);

	// current layer's forward output buffer is full
	cudaMemsetAsync(&layer[device].is_fw_output_ready[fw_buffer], 1, sizeof(int), stream[2 * device]);

	// wait for current layer's backward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_bw_input_ready, bw_buffer, __LINE__, bw_index);

	// wait for current layer's backward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device]>>>(layer[device].is_bw_output_ready, bw_buffer, __LINE__, bw_index);

	// compute backward output (delta)
	MatrixMultiply(layer[device].weight, layer[device].delta_next[bw_buffer], layer[device].delta[bw_buffer], num_node_arr[device], num_node_arr[device + 1], 1, device);
	GetHiddenLayerDelta<<<(num_node_arr[device] + 1023) / 1024, 1024, 0, stream[2 * device]>>>(layer[device].delta[bw_buffer], layer[device].a[bw_buffer], layer[device].weight, layer[device].delta_next[bw_buffer], num_node_arr[device]);

	// current layer's backward output buffer is full
	cudaMemsetAsync(&layer[device].is_bw_output_ready[bw_buffer], 1, sizeof(int), stream[2 * device]);

	// update weight
	UpdateWeight<<<(num_node_arr[device] * num_node_arr[device + 1] + 1023) / 1024, 1024, 0, stream[2 * device]>>>(layer[device].a[bw_buffer], layer[device].weight, layer[device].delta_next[bw_buffer], num_node_arr[device + 1], num_node_arr[device + 1] * num_node_arr[device], alpha);

	// current layer's forward input buffer is empty
	cudaMemsetAsync(&layer[device].is_fw_input_ready[fw_buffer], 0, sizeof(int), stream[2 * device]);
	cudaMemsetAsync(&layer[device - 1].is_fw_next_input_ready[fw_buffer], 0, sizeof(int), stream[2 * device]);

	// current layer's backward input buffer is empty
	cudaMemsetAsync(&layer[device].is_bw_input_ready[bw_buffer], 0, sizeof(int), stream[2 * device]);
	cudaMemsetAsync(&layer[device + 1].is_bw_prev_input_ready[bw_buffer], 0, sizeof(int), stream[2 * device]);
}

void HiddenForwardBackwardCommunication(int device, int fw_index, int bw_index)
{
	int fw_buffer = fw_index % 2;
	int bw_buffer = bw_index % 2;
	cudaSetDevice(device);

	// wait for current layer's forward output buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_fw_output_ready, fw_buffer, __LINE__, fw_index);
				
	// wait for next layer's forward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_fw_next_input_ready, fw_buffer, __LINE__, fw_index);

	// copy forward output (activation) to next layer
	cudaMemcpyPeerAsync(layer[device + 1].a[fw_buffer], device + 1, layer[device].a_next[fw_buffer], device, num_node_arr[device + 1] * sizeof(double), stream[2 * device + 1]);

	// next layer's forward input buffer is full
	cudaMemsetAsync(&layer[device + 1].is_fw_input_ready[fw_buffer], 1, sizeof(int), stream[2 * device + 1]);
	cudaMemsetAsync(&layer[device].is_fw_next_input_ready[fw_buffer], 1, sizeof(int), stream[2 * device + 1]);

	// current layer's forward output buffer is empty
	cudaMemsetAsync(&layer[device].is_fw_output_ready[fw_buffer], 0, sizeof(int), stream[2 * device + 1]);

	// wait for current layer's backward output buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_bw_output_ready, bw_buffer, __LINE__, bw_index);

	// wait for previous layer's backward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * device + 1]>>>(layer[device].is_bw_prev_input_ready, bw_buffer, __LINE__, bw_index);

	// copy backward output (delta) to preious layer
	cudaMemcpyPeerAsync(layer[device - 1].delta_next[bw_buffer], device - 1, layer[device].delta[bw_buffer], device, num_node_arr[device - 1] * sizeof(double), stream[2 * device + 1]);

	// previous layer's backward input buffer is full
	cudaMemsetAsync(&layer[device - 1].is_bw_input_ready[bw_buffer], 1, sizeof(int), stream[2 * device + 1]);
	cudaMemsetAsync(&layer[device].is_bw_prev_input_ready[bw_buffer], 1, sizeof(int), stream[2 * device + 1]);

	// current layer's backward output buffer is empty
	cudaMemsetAsync(&layer[device].is_bw_output_ready[bw_buffer], 0, sizeof(int), stream[2 * device + 1]);
}

void OutputForwardBackwardComputation(int fw_index, int bw_index)
{
	int fw_buffer = fw_index % 2;
	int bw_buffer = bw_index % 2;
	cudaSetDevice(num_layer - 1);

	// wait for current layer's forward input buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].is_fw_input_ready, fw_buffer, __LINE__, fw_index);

	// softmax
	Exponential<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[fw_buffer], num_node_arr[num_layer - 1]);
	cublasDasum(handle[num_layer - 1], num_node_arr[num_layer - 1] - 1, layer[num_layer - 1].a[fw_buffer], 1, &sum[fw_buffer]);
	Softmax<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[fw_buffer], sum[fw_buffer], num_node_arr[num_layer - 1]);

	// wait for current layer's backward output buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].is_bw_output_ready, bw_buffer, __LINE__, bw_index);
		
	// compute delta
	GetOutputLayerDelta<<<(num_node_arr[num_layer - 1] + 1023) / 1024, 1024, 0, stream[2 * (num_layer - 1)]>>>(layer[num_layer - 1].a[bw_buffer], layer[num_layer - 1].delta[bw_buffer], result_d, bw_index, num_node_arr[num_layer - 1]);

	// current layer's backward output buffer is full
	cudaMemsetAsync(&layer[num_layer - 1].is_bw_output_ready[bw_buffer], 1, sizeof(int), stream[2 * (num_layer - 1)]);

	// current layer's forward input buffer is empty
	cudaMemsetAsync(&layer[num_layer - 1].is_fw_input_ready[fw_buffer], 0, sizeof(int), stream[2 * (num_layer - 1)]);
	cudaMemsetAsync(&layer[num_layer - 2].is_fw_next_input_ready[fw_buffer], 0, sizeof(int), stream[2 * (num_layer - 1)]);
}

void OutputForwardBackwardCommunication(int fw_index, int bw_index)
{
	int bw_buffer = bw_index % 2;
	cudaSetDevice(num_layer - 1);

	// wait for previous layer's backward input buffer is empty
	WaituntilZero<<<1, 1, 0, stream[2 * num_layer - 1]>>>(layer[num_layer - 1].is_bw_prev_input_ready, bw_buffer, __LINE__, bw_index);

	// wait for current layer's backward output buffer is full
	WaituntilOne<<<1, 1, 0, stream[2 * num_layer - 1]>>>(layer[num_layer - 1].is_bw_output_ready, bw_buffer, __LINE__, bw_index);

	// copy backward output (delta) to previous layer
	cudaMemcpyPeerAsync(layer[num_layer - 2].delta_next[bw_buffer], num_layer - 2, layer[num_layer - 1].delta[bw_buffer], num_layer - 1, num_node_arr[num_layer - 1] * sizeof(double), stream[2 * num_layer -1]);

	// previous layer's backward input buffer is full
	cudaMemsetAsync(&layer[num_layer - 2].is_bw_input_ready[bw_buffer], 1, sizeof(int), stream[2 * num_layer - 1]);
	cudaMemsetAsync(&layer[num_layer - 1].is_bw_prev_input_ready[bw_buffer], 1, sizeof(int), stream[2 * num_layer - 1]);

	// current layer's backward output buffer is empty
	cudaMemsetAsync(&layer[num_layer - 1].is_bw_output_ready[bw_buffer], 0, sizeof(int), stream[2 * num_layer - 1]);
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

__global__ void WaituntilZero(int *ready, int index, int line, int x)
{
	while(ready[index] == 1)
	{
		printf("%d\t%d\n", line, x);
	}
}

__global__ void WaituntilOne(int *ready, int index, int line, int x)
{
	while(ready[index] == 0)
	{
		printf("%d\t%d\n", line, x);
	}
}

__global__ void SetFlag(int *ready, int index)
{
	ready[index] = 1 - ready[index];
}

__global__ void PrintFw(int device, int index)
{
	printf("device #%d\t fw -> %d\n", device, index);
}

__global__ void PrintBw(int device, int index)
{
	printf("device #%d\t bw -> %d\n", device, index);
}

__global__ void PrintInt(int *flag, int index, int line)
{
	printf("line #%d\t index #%d\t %d\n", line, index, flag[index]);
}

__global__ void PrintDouble(double *flag, int index, int size, int line)
{
	printf("line #%d\t", line);
	for(int i = 0; i < size; i++)
		printf("%lf ", flag[index + i]);
	printf("\n");
}
