#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#define alpha 0.001
//	weight = cur_node * next_node
//  atomicCAS : (old == compare) ? val : old		aotmicCAS(old, compare, val)

__device__ void GetOutputLayerDelta(double *output_a, double *output_delta, double *result, int size);
__device__ void GetHiddenLayerDelta(double *cur_delta, double *cur_a, double *cur_weight, double *prev_delta, int size);
__device__ void UpdateWeight(double *cur_a, double *cur_weight, double *next_delta, int W_W, int size, double lr);
__device__ void Sigmoid(double *a, int size);
__device__ void Exponential(double *a, int size);
__device__ void Softmax(double *a, double sum, int size);
__global__ void StartStageInputlayer(double *input, double **cur_a, double **cur_next_a, double **cur_weight, double **next_a, double **next_delta, int *next_fw_ready, int *cur_bw_ready, int cur_node, int next_node, int num_layer, double lr);
__global__ void StartStageHiddenlayer(double **cur_a, double **cur_next_a, double **cur_weight, double **next_a, double **cur_delta, double **next_delta, double **prev_next_delta, int *cur_fw_ready, int *next_fw_ready, int *prev_bw_ready, int *cur_bw_ready, int cur_node, int next_node, int num_layer, int layer_index, double lr);
__global__ void StartStageOutputlayer(double **cur_a, double **cur_delta, double **prev_next_delta, int *prev_bw_ready, int *cur_fw_ready, int *result, int cur_node, int num_layer);
__global__ void SteadyStageInputlayer(double *input, double **cur_a, double **cur_next_a, double **cur_weight, double **next_a, double **next_delta, int *next_fw_ready, int *cur_bw_ready, int cur_node, int next_node, int num_layer, double lr, int epoch, int num_data);
__global__ void SteadyStageHiddenlayer(double **cur_a, double **cur_next_a, double **cur_weight, double **next_a, double **cur_delta, double **cur_next_delta, double **prev_next_delta, int *cur_fw_ready, int *next_fw_ready, int *prev_bw_ready, int *cur_bw_ready, int cur_node, int next_node, int num_layer, int layer_index, double lr, int epoch, int num_data);
__global__ void SteadyStageOutputlayer(double **cur_a, double **cur_delta, double **prev_next_delta, int *prev_bw_ready, int *cur_fw_ready, int *result, int cur_node, int num_layer, int epoch, int num_data);
void StartStage();
void SteadyStage(int e);
void SetInputAndResult();
void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W, int i);
void SetParameter(int layer_index);
void train_model();
void test_accuracy();

struct layer_info {
	double **a;
	double **weight;
	double **delta;
	double **delta_next;
	double **a_next;
	int *fw_ready;
	int *bw_ready;
	int *cur_fw;
	int *cur_bw;
};

struct layer_info *layer_h;
cublasHandle_t *handle_h;
cudaStream_t *stream_h;

double *input_d;
int *result_d;
int *num_node_arr;
int num_layer_h, num_data_h, epoch;

clock_t start;


int main() {

	// input model's information
	printf("number of layers : ");
	scanf(" %d", &num_layer_h);

	num_node_arr = (int *) malloc(num_layer_h * sizeof(int));
	printf("number of nodes : ");
	for(int i = 0; i < num_layer_h; i++)
		scanf(" %d", &num_node_arr[i]);
	for(int i = 0; i < num_layer_h; i++)
		num_node_arr[i]++;

	printf("number of data : ");
	scanf(" %d", &num_data_h);
	
	// make tool
	handle_h = (cublasHandle_t *) malloc(num_layer_h * sizeof(cublasHandle_t));
	stream_h = (cudaStream_t *) malloc(num_layer_h * sizeof(cudaStream_t));
	for(int i = 0; i < num_layer_h - 1; i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&stream_h[i]);
		cublasCreate(&handle_h[i]);
		cublasSetStream(handle_h[i], stream_h[i]);
	}

	// enable peer access
	for(int i = 0; i < num_layer_h - 1; i++)
	{
		cudaSetDevice(i);
		cudaDeviceEnablePeerAccess(i + 1, 0);
	}
	for(int i = num_layer_h - 1; i > 0; i--)
	{
		cudaSetDevice(i);
		cudaDeviceEnablePeerAccess(i - 1, 0);
	}
	
	// build model
	cudaSetDevice(0);
	cudaMallocManaged((void**) &layer_h, sizeof(struct layer_info) * num_layer_h);
	for(int i = 0; i < num_layer_h; i++)
		SetParameter(i);

	SetInputAndResult();
	cudaDeviceSynchronize();

	// train and test
	printf("epoch : ");
	scanf(" %d", &epoch);
	StartStage();
	SteadyStage(epoch);

	return 0;
}

void SetParameter(int layer_index)
{
	int cur_node = num_node_arr[layer_index], next_node;

	cudaSetDevice(layer_index);
	cudaMallocManaged((void**) &layer_h[layer_index], sizeof(struct layer_info));

	cudaMallocManaged((void**) &layer_h[layer_index].fw_ready, num_layer_h * sizeof(int));
	cudaMallocManaged((void**) &layer_h[layer_index].bw_ready, num_layer_h * sizeof(int));
	for(int i = 0; i < num_layer_h; i++)
		layer_h[layer_index].fw_ready[i] = layer_h[layer_index].bw_ready[i] = 0;

	cudaMallocManaged((void**) &layer_h[layer_index].cur_fw, sizeof(int));
	cudaMallocManaged((void**) &layer_h[layer_index].cur_bw, sizeof(int));
	layer_h[layer_index].cur_fw[0] = layer_h[layer_index].cur_bw[0] = 0;

	cudaMallocManaged((void**) &layer_h[layer_index].a, num_layer_h * sizeof(double *));
	for(int i = 0; i < num_layer_h; i++)
		cudaMallocManaged((void**) &layer_h[layer_index].a[i], cur_node * sizeof(double));

	// except input layer
	if(layer_index != 0)
	{
		cudaMallocManaged((void**) &layer_h[layer_index].delta, num_layer_h * sizeof(double *));
		for(int i = 0; i < num_layer_h; i++)
			cudaMallocManaged((void**) &layer_h[layer_index].delta[i], cur_node * sizeof(double));
	}

	// except output layer
	if(layer_index < num_layer_h - 1)
	{
		next_node = num_node_arr[layer_index + 1];

		cudaMallocManaged((void**) &layer_h[layer_index].a_next, num_layer_h * sizeof(double *));
		cudaMallocManaged((void**) &layer_h[layer_index].delta_next, num_layer_h * sizeof(double *));
		for(int i = 0; i < num_layer_h; i++)
		{
			cudaMallocManaged((void**) &layer_h[layer_index].a_next[i], next_node * sizeof(double));
			cudaMallocManaged((void**) &layer_h[layer_index].delta_next[i], next_node * sizeof(double));
		}

		cudaMallocManaged((void**) &layer_h[layer_index].weight, num_layer_h * sizeof(double *));
		for(int i = 0; i < num_layer_h; i++)
		{
			cudaMallocManaged((void**) &layer_h[layer_index].weight[i], cur_node * next_node * sizeof(double));
			for (int j = 0; j < cur_node * next_node; j++)
				layer_h[layer_index].weight[i][j] = sqrt(6.0 / (cur_node + next_node)) * (rand() / (double)RAND_MAX * 2.0 - 1.0);
		}
	}
}

void SetInputAndResult()
{
	FILE* pFile = NULL;
	char str_tmp[num_node_arr[0] * 3], *p;
	// cudaStream_t dum;

	pFile = fopen("mnist_train_100.csv", "r");
	cudaSetDevice(0);
	cudaMallocManaged((void**) &input_d, num_data_h * num_node_arr[0] * sizeof(double));
	cudaSetDevice(num_layer_h - 1);
	cudaMallocManaged((void**) &result_d, num_data_h * sizeof(int));
	
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
				result_d[r_index++] = num_node_arr[num_layer_h - 1] - 1;
			else
				result_d[r_index++] = atoi(p);

			// set input
			input_d[i_index++] = 1.0;
			for (int i = 1; i < num_node_arr[0]; i++)
			{
				p = strtok(NULL, ",");
				input_d[i_index++] = atof(p) / 255.0;
			}
        }       
	}
}

void StartStage()
{
	cudaSetDevice(0);
	StartStageInputlayer<<<(num_node_arr[0] * num_node_arr[1] + 1023) / 1024, 1024, 0, stream_h[0]>>>(input_d, layer_h[0].a, layer_h[0].a_next, layer_h[0].weight, layer_h[1].a, layer_h[0].delta_next, layer_h[1].fw_ready, layer_h[0].bw_ready, num_node_arr[0], num_node_arr[1], num_layer_h, alpha);

	for(int i = 1; i < num_layer_h - 1; i++)
	{
		cudaSetDevice(i);
		StartStageHiddenlayer<<<(num_node_arr[i] * num_node_arr[i + 1] + 1023) / 1024, 1024, 0, stream_h[i]>>>(layer_h[i].a, layer_h[i].a_next, layer_h[i].weight, layer_h[i + 1].a, layer_h[i].delta, layer_h[i].delta_next, layer_h[i - 1].delta_next, layer_h[i].fw_ready, layer_h[i + 1].fw_ready, layer_h[i - 1].bw_ready, layer_h[i].bw_ready, num_node_arr[i], num_node_arr[i + 1], num_layer_h, i, alpha);
	}
	cudaSetDevice(num_layer_h - 1);
	StartStageOutputlayer<<<(num_node_arr[num_layer_h - 1] + 1023) / 1024, 1024, 0, stream_h[num_layer_h - 1]>>>(layer_h[num_layer_h - 1].a, layer_h[num_layer_h - 1].delta, layer_h[num_layer_h - 2].delta_next, layer_h[num_layer_h - 2].bw_ready, layer_h[num_layer_h - 1].fw_ready, result_d, num_node_arr[num_layer_h - 1], num_layer_h);
	
	for(int i = 0; i < num_layer_h; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
}

void SteadyStage(int e)
{
	cudaSetDevice(0);
	SteadyStageInputlayer<<<(num_node_arr[0] * num_node_arr[1] + 1023) / 1024, 1024, 0, stream_h[0]>>>(input_d, layer_h[0].a, layer_h[0].a_next, layer_h[0].weight, layer_h[1].a, layer_h[0].delta_next, layer_h[1].fw_ready, layer_h[0].bw_ready, num_node_arr[0], num_node_arr[1], num_layer_h, alpha, e, num_data_h);

	for(int i = 1; i < num_layer_h - 1; i++)
	{
		cudaSetDevice(i);
		SteadyStageHiddenlayer<<<(num_node_arr[i] * num_node_arr[i + 1] + 1023) / 1024, 1024, 0, stream_h[i]>>>(layer_h[i].a, layer_h[i].a_next, layer_h[i].weight, layer_h[i + 1].a, layer_h[i].delta, layer_h[i].delta_next, layer_h[i - 1].delta_next, layer_h[i].fw_ready, layer_h[i + 1].fw_ready, layer_h[i - 1].bw_ready, layer_h[i].bw_ready, num_node_arr[i], num_node_arr[i + 1], num_layer_h, i, alpha, e, num_data_h);
	}

	cudaSetDevice(num_layer_h - 1);
	SteadyStageOutputlayer<<<(num_node_arr[num_layer_h - 1] + 1023) / 1024, 1024, 0, stream_h[num_layer_h - 1]>>>(layer_h[num_layer_h - 1].a, layer_h[num_layer_h - 1].delta, layer_h[num_layer_h - 2].delta_next, layer_h[num_layer_h - 2].bw_ready, layer_h[num_layer_h - 1].fw_ready, result_d, num_node_arr[num_layer_h - 1], num_layer_h, e, num_data_h);
	
	for(int i = 0; i < num_layer_h; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
}

__global__ void SteadyStageInputlayer(double *input, double **cur_a, double **cur_next_a, double **cur_weight, double **next_a, double **next_delta, int *next_fw_ready, int *cur_bw_ready, int cur_node, int next_node, int num_layer, double lr, int epoch, int num_data)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int fw_index = num_layer , bw_index = 1, cur_epoch = 0, j;
	// cur index = (num_layer, 1)
	// data index , cur_epoch
		
	while(cur_epoch < epoch)
	{
		j = fw_index % num_layer;
		if(idx < cur_node)
			cur_a[j][idx] = input[j * cur_node + idx];
		
		__syncthreads();

		if(idx < next_node)
		{
			// MatrixMultiply
			cur_next_a[j][idx] = 0;
			for(int i = 0; i < cur_node; i++)
				cur_next_a[j][idx] += cur_a[j][i] * cur_weight[j][idx * cur_node + i];
			
			// Sigmoid
			cur_next_a[j][idx] = 1.0 / (1.0 + exp(-cur_next_a[j][idx]));

			// Wait until next layer finishes task
			if(idx == 0)
			 	while( atomicCAS (next_fw_ready + j , 1 , 1 ) == 1);

			__syncthreads();

			// cudaMemcpyPeer
			if(idx == 0)
				next_a[j][idx] = 1.0;
			else
				next_a[j][idx] = cur_next_a[j][idx];

			__syncthreads();

			if(idx == 0)
			{
				next_fw_ready[j] = 1;
				printf("input layer forward epoch and #data: %d , %d\n", cur_epoch, fw_index);
				if(++fw_index == num_data)
					fw_index = 0;
			}
		} // forward

		__syncthreads();

		j = bw_index % num_layer;
		
		if(idx < cur_node * next_node)
		{
			// Wait until next layer sends delta
			if(idx == 0)
				while(atomicCAS(cur_bw_ready + j, 1, 1) != 1);

			__syncthreads();

			cur_weight[j][idx] = cur_weight[j][idx] + lr * cur_a[j][idx % cur_node] * next_delta[j][idx / cur_node];
			
			__syncthreads();

			if(idx == 0) {
				cur_bw_ready[j] = 0;
				printf("input layer backward epoch and #data: %d , %d\n", cur_epoch, bw_index);
				if(++bw_index == num_data)
				{
					bw_index = 0;
					cur_epoch++;
				}
			}
		}
	}

	return;
}

__global__ void SteadyStageHiddenlayer(double **cur_a, double **cur_next_a, double **cur_weight, double **next_a, double **cur_delta, double **cur_next_delta, double **prev_next_delta, int *cur_fw_ready, int *next_fw_ready, int *prev_bw_ready, int *cur_bw_ready, int cur_node, int next_node, int num_layer, int layer_index, double lr, int epoch, int num_data)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int fw_index = num_layer , bw_index = layer_index + 1, cur_epoch = 0, j;

	while(cur_epoch < epoch)
	{
		j = fw_index % num_layer;
		if(idx == 0)
			while(atomicCAS(cur_fw_ready + j, 1, 1) != 1);

		__syncthreads();
		
		if(idx < next_node)
		{
			// MatrixMultiply
			cur_next_a[j][idx] = 0.0;
			for(int i = 0; i < cur_node; i++)
				cur_next_a[j][idx] += cur_a[j][i] * cur_weight[j][idx * cur_node + i];
			
			// Sigmoid
			cur_next_a[j][idx] = 1.0 / (1.0 + exp(-cur_next_a[j][idx]));

			// Wait until next layer finishes task
			if(idx == 0)
			 	while( atomicCAS ( next_fw_ready + j , 1 , 1 ) == 1);

			__syncthreads();

			// cudaMemcpyPeer
			if(idx == 0)
				next_a[j][idx] = 1.0;
			else
				next_a[j][idx] = cur_next_a[j][idx];

			__syncthreads();

			if(idx == 0)
			{
				next_fw_ready[j] = 1;
				printf("hidden layer %d forward epoch and #data: %d , %d\n", layer_index, cur_epoch, fw_index);
				if(++fw_index == num_data)
					fw_index = 0;
			}
		} // forward

		__syncthreads();

		j = bw_index % num_layer;

		if(idx < cur_node * next_node)
		{
			// Wait until next layer sends delta
			if(idx == 0)
				while(atomicCAS(cur_bw_ready + j, 1, 1) != 1);
			
			__syncthreads();
				
			if(idx < cur_node)
			{
				cur_delta[j][idx] = 0.0;
				for(int i = 1; i < next_node; i++)
					cur_delta[j][idx] += cur_next_delta[j][i] * cur_weight[j][i * cur_node + idx];
			}

			__syncthreads();

			// while(prev_bw_ready[j] != 0) { }
			if(idx == 0)
				while(atomicCAS(prev_bw_ready + j, 1, 1) == 1);

			__syncthreads();

			if(idx < cur_node)
			{
				prev_next_delta[j][idx] = cur_delta[j][idx];
				if(idx == 0)
				{
					prev_bw_ready[j] = 1;
					printf("hidden layer %d backward epoch and #data: %d , %d\n", layer_index, cur_epoch, bw_index);
				}
			}

			__syncthreads();

			if(idx < cur_node * next_node)
			{
				cur_weight[j][idx] = cur_weight[j][idx] + lr * cur_a[j][idx % cur_node] * cur_next_delta[j][idx / cur_node];

				__syncthreads();

				if(idx == 0) {
					cur_bw_ready[j] = 0;
					cur_fw_ready[j] = 0;
					if(++bw_index == num_data)
					{
						bw_index = 0;
						cur_epoch++;
					}
				}
			}
		}
	}

	return;
}

__global__ void SteadyStageOutputlayer(double **cur_a, double **cur_delta, double **prev_next_delta, int *prev_bw_ready, int *cur_fw_ready, int *result, int cur_node, int num_layer, int epoch, int num_data)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int index = num_layer, cur_epoch = 0, j;
	double sum;

	while(cur_epoch < epoch)
	{
		j = index % num_layer;
		if(idx == 0)
			while( atomicCAS ( cur_fw_ready + j , 1 , 1 ) != 1);

		__syncthreads();

		if(idx < cur_node)
		{
			cur_a[j][idx] = exp(cur_a[j][idx]);
			if(idx == 0)
				cur_a[j][idx] = 0.0;

			__syncthreads();

			sum = 0;
			for(int i = 1; i < cur_node; i++)
				sum += cur_a[j][i];

			cur_a[j][idx] /= sum;

			if(idx != result[index])
				cur_delta[j][idx] = 0.0 - cur_a[j][idx];
			else
				cur_delta[j][idx] = 1.0 - cur_a[j][idx];
			
			if(idx == 0)
				while(atomicCAS(prev_bw_ready + j, 1, 1) == 1);

			__syncthreads();

			prev_next_delta[j][idx] = cur_delta[j][idx];

			__syncthreads();

			if(idx == 0)
			{
				prev_bw_ready[j] = 1;
				cur_fw_ready[j] = 0;
				printf("output layer forward + backward epoch and #data: %d , %d\n", cur_epoch, index);
				if(++index == num_data)
				{
					index = 0;
					cur_epoch++;
				}
			}
		}
	}

	return;
}

__global__ void StartStageInputlayer(double *input, double **cur_a, double **cur_next_a, double **cur_weight, double **next_a, double **next_delta, int *next_fw_ready, int *cur_bw_ready, int cur_node, int next_node, int num_layer, double lr)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
		
	for(int j = 0; j < num_layer; j++)
	{
		if(idx < cur_node)
			cur_a[j][idx] = input[j * cur_node + idx];
		
		__syncthreads();

		if(idx < next_node)
		{
			// MatrixMultiply
			cur_next_a[j][idx] = 0;
			for(int i = 0; i < cur_node; i++)
				cur_next_a[j][idx] += cur_a[j][i] * cur_weight[j][idx * cur_node + i];
			
			// Sigmoid
			cur_next_a[j][idx] = 1.0 / (1.0 + exp(-cur_next_a[j][idx]));

			// Wait until next layer finishes task
			if(idx == 0)
			 	while( atomicCAS (next_fw_ready + j , 1 , 1 ) == 1);

			__syncthreads();

			// cudaMemcpyPeer
			if(idx == 0)
				next_a[j][idx] = 1.0;
			else
				next_a[j][idx] = cur_next_a[j][idx];

			__syncthreads();

			if(idx == 0)
			{
				next_fw_ready[j] = 1;
				printf("input layer forward : %d\n", j);
			}
		}
	} // Forward  X num_layer

	__syncthreads();

	if(idx < cur_node * next_node)
	{
		// Wait until next layer sends delta
		if(idx == 0)
			while(atomicCAS(cur_bw_ready, 1, 1) != 1);

		__syncthreads();

		cur_weight[0][idx] = cur_weight[0][idx] + lr * cur_a[0][idx % cur_node] * next_delta[0][idx / cur_node];
		
		__syncthreads();

		if(idx == 0) {
			cur_bw_ready[0] = 0;
			printf("input layer backward: 0\n");
		}
	}
}

__global__ void StartStageHiddenlayer(double **cur_a, double **cur_next_a, double **cur_weight, double **next_a, double **cur_delta, double **cur_next_delta, double **prev_next_delta, int *cur_fw_ready, int *next_fw_ready, int *prev_bw_ready, int *cur_bw_ready, int cur_node, int next_node, int num_layer, int layer_index, double lr)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for(int j = 0; j < num_layer; j++)
	{
		if(idx == 0)
			while(atomicCAS(cur_fw_ready + j, 1, 1) != 1);

		__syncthreads();
			
		if(idx < next_node)
		{
			// MatrixMultiply
			cur_next_a[j][idx] = 0.0;
			for(int i = 0; i < cur_node; i++)
				cur_next_a[j][idx] += cur_a[j][i] * cur_weight[j][idx * cur_node + i];
			
			// Sigmoid
			cur_next_a[j][idx] = 1.0 / (1.0 + exp(-cur_next_a[j][idx]));

			// Wait until next layer finishes task
			if(idx == 0)
			 	while( atomicCAS ( next_fw_ready + j , 1 , 1 ) == 1);

			__syncthreads();

			// cudaMemcpyPeer
			if(idx == 0)
				next_a[j][idx] = 1.0;
			else
				next_a[j][idx] = cur_next_a[j][idx];

			__syncthreads();

			if(idx == 0)
			{
				next_fw_ready[j] = 1;
				printf("hidden layer %d forward : %d\n", layer_index, j);
			}

		}
	}

	__syncthreads();

	for(int j = 0; j < layer_index + 1; j++)
	{
		if(idx < cur_node * next_node)
		{
			// Wait until next layer sends delta
			if(idx == 0)
				while(atomicCAS(cur_bw_ready + j, 1, 1) != 1);
			
			__syncthreads();
				
			if(idx < cur_node)
			{
				cur_delta[j][idx] = 0.0;
				for(int i = 1; i < next_node; i++)
					cur_delta[j][idx] += cur_next_delta[j][i] * cur_weight[j][i * cur_node + idx];
			}

			__syncthreads();

			// while(prev_bw_ready[j] != 0) { }
			if(idx == 0)
				while(atomicCAS(prev_bw_ready + j, 1, 1) == 1);

			__syncthreads();

			if(idx < cur_node)
			{
				prev_next_delta[j][idx] = cur_delta[j][idx];
				if(idx == 0)
				{
					prev_bw_ready[j] = 1;
					printf("hidden layer %d backward : %d\n", layer_index, j);
				}
			}

			__syncthreads();

			if(idx < cur_node * next_node)
			{
				cur_weight[j][idx] = cur_weight[j][idx] + lr * cur_a[j][idx % cur_node] * cur_next_delta[j][idx / cur_node];

				__syncthreads();

				if(idx == 0) {
					cur_bw_ready[j] = 0;
					cur_fw_ready[j] = 0;
				}
			}
		}
	}
}

__global__ void StartStageOutputlayer(double **cur_a, double **cur_delta, double **prev_next_delta, int *prev_bw_ready, int *cur_fw_ready, int *result, int cur_node, int num_layer)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	double sum;
	for(int j = 0; j < num_layer; j++)
	{
		if(idx < cur_node)
		{
			if(idx == 0)
				while( atomicCAS ( cur_fw_ready + j , 1 , 1 ) != 1);

			__syncthreads();

			cur_a[j][idx] = exp(cur_a[j][idx]);
			if(idx == 0)
				cur_a[j][idx] = 0.0;

			__syncthreads();

			sum = 0;
			for(int i = 1; i < cur_node; i++)
				sum += cur_a[j][i];

			cur_a[j][idx] /= sum;

			if(idx != result[j])
				cur_delta[j][idx] = 0.0 - cur_a[j][idx];
			else
				cur_delta[j][idx] = 1.0 - cur_a[j][idx];
			
			if(idx == 0)
				while(atomicCAS(prev_bw_ready + j, 1, 1) == 1);

			__syncthreads();

			prev_next_delta[j][idx] = cur_delta[j][idx];

			__syncthreads();

			if(idx == 0)
			{
				prev_bw_ready[j] = 1;
				cur_fw_ready[j] = 0;
				printf("output layer forward + backward : %d\n", j);
			}
		}
	}
}

/*
void MatrixMultiply(double *d_A, double *d_B, double *d_C, int A_H, int A_W, int B_W, int i)
{
	const double alp = 1.0f;
	const double bet  = 0.0f;
		
	cublasDgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, B_W, A_H, A_W, &alp, d_B, B_W, d_A, A_W, &bet, d_C, B_W);
}*/

__device__ void GetOutputLayerDelta(double *output_a, double *output_delta, double *result, int size)
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

__device__ void GetHiddenLayerDelta(double *cur_delta, double *cur_a, double *cur_weight, double *prev_delta, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		cur_delta[i] = (1.0 - cur_a[i]) * cur_a[i] * (cur_delta[i] - cur_weight[i] * prev_delta[0]);
}

__device__ void UpdateWeight(double *cur_a, double *cur_weight, double *next_delta, int W_W, int size, double lr)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		cur_weight[i] = cur_weight[i] + lr * cur_a[i / W_W] * next_delta[i % W_W];
}

__device__ void Sigmoid(double *a, int size) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		a[i] = 1.0 / (1.0 + exp(-a[i]));
	if(i == 0)
		a[i] = 1.0;
}

__device__ void Exponential(double *a, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		a[i] = exp(a[i]);
	if(i == 0)
		a[i] = 0.0;
}

__device__ void Softmax(double *a, double sum, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		a[i] /= sum;
}
