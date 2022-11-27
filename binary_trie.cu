#include "binary_trie.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "defines.h"

#include <cstdlib>
#include <cstdio>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <iostream>

cudaError_t allocateResources(int** dev_children, int** dev_preScanned, int** dev_minRange, int** dev_maxRange, int** dev_newMinRange, int** dev_newMaxRange, 
	int** dev_counter, int** dev_prevCounter, int** dev_childrenCount, int** dev_prevChildrenCount, const int numberOfSequences)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)dev_preScanned, sizeof(int) * 2 * numberOfSequences);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_preScanned cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)dev_children, sizeof(int) * 2 * numberOfSequences);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_children cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)dev_minRange, sizeof(int) * numberOfSequences + 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_minRange cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)dev_maxRange, sizeof(int) * numberOfSequences + 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_maxRange cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)dev_newMinRange, sizeof(int) * numberOfSequences + 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_newMinRang cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)dev_newMaxRange, sizeof(int) * numberOfSequences + 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_newMaxRange cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)dev_counter, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_counter cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)dev_prevCounter, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_prevCounter cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)dev_childrenCount, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_childrenCount cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)dev_prevChildrenCount, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_childrenCount cudaMalloc failed!");
	}
	return cudaStatus;
}

__global__ void markChildrenKernel(const unsigned int* input, int* children, const int* minRange, const int* maxRange, const int numberOfSequences,
	const int* counter, const int i, const int j, const int* childrenCount)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= *childrenCount) return;
	//printf("entering mark children (thread %d)\n", id);
	//printf("minRange (thread %d): %d\n", id, minRange[id]);
	//printf("maxRange (thread %d): %d\n", id, maxRange[id]);
	// we have a left child
	if (!((input[i * numberOfSequences + minRange[id]]) & (1 << j)))
	{
		//printf("left child (thread %d), %d\n", id, i * numberOfSequences + minRange[id]);
		children[2 * id] = 1;
	}
		
	else
	{
		//printf("no left child (thread %d), %d\n", id, i * numberOfSequences + minRange[id]);
		children[2 * id] = 0;
	}
			
	//printf("stop 1\n");
	// we have a right child
	if ((input[i * numberOfSequences + maxRange[id]]) & (1 << j))
	{
		//printf("right child (thread %d), %d\n", id, i * numberOfSequences + maxRange[id]);
		children[2 * id + 1] = 1;
	}
		
	else
	{
		//printf("no right child (thread %d): %d\n", id, i * numberOfSequences + maxRange[id]);
		children[2 * id + 1] = 0;
	}
		
	//printf("exiting mark children\n");

	//printf("Children: %d, %d\n", children[2 * id], children[2 * id + 1]);
}

__global__ void childrenUpdateKernel(int* childrenCount, int* prevChildrenCount, const int* children, const int* preScanned)
{
	//printf("entering children update, childrenCount: %d, prescanned: %d\n", *childrenCount, preScanned[2 * (*childrenCount) - 1]);
	//printf("children: %d, %d, %d, %d", children[0], children[1], children[2], children[3]);
	*prevChildrenCount = *childrenCount;
	*childrenCount = preScanned[2 * (*childrenCount) - 1] + children[2 * (*childrenCount) - 1];
	//printf("exiting children update, children:%d\n", *childrenCount);
}

__global__ void fillIndicesKernel(const int* children, const int* preScanned, int* L, int* R, const int* counter, const int* prevCounter, const int* childrenCount)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= *childrenCount) return;
	//printf("entering fillIndices (thread: %d)\n", id);

	// we have a left child
	if (children[2 * id] == 1)
		L[*prevCounter + id] = *counter + preScanned[2 * id];
	else
		L[*prevCounter + id] = NOCHILD;

	// we have a right child
	if (children[2 * id + 1] == 1)
		R[*prevCounter + id] = *counter + preScanned[2 * id + 1];
	else
		R[*prevCounter + id] = NOCHILD;
	//printf("exiting fillIndices (thread: %d)\n", id);
}


__global__ void calculateRangesKernel(const unsigned int* input, const int* children, const int* preScanned, int* minRange, int* maxRange,
	int* newMinRange, int* newMaxRange, const int numberOfSequences, const int sequenceLength, const int* counter, const int i, const int j, const int* prevChildrenCount)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= *prevChildrenCount) return;
	//printf("entering calculateRanges kernel (thread: %d)\n", id);
	// if we have a left child
	if (children[2 * id] == 1)
	{
		//printf("we have a left child (thread: %d), minrange: %d\n", id, minRange[id]);
		newMinRange[preScanned[2 * id]] = minRange[id];
		//printf("filled minrange (thread: %d)\n", id);
		// find max range - first 1 from the right
		int max = maxRange[id];
		for (int sequence = minRange[id]; sequence <= maxRange[id]; sequence++)
		{
			if ((input[i* numberOfSequences + sequence]) & (1 << j))
			{
				max = sequence - 1;
				break;
			}
		}
		newMaxRange[preScanned[2 * id]] = max < 0 ? 0 : max;
	}
	// if we have a right child
	if (children[2 * id + 1] == 1)
	{
		//printf("we have a right child (thread: %d)\n", id);
		newMaxRange[preScanned[2 * id + 1]] = maxRange[id];
		// find min range - last 0 from the left
		int min = maxRange[id];
		for (int sequence = maxRange[id]; sequence >= minRange[id]; sequence--)
		{
			if (!((input[i * numberOfSequences + sequence]) & (1 << j)))
			{
				min = sequence + 1;
				break;
			}
		}
		newMinRange[preScanned[2 * id + 1]] = min >= numberOfSequences ? numberOfSequences - 1 : min;
	}
	//printf("exiting calculateRanges kernel (thread: %d)\n", id);
}

__global__ void buildFirstLevelKernel(const unsigned int* input, int* L, int* R, const int numberOfSequences,
	int* counter, int* prevCounter, int* childrenCount, int* minRange, int* maxRange)
{
	*counter = 1;
	// mark children
	
	if (!(input[0] & (1 << (INTSIZE - 1))))
		L[0] = ++(*counter);
	else
		L[0] = NOCHILD;
	if (input[numberOfSequences - 1] & (1 << (INTSIZE - 1)))
		R[0] = ++(*counter);
	else
		R[0] = NOCHILD;
	//printf("marked\n");
	// fill ranges

	minRange[0] = 0;
	// both children
	if (*counter == 3)
	{
		int index = 0;
		//printf("Index: %d\n", index);
		for (int sequence = 0; sequence < numberOfSequences; sequence++)
		{
			if (input[sequence] & (1 << INTSIZE - 1))
			{
				index = sequence;
				break;
			}
		}
		//printf("Index: %d\n", index);
		maxRange[0] = index - 1;
		minRange[1] = index;
		maxRange[1] = numberOfSequences - 1;
	}
	// only one child
	else
	{
		maxRange[0] = numberOfSequences - 1;
	}

	*childrenCount = *counter - 1;
	//printf("childrencount %d\n", *childrenCount);
}

// increments node counter
__global__ void incrementCounterKernel(int* counter, int* prevCounter, const int* childrenCount)
{
	//printf("entering increment counter\n");
	*prevCounter = *counter;
	*counter += *childrenCount;
	//printf("exiting increment counter\n");
}

// fills last level
//__global__ void buildLastLevelKernel(const int* counter, const int* prevCounter, int* L, int* R)
//{
//	for (int leaf = *prevCounter; leaf < *counter; leaf++)
//	{
//		L[leaf] = FOUND;
//		R[leaf] = FOUND;
//	}
//}

// special case of buulding a binary trie for one-int sequences
void buildTrie(const unsigned int* input, int* L, int* R, const int numberOfSequences, const int sequenceLength)
{
	cudaError_t cudaStatus;
	int* dev_children = 0;
	int* dev_preScanned = 0;
	int* dev_minRange = 0;
	int* dev_maxRange = 0;
	int* dev_newMinRange = 0;
	int* dev_newMaxRange = 0;
	int* dev_counter = 0;
	int* dev_prevCounter = 0;
	int* dev_childrenCount = 0;
	int* dev_prevChildrenCount = 0;

	cudaStatus = allocateResources(&dev_children, &dev_preScanned, &dev_minRange, &dev_maxRange, &dev_newMinRange, &dev_newMaxRange, &dev_counter, &dev_prevCounter,
		&dev_childrenCount, &dev_prevChildrenCount, numberOfSequences);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "allocateResources failed!");
		goto Error;
	}

	thrust::device_ptr<int> dev_children_ptr = thrust::device_pointer_cast<int>(dev_children);
	thrust::device_ptr<int> dev_preScanned_ptr = thrust::device_pointer_cast<int>(dev_preScanned);

	// first level
	
	cudaMemset(dev_children, 0, 2 * numberOfSequences);
	buildFirstLevelKernel<<<1,1>>>(input, L, R, numberOfSequences, dev_counter, dev_prevCounter, dev_childrenCount, dev_minRange, dev_maxRange);
	int threadsPerBlock = numberOfSequences < 1024 ? numberOfSequences : 1024;

	// middle
	for (int i = 0; i < sequenceLength; i++)
	{
		int from = i == 0 ? INTSIZE - 2 : INTSIZE - 1;
		int to = i == sequenceLength - 1 ? 1 : 0;
		for (int j = from; j > to; j--)
		{
			// mark children
			markChildrenKernel << <(numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (input, dev_children,
				dev_minRange, dev_maxRange, numberOfSequences, dev_counter, i, j, dev_childrenCount);
			cudaDeviceSynchronize();
			
			//preScan
			thrust::exclusive_scan(dev_children_ptr, dev_children_ptr + 2 * (numberOfSequences), dev_preScanned_ptr);
			cudaDeviceSynchronize();
			// number of children
			childrenUpdateKernel << <1, 1 >> > (dev_childrenCount, dev_prevChildrenCount, dev_children, dev_preScanned);
			cudaDeviceSynchronize();
			// fill indices		
			fillIndicesKernel << < (numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_children, dev_preScanned, L, R, dev_counter, dev_prevCounter, dev_prevChildrenCount);
			cudaDeviceSynchronize();
			// calculate ranges
			calculateRangesKernel << < (numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (input, dev_children, dev_preScanned, dev_minRange, dev_maxRange,
				dev_newMinRange, dev_newMaxRange, numberOfSequences, 1, dev_counter, i, j, dev_prevChildrenCount);
			cudaDeviceSynchronize();
			// swap ranges
			int* temp = dev_minRange;
			dev_minRange = dev_newMinRange;
			dev_newMinRange = temp;

			temp = dev_maxRange;
			dev_maxRange = dev_newMaxRange;
			dev_newMaxRange = temp;

			incrementCounterKernel << <1, 1 >> > (dev_counter, dev_prevCounter, dev_childrenCount);
			cudaDeviceSynchronize();
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSynchronize failed!");
			goto Error;
		}
	}
	

	// last level
	/*int threadsPerBlock = numberOfSequences < 1024 ? numberOfSequences : 1024;
	buildLastLevelKernel << <(numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > ();*/


Error:
	cudaFree(dev_children);
	cudaFree(dev_preScanned);
	cudaFree(dev_counter);
	cudaFree(dev_prevCounter);
	cudaFree(dev_childrenCount);
	cudaFree(dev_minRange);
	cudaFree(dev_maxRange);
	cudaFree(dev_newMinRange);
	cudaFree(dev_newMaxRange);
}
