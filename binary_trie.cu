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
	int** dev_counter, int** dev_prevCounter, int** dev_childrenCount, int** dev_prevChildrenCount, int** dev_tempChildrenCount, const int numberOfSequences)
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
	cudaStatus = cudaMalloc((void**)dev_tempChildrenCount, sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
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
	//if (j == 0) printf("minRange: %d, maxrange: %d, childrencount: %d\n", minRange[0], maxRange[0], *childrenCount);
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

__global__ void childrenUpdateKernel(int* childrenCount, int* tempChildrenCount, const int* children, const int* preScanned)
{
	//printf("entering children update, childrenCount: %d, prescanned: %d\n", *childrenCount, preScanned[2 * (*childrenCount) - 1]);
	//printf("children: %d, %d, ", children[0], children[1]);
	*tempChildrenCount = *childrenCount;
	*childrenCount = preScanned[2 * (*childrenCount) - 1] + children[2 * (*childrenCount) - 1];
	//printf("childrenCount = %d\n", *childrenCount);
	//printf("exiting children update, children:%d\n", *childrenCount);
}

__global__ void fillIndicesKernel(const int* children, const int* preScanned, int* L, int* R, const int* counter, const int* prevCounter, const int* childrenCount)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= *childrenCount) return;
	//printf("entering fillIndices (thread: %d)\n", id);
	//if (id <= DEBUG_MAXID && *prevCounter < 50) printf("prevCounter = %d, childrenCount = %d\n", *prevCounter, *childrenCount);
	//if (id == 1) printf("1!!!!!\n");
	// we have a left child
	if (children[2 * id] == 1)
		L[*prevCounter + id] = *counter + preScanned[2 * id];
	else
		L[*prevCounter + id] = NOCHILD;
	//printf("L[%d] = %d,id: %d\n", *prevCounter + id, L[*prevCounter + id], id);
	// we have a right child
	if (children[2 * id + 1] == 1)
		R[*prevCounter + id] = *counter + preScanned[2 * id + 1];
	else
		R[*prevCounter + id] = NOCHILD;
	//printf("R[%d] = %d, id: %d\n", *prevCounter + id, R[*prevCounter + id], id);
	//printf("exiting fillIndices (thread: %d)\n", id);
}


__global__ void calculateRangesKernel(const unsigned int* input, const int* children, const int* preScanned, int* minRange, int* maxRange,
	int* newMinRange, int* newMaxRange, const int numberOfSequences, const int sequenceLength, const int* counter, const int i, const int j, const int* prevChildrenCount)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= *prevChildrenCount) return;
	//printf("entering calculateRanges kernel (thread: %d)\n", id);
	//if (j == 0) printf("ranges before: %d, %d\n", minRange[id], maxRange[id]);
	// if we have only one child
	 if (children[2 * id] + children[2 * id + 1] == 1)
	 {
		 // if we have a left child
		 if (children[2 * id] == 1)
		 {
			 newMinRange[preScanned[2 * id]] = minRange[id];
			 newMaxRange[preScanned[2 * id]] = maxRange[id];
		 }
		 if (children[2 * id + 1] == 1)
		 {
			 newMinRange[preScanned[2 * id + 1]] = minRange[id];
			 newMaxRange[preScanned[2 * id + 1]] = maxRange[id];
		 }
		 return;
	 }
	 //if (j == 0) printf("both children\n");
	// if we have a left child
	//if (children[2 * id] == 1)
	//{
		//if (j == 1)printf("we have a left child (thread: %d), minrange: %d\n", id, minRange[id]);
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
	//}
	// if we have a right child
	//if (children[2 * id + 1] == 1)
	//{
		//if (j <= 1) printf("we have a right child (thread: %d)\n", id);
		newMaxRange[preScanned[2 * id + 1]] = maxRange[id];
		// find min range - last 0 from the left
		int min = maxRange[id];
		for (int sequence = maxRange[id]; sequence >= minRange[id]; sequence--)
		{
			if (!((input[i * numberOfSequences + sequence]) & (1 << j)))
			{
				min = sequence + 1;
				//if (j == 0) printf("min: %d\n", min);
				break;
			}
		}
		newMinRange[preScanned[2 * id + 1]] = min >= numberOfSequences ? numberOfSequences - 1 : min;
	//}
	//printf("exiting calculateRanges kernel (thread: %d)\n", id);
}

__global__ void buildFirstLevelKernel(const unsigned int* input, int* L, int* R, const int numberOfSequences,
	int* counter, int* prevCounter, int* childrenCount, int* prevChildrenCount, int* minRange, int* maxRange)
{
	*counter = 1;
	*prevChildrenCount = 1;
	*prevCounter = 1;
	// mark children
	
	if (!(input[0] & (1 << (INTSIZE - 1))))
		L[0] = (*counter)++;
	else
		L[0] = NOCHILD;
	if (input[numberOfSequences - 1] & (1 << (INTSIZE - 1)))
		R[0] = (*counter)++;
	else
		R[0] = NOCHILD;

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
	*prevCounter = *counter;
	*counter += *childrenCount;
}

__global__ void fillLastLevelKernel(const int* children, const int* preScanned, int* L, int* R, int* minRange, int* maxRange, const int* counter, const int* prevCounter,
	const int* childrenCount, int* outLMinRange, int* outRMinRange, const bool printPairs)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= *childrenCount) return;

	// we have a left child
	if (children[2 * id] == 1)
	{
		L[*prevCounter + id] = maxRange[preScanned[2 * id]] - minRange[preScanned[2 * id]] + 1;
		if (printPairs)
			outLMinRange[*prevCounter + id] = minRange[preScanned[2 * id]];
	}	
	else
		L[*prevCounter + id] = NOCHILD;
	// we have a right child
	if (children[2 * id + 1] == 1)
	{
		R[*prevCounter + id] = maxRange[preScanned[2 * id + 1]] - minRange[preScanned[2 * id + 1]] + 1;
		if (printPairs)
			outRMinRange[*prevCounter + id] = minRange[preScanned[2 * id + 1]];
	}	
	else
		R[*prevCounter + id] = NOCHILD;
}

// build the tree
void buildTrie(const unsigned int* input, int* L, int* R, int* outLMinRange, int* outRMinRange, const int numberOfSequences, const int sequenceLength, bool printPairs)
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
	int* dev_tempChildrenCount = 0;

	cudaStatus = allocateResources(&dev_children, &dev_preScanned, &dev_minRange, &dev_maxRange, &dev_newMinRange, &dev_newMaxRange, &dev_counter, &dev_prevCounter,
		&dev_childrenCount, &dev_prevChildrenCount, &dev_tempChildrenCount, numberOfSequences);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "allocateResources failed!");
		goto Error;
	}

	thrust::device_ptr<int> dev_children_ptr = thrust::device_pointer_cast<int>(dev_children);
	thrust::device_ptr<int> dev_preScanned_ptr = thrust::device_pointer_cast<int>(dev_preScanned);

	// first level
	
	cudaMemset(dev_children, 0, 2 * numberOfSequences);
	buildFirstLevelKernel<<<1,1>>>(input, L, R, numberOfSequences, dev_counter, dev_prevCounter, dev_childrenCount, dev_prevChildrenCount, dev_minRange, dev_maxRange);
	int threadsPerBlock = numberOfSequences < 1024 ? numberOfSequences : 1024;

	// middle
	for (int i = 0; i < sequenceLength; i++)
	{
		int from = i == 0 ? INTSIZE - 2 : INTSIZE - 1;
		int to = i == sequenceLength - 1 ? 1 : 0;
		for (int j = from; j >= to; j--)
		{
			
			// mark children
			markChildrenKernel << <(numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (input, dev_children,
				dev_minRange, dev_maxRange, numberOfSequences, dev_counter, i, j, dev_childrenCount);
			//preScan
			thrust::exclusive_scan(dev_children_ptr, dev_children_ptr + 2 * (numberOfSequences), dev_preScanned_ptr);;
			// number of children
			childrenUpdateKernel << <1, 1 >> > (dev_childrenCount, dev_prevChildrenCount, dev_children, dev_preScanned);
			// fill indices		
			fillIndicesKernel << < (numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_children, dev_preScanned, L, R, dev_counter, dev_prevCounter, dev_prevChildrenCount);
			// calculate ranges
			calculateRangesKernel << < (numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (input, dev_children, dev_preScanned, dev_minRange, dev_maxRange,
				dev_newMinRange, dev_newMaxRange, numberOfSequences, 1, dev_counter, i, j, dev_prevChildrenCount);
			cudaDeviceSynchronize();
			
			// increment prevChildren count by value calculatet in childrenUpdateKernel
			int* temp = dev_tempChildrenCount;
			dev_tempChildrenCount = dev_prevChildrenCount;
			dev_prevChildrenCount = temp;

			// swap ranges
			temp = dev_minRange;
			dev_minRange = dev_newMinRange;
			dev_newMinRange = temp;

			temp = dev_maxRange;
			dev_maxRange = dev_newMaxRange;
			dev_newMaxRange = temp;

			// increment counter
			incrementCounterKernel << <1, 1 >> > (dev_counter, dev_prevCounter, dev_childrenCount);
		}
	}

	// last level

	markChildrenKernel << <(numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (input, dev_children,
		dev_minRange, dev_maxRange, numberOfSequences, dev_counter, sequenceLength - 1, 0, dev_childrenCount);
	//preScan
	thrust::exclusive_scan(dev_children_ptr, dev_children_ptr + 2 * (numberOfSequences), dev_preScanned_ptr);;
	// number of children
	childrenUpdateKernel << <1, 1 >> > (dev_childrenCount, dev_prevChildrenCount, dev_children, dev_preScanned);
	calculateRangesKernel << < (numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (input, dev_children, dev_preScanned, dev_minRange, dev_maxRange,
		dev_newMinRange, dev_newMaxRange, numberOfSequences, 1, dev_counter, sequenceLength - 1, 0, dev_prevChildrenCount);
	// fill indices		
	fillLastLevelKernel << < (numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_children, dev_preScanned, L, R, dev_newMinRange, dev_newMaxRange,
		dev_counter, dev_prevCounter, dev_prevChildrenCount, outLMinRange, outRMinRange, printPairs);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSynchronize failed!");
		goto Error;
	}


Error:
	cudaFree(dev_children);
	cudaFree(dev_preScanned);
	cudaFree(dev_counter);
	cudaFree(dev_prevCounter);
	cudaFree(dev_childrenCount);
	cudaFree(dev_prevChildrenCount);
	cudaFree(dev_tempChildrenCount);
	cudaFree(dev_minRange);
	cudaFree(dev_maxRange);
	cudaFree(dev_newMinRange);
	cudaFree(dev_newMaxRange);
}
