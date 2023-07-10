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

namespace GPU
{
	cudaError_t allocateResources(int** dev_children, int** dev_preScanned, int** dev_minRange, int** dev_maxRange, int** dev_newMinRange, int** dev_newMaxRange,
		int** dev_counter, int** dev_prevCounter, int** dev_childrenCount, int** dev_prevChildrenCount, int** dev_tempChildrenCount, const int numberOfSequences)
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc((void**)dev_preScanned, sizeof(int) * 2 * numberOfSequences);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "dev_preScanned cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)dev_children, sizeof(int) * 2 * numberOfSequences);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "dev_children cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)dev_minRange, sizeof(int) * numberOfSequences + 1);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "dev_minRange cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)dev_maxRange, sizeof(int) * numberOfSequences + 1);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "dev_maxRange cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)dev_newMinRange, sizeof(int) * numberOfSequences + 1);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "dev_newMinRang cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)dev_newMaxRange, sizeof(int) * numberOfSequences + 1);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "dev_newMaxRange cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)dev_counter, sizeof(int));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "dev_counter cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)dev_prevCounter, sizeof(int));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "dev_prevCounter cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)dev_childrenCount, sizeof(int));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "dev_childrenCount cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)dev_prevChildrenCount, sizeof(int));
		if (cudaStatus != cudaSuccess)
		{
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

		// we have a left child
		if (!((input[i * numberOfSequences + minRange[id]]) & (1 << j)))
		{
			children[2 * id] = 1;
		}
		else
		{
			children[2 * id] = 0;
		}

		// we have a right child
		if ((input[i * numberOfSequences + maxRange[id]]) & (1 << j))
		{
			children[2 * id + 1] = 1;
		}
		else
		{
			children[2 * id + 1] = 0;
		}
	}

	__global__ void childrenUpdateKernel(int* childrenCount, int* tempChildrenCount, const int* children, const int* preScanned)
	{
		*tempChildrenCount = *childrenCount;
		*childrenCount = preScanned[2 * (*childrenCount) - 1] + children[2 * (*childrenCount) - 1];
	}

	__global__ void fillIndicesKernel(const int* children, const int* preScanned, int* L, int* R, const int* counter, const int* prevCounter, const int* childrenCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= *childrenCount) return;

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
	}


	__global__ void calculateRangesKernel(const unsigned int* input, const int* children, const int* preScanned, int* minRange, int* maxRange,
		int* newMinRange, int* newMaxRange, const int numberOfSequences, const int sequenceLength, const int* counter, const int i, const int j, const int* prevChildrenCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= *prevChildrenCount) return;

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

	   // if we have a left child
		newMinRange[preScanned[2 * id]] = minRange[id];
		// find max range - first 1 from the right
		int max = maxRange[id];
		for (int sequence = minRange[id]; sequence <= maxRange[id]; sequence++)
		{
			if ((input[i * numberOfSequences + sequence]) & (1 << j))
			{
				max = sequence - 1;
				break;
			}
		}
		newMaxRange[preScanned[2 * id]] = max < 0 ? 0 : max;

		// if we have a right child
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
			for (int sequence = 0; sequence < numberOfSequences; sequence++)
			{
				if (input[sequence] & (1 << INTSIZE - 1))
				{
					index = sequence;
					break;
				}
			}
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

	// build the tree top-down
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
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "allocateResources failed!");
			goto Error;
		}

		thrust::device_ptr<int> dev_children_ptr = thrust::device_pointer_cast<int>(dev_children);
		thrust::device_ptr<int> dev_preScanned_ptr = thrust::device_pointer_cast<int>(dev_preScanned);

		// first level

		cudaMemset(dev_children, 0, 2 * numberOfSequences);
		buildFirstLevelKernel << <1, 1 >> > (input, L, R, numberOfSequences, dev_counter, dev_prevCounter, dev_childrenCount, dev_prevChildrenCount, dev_minRange, dev_maxRange);
		int threadsPerBlock = numberOfSequences < 1024 ? numberOfSequences : 1024;

		// middle
		for (int i = 0; i < sequenceLength; i++)
		{
			int from = i == 0 ? INTSIZE - 2 : INTSIZE - 1;
			int to = i == sequenceLength - 1 ? 1 : 0;
			for (int j = from; j >= to; j--)
			{

				// determine how many children each node on the current level has and mark them
				markChildrenKernel << <(numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (input, dev_children,
					dev_minRange, dev_maxRange, numberOfSequences, dev_counter, i, j, dev_childrenCount);
				//preScan
				thrust::exclusive_scan(dev_children_ptr, dev_children_ptr + 2 * (numberOfSequences), dev_preScanned_ptr);;
				// determine the total number of children
				childrenUpdateKernel << <1, 1 >> > (dev_childrenCount, dev_prevChildrenCount, dev_children, dev_preScanned);
				// fill indices
				fillIndicesKernel << < (numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_children, dev_preScanned, L, R, dev_counter, dev_prevCounter, dev_prevChildrenCount);
				// calculate ranges
				calculateRangesKernel << < (numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (input, dev_children, dev_preScanned, dev_minRange, dev_maxRange,
					dev_newMinRange, dev_newMaxRange, numberOfSequences, 1, dev_counter, i, j, dev_prevChildrenCount);
				cudaDeviceSynchronize();

				// increment prevChildren count by value calculated in childrenUpdateKernel
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
}