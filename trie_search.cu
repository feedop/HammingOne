#include "trie_search.cuh"
#include "binary_trie.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "defines.h"

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

namespace GPU
{
	template<bool printPairs>
	__device__ inline void search(const unsigned int* input, const int* L, const int* R, const int numberOfSequences, const int sequenceLength, long long* matches,
		const int integer, const int bit, const int id, const int* minLRange, const int* minRRange)
	{
		// look for a our sequence with one bit changed in the trie
		int ptr = 0;
		int prevPtr = 0;
		// unchanged bits on the left
		for (int i = 0; i < integer; i++)
		{
			for (int j = INTSIZE - 1; j >= 0; j--)
			{
				if ((input[i * numberOfSequences + id]) & (1 << j))
				{
					if (printPairs) prevPtr = ptr;
					ptr = R[ptr];
				}
				else
				{
					if (printPairs) prevPtr = ptr;
					ptr = L[ptr];
				}

				if (ptr == NOCHILD) return;
			}
		}
		// same integer, before changed bit
		for (int j = INTSIZE - 1; j > bit; j--)
		{
			if ((input[integer * numberOfSequences + id]) & (1 << j))
			{
				if (printPairs) prevPtr = ptr;
				ptr = R[ptr];
			}
			else
			{
				if (printPairs) prevPtr = ptr;
				ptr = L[ptr];
			}

			if (ptr == NOCHILD) return;
		}
		// bit change
		if ((input[integer * numberOfSequences + id]) & (1 << bit))
		{
			if (printPairs) prevPtr = ptr;
			ptr = L[ptr];
		}
		else
		{
			if (printPairs) prevPtr = ptr;
			ptr = R[ptr];
		}
		if (ptr == NOCHILD)
		{
			return;
		}

		// same integer, after changed bit
		for (int j = bit - 1; j >= 0; j--)
		{
			if ((input[integer * numberOfSequences + id]) & (1 << j))
			{
				if (printPairs) prevPtr = ptr;
				ptr = R[ptr];
			}
			else
			{
				if (printPairs) prevPtr = ptr;
				ptr = L[ptr];
			}

			if (ptr == NOCHILD) return;
		}
		// unchanged bits on the right
		for (int i = integer + 1; i < sequenceLength; i++)
		{
			for (int j = INTSIZE - 1; j >= 0; j--)
			{
				if ((input[i * numberOfSequences + id]) & (1 << j))
				{
					if (printPairs) prevPtr = ptr;
					ptr = R[ptr];
				}
				else
				{
					if (printPairs) prevPtr = ptr;
					ptr = L[ptr];
				}

				if (ptr == NOCHILD) return;
			}
		}
		// found a match if we arrived at the end
		matches[id] += ptr;
		// print pairs if set to verbose
		if (printPairs)
		{
			// find first element of range, special case for last bit
			int min;
			if (integer == sequenceLength - 1 && bit == 0)
				min = ((input[(sequenceLength - 1) * numberOfSequences + id]) & 1) ? minLRange[prevPtr] : minRRange[prevPtr];
			else
				min = ((input[(sequenceLength - 1) * numberOfSequences + id]) & 1) ? minRRange[prevPtr] : minLRange[prevPtr];

			char* buf = new char[2 * ptr * sequenceLength * INTSIZE + 9];
			int bufCounter = 0;
			for (int sequence = min; sequence <= min + ptr - 1; sequence++)
			{
				buf[bufCounter++] = '[';
				buf[bufCounter++] = '\n';
				for (int i = 0; i < sequenceLength; i++)
				{
					for (int j = INTSIZE - 1; j >= 0; j--)
					{
						buf[bufCounter++] = '0' + ((input[i * numberOfSequences + id] >> j) & 0x0001);
					}
				}
				buf[bufCounter++] = ',';
				buf[bufCounter++] = '\n';
				for (int i = 0; i < sequenceLength; i++)
				{
					for (int j = INTSIZE - 1; j >= 0; j--)
					{
						buf[bufCounter++] = '0' + ((input[i * numberOfSequences + sequence] >> j) & 0x0001);
					}
				}
				buf[bufCounter++] = '\n';
				buf[bufCounter++] = ']';
				buf[bufCounter++] = ';';
				buf[bufCounter++] = '\n';
				buf[bufCounter++] = '\0';
				printf(buf);
			}
			delete buf;
		}
	}

	template <bool printPairs>
	__global__ void searchKernel(const unsigned int* input, const int* L, const int* R, const int numberOfSequences, const int sequenceLength, long long* matches,
		const int* minLRange, const int* minRRange)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= numberOfSequences) return;

		matches[id] = 0;
		// for every bit
		for (int integer = 0; integer < sequenceLength; integer++)
		{
			for (int bit = INTSIZE - 1; bit >= 0; bit--)
			{
				search<printPairs>(input, L, R, numberOfSequences, sequenceLength, matches, integer, bit, id, minLRange, minRRange);
			}
		}
	}

	cudaError_t allocateMemory(int** L, int** R, long long** matches, int** minLRange, int** minRRange, const int numberOfSequences, const int sequenceLength)
	{
		cudaError_t cudaStatus;
		long long memory = (long long)numberOfSequences * (long long)sequenceLength * (long long)32;

		cudaStatus = cudaMalloc((void**)L, sizeof(int) * memory);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "L cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)R, sizeof(int) * memory);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "R cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)matches, sizeof(long long) * numberOfSequences);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)minLRange, sizeof(int) * memory);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "minLRange cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)minRRange, sizeof(int) * memory);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "minRRange cudaMalloc failed!");
		}

		return cudaStatus;
	}

	cudaError_t hammingOne(const unsigned int* input, const int numberOfSequences, const int sequenceLength, long long& matchCount, cudaEvent_t& start, cudaEvent_t& stop,
		float& totalTime, const int printPairs)
	{
		cudaError_t cudaStatus;
		int* L = 0;
		int* R = 0;
		int* minLRange = 0;
		int* minRRange = 0;
		long long* matches = 0;
		float milliseconds = 0;

		cudaStatus = allocateMemory(&L, &R, &matches, &minLRange, &minRRange, numberOfSequences, sequenceLength);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "allocateMemory failed!");
		}
		//build tree

		cudaEventRecord(start);

		buildTrie(input, L, R, minLRange, minRRange, numberOfSequences, sequenceLength, printPairs);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaSynchronize failed!");
			goto Error;
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		totalTime += milliseconds;

		std::cout << "Building the tree completed. Time elapsed: " << milliseconds / 1000 << " seconds\n";

		// search for all possible sequences with Hamming distance equal to 1

		int threadsPerBlock = numberOfSequences < 1024 ? numberOfSequences : 1024;
		int blocks = (numberOfSequences + threadsPerBlock - 1) / threadsPerBlock;
		cudaEventRecord(start);
		if (printPairs)
			searchKernel<true> << <blocks, threadsPerBlock >> > (input, L, R, numberOfSequences, sequenceLength, matches, minLRange, minRRange);
		else
			searchKernel<false> << <blocks, threadsPerBlock >> > (input, L, R, numberOfSequences, sequenceLength, matches, minLRange, minRRange);

		thrust::device_ptr<long long> ptr = thrust::device_pointer_cast(matches);
		matchCount = thrust::reduce(ptr, ptr + numberOfSequences);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaSynchronize failed!");
			goto Error;
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		totalTime += milliseconds;

		std::cout << "Search completed. Time elapsed: " << milliseconds / 1000 << " seconds\n";

	Error:
		cudaFree(L);
		cudaFree(R);
		cudaFree(matches);
		cudaFree(minLRange);
		cudaFree(minRRange);
		return cudaStatus;
	}
}