#include "trie_search.cuh"
#include "binary_trie.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "defines.h"

#include <cstdlib>
#include <cstdio>
#include <iostream>

cudaError_t hammingOne(const unsigned int* input, const int numberOfSequences, const int sequenceLength)
{
	cudaError_t cudaStatus;
	int* L = 0;
	int* R = 0;

	long long memory = (long long)numberOfSequences * (long long)sequenceLength * (long long)32;

	cudaStatus = cudaMalloc((void**)&L, sizeof(int) * memory);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&R, sizeof(int) * memory);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	printf("allocated %lld for L and R\n", memory * 32);

	//build tree
	buildTrie(input, L, R, numberOfSequences, sequenceLength);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSynchronize failed!");
		goto Error;
	}

Error:
	cudaFree(L);
	cudaFree(R);
	return cudaStatus;
}