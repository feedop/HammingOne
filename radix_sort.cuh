#pragma once
namespace GPU
{
	cudaError_t radixSort(unsigned int** input, const int sequenceLength, const int numberOfSequences);
}

