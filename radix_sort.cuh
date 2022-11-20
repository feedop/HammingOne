#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void generateFlagsKernel(unsigned int* input, int* flags, const int sequenceLength, const int numberOfSequences, const int i, const int j);

__global__ void splitKernel(const int* flags, int* indices, const int sequenceLength, const int numberOfSequences, const int* I_down, const int* I_up);

__global__ void permuteKernel(unsigned int* input, unsigned int* output, int* indices, const int sequenceLength, const int numberOfSequences);

cudaError_t radixSort(unsigned int** input, const int sequenceLength, const int numberOfSequences);

