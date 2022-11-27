#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t allocateResources(int** dev_children, int** dev_preScanned, int** dev_minRange, int** dev_maxRange, int** dev_newMinRange, int** dev_newMaxRange,
	int** dev_counter, int** dev_prevCounter, int** dev_childrenCount, int** dev_prevChildrenCount, const int numberOfSequences);

__global__ void markChildrenKernel(const unsigned int* input, int* children, const int* minRange, const int* maxRange, const int numberOfSequences,
	const int* counter, const int i, const int j, const int* childrenCount);

__global__ void childrenUpdateKernel(int* childrenCount, int* prevChildrenCount, const int* children, const int* preScanned);

__global__ void fillIndicesKernel(const int* children, const int* preScanned, int* L, int* R, const int* counter, const int* prevCounter, const int* childrenCount);

__global__ void calculateRangesKernel(const unsigned int* input, const int* children, const int* preScanned, int* minRange, int* maxRange,
	int* newMinRange, int* newMaxRange, const int numberOfSequences, const int sequenceLength, const int* counter, const int i, const int j, const int* prevChildrenCount);

__global__ void buildFirstLevelKernel(const unsigned int* input, int* L, int* R, const int numberOfSequences,
	int* counter, int* prevCounter, int* childrenCount, int* minRange, int* maxRange);

__global__ void incrementCounterKernel(int* counter, int* prevCounter, const int* childrenCount);

__global__ void buildLastLevelKernel(const int* counter, const int* prevCounter, int* L, int* R);

void buildTrie(const unsigned int* input, int* L, int* R, const int numberOfSequences, const int sequenceLength);
