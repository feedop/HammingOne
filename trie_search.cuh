#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t hammingOne(const unsigned int* input, const int numberOfSequences, const int sequenceLength, long long& matchCount, cudaEvent_t& start, cudaEvent_t& stop, float& totalTime, const int printPairs);