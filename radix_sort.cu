#include "radix_sort.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "defines.h"

#include <cstdlib>
#include <cstdio>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

// check whether a sequence has n-th bit set for all n and put the results into an array
__global__ void generateFlagsKernel(const unsigned int* input, int* flags, const int sequenceLength, const int numberOfSequences, const int i, const int j)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numberOfSequences) return;
	flags[id] = (bool)((input[i * numberOfSequences + id]) & (1 << j));
}

// parallel split
__global__ void splitKernel(const int* flags, int* indices, const int sequenceLength, const int numberOfSequences, const int* I_down, const int* I_up)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numberOfSequences) return;
	if (flags[id])
		indices[id] = I_up[id];
	else
		indices[id] = I_down[id];
}

// permute using the indices array
__global__ void permuteKernel(const unsigned int* input, unsigned int* output, const int* indices, const int sequenceLength, const int numberOfSequences)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numberOfSequences) return;
    for (int i = 0; i < sequenceLength; i++)
    {
        output[i * numberOfSequences + indices[id]] = input[i * numberOfSequences + id];
    }
}

// negation functor
struct negation : public thrust::unary_function<int, int>
{
    __host__ __device__ int operator()(const int& x) const
    {
        return -x + 1;
    }
};

// sort the input using radix sort
// Guy E Blelloch. Prefix sums and their applications, 1990
cudaError_t radixSort(unsigned int** dev_input, const int sequenceLength, const int numberOfSequences)
{
    cudaError_t cudaStatus;
    int* dev_flags = 0;
    int* dev_indices = 0;
    unsigned int* dev_output = 0;

    int threadsPerBlock = numberOfSequences < 1024 ? sequenceLength * numberOfSequences : 1024;

    cudaStatus = cudaMalloc((void**)&dev_output, sizeof(int) * sequenceLength * numberOfSequences);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_indices, sizeof(int) * numberOfSequences);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    {
        // Sort
        thrust::device_vector<int> dev_prescanned(numberOfSequences);
        thrust::device_vector<int> dev_scanned(numberOfSequences);
        thrust::device_vector<int> dev_flags(numberOfSequences);
        int* dev_flags_ptr = thrust::raw_pointer_cast(dev_flags.data());

        for (int i = sequenceLength - 1; i >= 0; i--)
        {
            for (int j = 0; j < INTSIZE; j++)
            {
                // generate flags
                generateFlagsKernel << <(numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > 
                    (*dev_input, dev_flags_ptr, sequenceLength, numberOfSequences, i, j);

                // calculate I_down
                auto begin = thrust::make_transform_iterator(dev_flags.begin(), negation());
                auto end = thrust::make_transform_iterator(dev_flags.end(), negation());
                thrust::exclusive_scan(begin, end, dev_prescanned.begin());
                int* I_down = thrust::raw_pointer_cast(dev_prescanned.data());

                // calculate I_up
                thrust::inclusive_scan(dev_flags.rbegin(), dev_flags.rend(), dev_scanned.rbegin());
                using namespace thrust::placeholders;
                thrust::for_each(dev_scanned.begin(), dev_scanned.end(), _1 = numberOfSequences - _1);
                int* I_up = thrust::raw_pointer_cast(dev_scanned.data());

                // split
                splitKernel << <(numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > 
                    (dev_flags_ptr, dev_indices, sequenceLength, numberOfSequences, I_down, I_up);

                // permute input
                permuteKernel << <(numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > 
                    (*dev_input, dev_output, dev_indices, sequenceLength, numberOfSequences);

                unsigned int* temp;

                temp = *dev_input;
                *dev_input = dev_output;
                dev_output = temp;
            }
        }
    }
Error:
    cudaFree(dev_output);
    cudaFree(dev_indices);
    return cudaStatus;
}