
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "radix_sort.cuh"

#include <iostream>
#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <vector>

cudaError_t generateRandoms(unsigned int* dev_input, const int& sequenceLength, const int& numberOfSequences);

// prints a few sequences from the beginning 
void head(unsigned int* input, float milliseconds, const int& sequenceLength, const int& numberOfSequences)
{
    int vecLen = sequenceLength > 3 ? 3 : sequenceLength;
    int vectors = numberOfSequences > 10 ? 10 : numberOfSequences;
    std::vector<std::bitset<32>>bitset(vecLen * vectors);
    for (int i = 0; i < vecLen; i++)
    {
        for (int j = 0; j < vectors; j++)
        {
            bitset[i * vectors + j] = std::bitset<32>(input[(i + sequenceLength - vecLen) * numberOfSequences + j]);
        }
    }

    std::cout << "Values: \n";
    for (int i = 0; i < vectors; i++)
    {
        for (int j = vecLen - 1; j >= 0; j--)
        {
            std::cout << bitset[j * vectors + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Time elapsed: " << milliseconds / 1000 << " seconds\n";
}

__global__ void generateRandomsKernel(unsigned int *input, const int sequenceLength, const int numberOfSequences)
{
    unsigned int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= sequenceLength * numberOfSequences) return;
    curandState localState;
    curand_init(seed, 0, 0, &localState);
    input[seed] = curand(&localState);
}

int main()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int sequenceLength = 32;
    int numberOfSequences = 100000;    

    unsigned int* input = (unsigned int*)malloc(sizeof(int) * sequenceLength * numberOfSequences);
    unsigned int* dev_input = 0;

    cudaStatus = cudaMalloc((void**)&dev_input, sizeof(int) * sequenceLength * numberOfSequences);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // generate random sequences
    cudaEventRecord(start);
    cudaStatus = generateRandoms(dev_input, sequenceLength, numberOfSequences);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(input, dev_input, sizeof(int) * sequenceLength * numberOfSequences, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // print
    head(input, milliseconds, sequenceLength, numberOfSequences);

    // ---------------------------------------------------------------------------------

    // radix sort input
    cudaEventRecord(start);
    cudaStatus = radixSort(&dev_input, sequenceLength, numberOfSequences);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(input, dev_input, sizeof(int) * sequenceLength * numberOfSequences, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // print
    head(input, milliseconds, sequenceLength, numberOfSequences);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
Error:
    free(input);
    cudaFree(dev_input);
    //getchar();
    return 0;
}

// fills the input array with random unsigned ints
cudaError_t generateRandoms(unsigned int* dev_input, const int& sequenceLength, const int& numberOfSequences)
{
    cudaError_t cudaStatus;

    int threadsPerBlock = sequenceLength * numberOfSequences < 1024 ? sequenceLength * numberOfSequences : 1024;

    generateRandomsKernel << <(sequenceLength * numberOfSequences + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_input, sequenceLength, numberOfSequences);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching generateRandomsKernel!\n", cudaStatus);
    }

    return cudaStatus;
}
