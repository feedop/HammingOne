
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "defines.h"

#include "radix_sort.cuh"
#include "trie_search.cuh"


#include <iostream>
#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <vector>

// reads integers from a file
int readFile(unsigned int* input, FILE* file, int& sequenceLength, int& numberOfSequences)
{
    char buf[BUFSIZE + 1];
    char* info;
    info = fgets(buf, BUFSIZE, file);
    int offset = strlen(info) + 1;
    if (info == NULL)
    {
        fprintf(stderr, "read error");
        return EXIT_FAILURE;
    }
    // get number of sequences
    info = strtok(info, ",");
    numberOfSequences = atoi(info);

    // get number of bits
    info = strtok(NULL, ",");
    int bits = atoi(info);
    sequenceLength = (bits + INTSIZE - 1) / INTSIZE;
    int remainder = bits % INTSIZE;

    // read file to buffer
    fseek(file, offset, 0);
    char* bytes = (char*)malloc(numberOfSequences * (bits + 1) * sizeof(char));
    fread(bytes, sizeof(char), numberOfSequences * (bits + 1), file);
    // copy to input
    char temp[INTSIZE + 1];
    if (remainder == 0)
    {
        for (int i = 0; i < numberOfSequences; i++)
        {
            for (int j = 0; j < sequenceLength; j++)
            {
                memcpy(temp, bytes + i * (bits + 1) + j * INTSIZE, INTSIZE * sizeof(char));
                temp[BUFSIZE] = '\0';
                input[j * numberOfSequences + i] = strtoul(temp, NULL, 2);
            }
        }
    }
    else
    {
        for (int i = 0; i < numberOfSequences; i++)
        {
            // fill with leading zeros
            memcpy(temp, bytes + i * (bits + 1), remainder * sizeof(char));
            temp[remainder + 1] = '\0';
            input[i] = strtoul(temp, NULL, 2);
            for (int j = 1; j < sequenceLength; j++)
            {
                memcpy(temp, bytes + i * (bits + 1) + j * INTSIZE, INTSIZE * sizeof(char));
                temp[BUFSIZE] = '\0';
                input[j * numberOfSequences + i] = strtoul(temp, NULL, 2);
            }
        }
    }
    free(bytes);
    return EXIT_SUCCESS;
}

cudaError_t generateRandoms(unsigned int* dev_input, const int sequenceLength, const int numberOfSequences);

// prints a few sequences from the beginning 
void head(unsigned int* input, float milliseconds, const int sequenceLength, const int numberOfSequences)
{
    int vecLen = sequenceLength > 3 ? 3 : sequenceLength;
    int vectors = numberOfSequences > 10 ? 10 : numberOfSequences;
    std::vector<std::bitset<32>>bitset(vecLen * vectors);
    for (int i = 0; i < vecLen; i++)
    {
        for (int j = 0; j < vectors; j++)
        {
            bitset[i * vectors + j] = std::bitset<32>(input[i * numberOfSequences + j]);
        }
    }

    std::cout << "Values: \n";
    for (int i = 0; i < vectors; i++)
    {
        for (int j = 0; j < vecLen; j++)
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

int main(int argc, char* argv[])
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
    float milliseconds = 0;
    float totalTime = 0;

    unsigned int* input = (unsigned int*)malloc(sizeof(int) * sequenceLength * numberOfSequences);
    unsigned int* dev_input = 0;

    cudaStatus = cudaMalloc((void**)&dev_input, sizeof(int) * sequenceLength * numberOfSequences);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    if (argc > 3)
    {
        fprintf(stderr, "too many arguments");
        goto Error;
    }
    if (argc >= 2)
    {
        // read file
        FILE* file = fopen(argv[1], "r");
        if (!file)
        {
            fprintf(stderr, "no such file");
            fclose(file);
            goto Error;
        }

        if (readFile(input, file, sequenceLength, numberOfSequences))
        {
            fclose(file);
            goto Error;
        }
    }
    if (argc == 3)
    {
        // display pairs
    }

    if (argc == 1)
    {
        // generate random sequences
        cudaEventRecord(start);
        cudaStatus = generateRandoms(dev_input, sequenceLength, numberOfSequences);
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

        //std::cout << "Generating random input completed. Time elapsed: " << milliseconds / 1000 << " seconds\n";

        // print
        head(input, milliseconds, sequenceLength, numberOfSequences);
    }
    else
    {
        // copy ints read from file to device
        cudaStatus = cudaMemcpy(dev_input, input, sizeof(int) * sequenceLength * numberOfSequences, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }
    

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

    cudaDeviceSynchronize();

    // print
    head(input, milliseconds, sequenceLength, numberOfSequences);

    //std::cout << "Radix sort completed. Time elapsed: " << milliseconds / 1000 << " seconds\n";

    // ---------------------------------------------------------------------------------

    // build a binary trie and search for all pairs with Hamming distance equal to 1
    cudaEventRecord(start);
    cudaStatus = hammingOne(dev_input, numberOfSequences, sequenceLength);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Building the tree completed. Time elapsed: " << milliseconds / 1000 << " seconds\n";

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
cudaError_t generateRandoms(unsigned int* dev_input, const int sequenceLength, const int numberOfSequences)
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
