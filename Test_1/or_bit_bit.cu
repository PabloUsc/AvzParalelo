
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void orBitByBit(unsigned char *a, unsigned char *b, unsigned char *c, int size)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * threadIdx.z;
    int block_offset = (blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y)) * (blockDim.x * blockDim.y * blockDim.z);
    int gid = tid + block_offset;

    if (gid < size)
    {
        c[gid] = a[gid] | b[gid];
    }
}

int main()
{
    //Initialization
    const int size_in_bytes = 8;
    unsigned char a_cpu[size_in_bytes];
    unsigned char b_cpu[size_in_bytes];
    unsigned char c_cpu[size_in_bytes];
    unsigned char a_dev[size_in_bytes];
    unsigned char b_dev[size_in_bytes];
    unsigned char c_dev[size_in_bytes];
    int N_size = size_in_bytes / sizeof(unsigned char);
    int threadsTotal = 1024 * 16 * 46;
    int threadsBlock = threadsTotal / 16;
    dim3 gridSize(4,2,2);
    dim3 blockSize(16, 8, 8);

    //Memory allocation
    cudaMalloc((void**)&a_dev, size_in_bytes);
    cudaMalloc((void**)&b_dev, size_in_bytes);
    cudaMalloc((void**)&c_dev, size_in_bytes);

    //transfer CPU to GPU
    cudaMemcpy(a_dev, a_cpu, size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_cpu, size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(c_dev, c_cpu, size_in_bytes, cudaMemcpyHostToDevice);

    //Kernel
    orBitByBit << <gridSize, blockSize >> > (a_dev,b_dev,c_dev,N_size);

    //transfer GPU device to CPU host
    cudaMemcpy(c_cpu, c_dev, size_in_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_cpu, a_dev, size_in_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_dev, size_in_bytes, cudaMemcpyDeviceToHost);

    printf("%d\n", c_cpu);
    printf("Total threads: %d\n", threadsTotal);
    printf("Threads per block: %d\n", threadsBlock);

    //Clean
    cudaDeviceReset();
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    return 0;
}

