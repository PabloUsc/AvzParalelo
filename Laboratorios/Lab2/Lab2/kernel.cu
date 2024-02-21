
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


__global__ void sumVectors(int* a, int* b, int* c, int size)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * threadIdx.z;
    int block_offset = (blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y)) * (blockDim.x * blockDim.y * blockDim.z);
    int gid = tid + block_offset;

    //printf("[DEVICE] x:%d, y:%d, z:%d, Bx:%d, By:%d, Bz:%d, gid: %d \n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, gid);
    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }
}

int main()
{
    //initialization
    dim3 blockSize(16, 8, 8);
    dim3 gridSize(10, 10, 10);

    const int data_count = 10000;
    int data_size = data_count * sizeof(int);

    int* a_cpu, * b_cpu, * c_cpu;
    int* a_dev, * b_dev, * c_dev;

    a_cpu = (int*)malloc(data_size);
    b_cpu = (int*)malloc(data_size);
    c_cpu = (int*)malloc(data_size);

    for (int i = 0; i < data_count; i++)
    {
        a_cpu[i] = rand();
        b_cpu[i] = rand();
    }

    //memory allocation
    cudaMalloc((void**)&a_dev, data_size);
    cudaMalloc((void**)&b_dev, data_size);
    cudaMalloc((void**)&c_dev, data_size);

    //transfer CPU host to GPU device
    cudaMemcpy(c_dev, c_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_dev, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_cpu, data_size, cudaMemcpyHostToDevice);

    //launch kernel
    sumVectors << <gridSize, blockSize >> > (a_dev,b_dev,c_dev,data_count);

    //transfer GPU device to CPU host
    cudaMemcpy(c_cpu, c_dev, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_cpu, a_dev, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_dev, data_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < data_count; i++)
    {
        printf("c[%d] = %d\n",i,c_cpu[i]);
    }

    //clean
    cudaDeviceReset();
    cudaFree(c_dev);
    cudaFree(a_dev);
    cudaFree(b_dev); 

    return 0;
}
