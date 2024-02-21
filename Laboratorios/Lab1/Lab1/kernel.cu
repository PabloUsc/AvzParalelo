#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void print_gloabl_id_2d()
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_offset = (blockIdx.x + (blockIdx.y * gridDim.x)) * (blockDim.x * blockDim.y);
    int gid = tid + block_offset;

    printf("[DEVICE] threadIdx.x %d, threadIdx.y %d, blockIdx.x: %d, blockIdx.y: %d, gid: %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, gid);
}

int main()
{
    //initialization
    dim3 blockSize(4, 2, 1);
    dim3 gridSize(2, 2, 1);

    //launch kernel
    print_gloabl_id_2d << <gridSize, blockSize >> > ();

    //clean
    cudaDeviceReset();

    return 0;
}