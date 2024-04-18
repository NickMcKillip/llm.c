#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"
#include <iostream>

// ----------------------------------------------------------------------------
// CPU code reference lol

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
__global__ void residual_forward_kernel(float* out, const float* inp1, const float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = inp1[idx] + inp2[idx];
    }
}

// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
__global__ void residual_forward_kernel_2(float* out, const float* inp1, const float* inp2, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx<N; idx+=stride) {
        out[idx] = inp1[idx] + inp2[idx];
    }
}

// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
__global__ void residual_forward_kernel_3(float* out, float* inp1, float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N/2) {
        float2 inp1_r;
        float2 inp2_r;
        inp1_r = reinterpret_cast<float2*>(inp1)[idx];
        inp2_r = reinterpret_cast<float2*>(inp2)[idx];
        inp1_r.x = inp1_r.x + inp2_r.x;
        inp1_r.y = inp1_r.y + inp2_r.y;
        reinterpret_cast<float2*>(out)[idx] = inp1_r;
    } else if (idx==N/2 && N%2==1) {
        out[N-1] = inp1[N-1] + inp2[N-1];
    }
}

__global__ void residual_forward_kernel_4(float* out, float* inp1, float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N/4) {
        float4 inp1_r;
        float4 inp2_r;
        inp1_r = reinterpret_cast<float4*>(inp1)[idx];
        inp2_r = reinterpret_cast<float4*>(inp2)[idx];
        inp1_r.x = inp1_r.x + inp2_r.x;
        inp1_r.y = inp1_r.y + inp2_r.y;
        inp1_r.z = inp1_r.z + inp2_r.z;
        inp1_r.w = inp1_r.w + inp2_r.w;
        reinterpret_cast<float4*>(out)[idx] = inp1_r;
        return;
    }
    if (idx == N/4) {
        int remainder = N%4;
        while (remainder) {
            int i = N - remainder;
            out[i] = inp1[i] + inp2[i];
            remainder--;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void residual_forward1(float* out, const float* inp1, const float* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    residual_forward_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void residual_forward2(float* out, const float* inp1, const float* inp2, int N, const int block_size) {
    residual_forward_kernel_2<<<1200, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void residual_forward3(float* out, float* inp1, float* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size*2);
    residual_forward_kernel_3<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void residual_forward4(float* out, float* inp1, float* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size*4);
    residual_forward_kernel_4<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}


// kernel version dispatch
void residual_forward(int kernel_num,
                  float* out,
                  float* inp1,
                  float* inp2,
                  int N,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            residual_forward1(out, inp1, inp2, N, block_size);
            break;
        case 2:
            residual_forward2(out, inp1, inp2, N, block_size);
            break;
        case 3:
            residual_forward3(out, inp1, inp2, N, block_size);
            break;
        case 4:
            residual_forward4(out, inp1, inp2, N, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* inp1 = make_random_float(B * T * C);
    float* inp2 = make_random_float(B * T * C);

    // move to GPU
    float* d_out;
    float* d_inp1;
    float* d_inp2;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp1, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp2, B * T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp1, inp1, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_inp2, inp2, B * T * C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    int repeat_times = 1000;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    if (argc > 2) {
        repeat_times = atoi(argv[2]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    residual_forward_cpu(out, inp1, inp2, B * T * C);


    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        residual_forward(kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size);
        validate_result(d_out, out, "out", B * T * C, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        float elapsed_time = benchmark_kernel(repeat_times, residual_forward,
                                              kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size
                                              );

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 2 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 3 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp1);
    free(inp2);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp1));
    cudaCheck(cudaFree(d_inp2));

    return 0;
}