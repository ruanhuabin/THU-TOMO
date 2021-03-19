#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cufft.h>
#include <sys/time.h>
#include <cstring>
#include "math.h"
#include "omp.h"
#include "../include/common.h"
#include "../../include/mrc.h"
using namespace std;

__constant__ float lp_const_g[2];

__global__ void low_pass_gpu(cufftComplex* image_fft_g)
{
    int Nx = int(lp_const_g[0]);
    int Ny = gridDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;

    float fft_x, fft_y;
    float lp = lp_const_g[1];

    if (i < int(Nx / 2 + 1))
    {
        fft_x = 2 * M_PI / Nx * i;
        fft_x = (fft_x > M_PI)? fft_x - 2 * M_PI: fft_x;
        fft_y = 2 * M_PI / Ny * j;
        fft_y = (fft_y > M_PI)? fft_y - 2 * M_PI: fft_y;
        if ((fft_x * fft_x + fft_y * fft_y) > lp * M_PI * lp * M_PI)
        {
            if ((fft_x * fft_x + fft_y * fft_y) > (1.2 * lp * M_PI) * (1.2 * lp * M_PI))
            {
                image_fft_g[i + j * int(Nx / 2 + 1)].x = 0;
                image_fft_g[i + j * int(Nx / 2 + 1)].y = 0;
            }
            else 
            {
                image_fft_g[i + j * int(Nx / 2 + 1)].x *= ((1 + cos((sqrt(fft_x * fft_x + fft_y * fft_y) - sqrt((lp*M_PI)*(lp*M_PI))) / sqrt((0.2*lp)*(0.2*lp)))) / 2.0);
                image_fft_g[i + j * int(Nx / 2 + 1)].y *= ((1 + cos((sqrt(fft_x * fft_x + fft_y * fft_y) - sqrt((lp*M_PI)*(lp*M_PI))) / sqrt((0.2*lp)*(0.2*lp)))) / 2.0);
            }
        }
	}
}

__global__ void image_normalization_lp_gpu(float* img)
{
	int Nx = int(lp_const_g[0]);
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < Nx) img[blockIdx.y * Nx + x] /= float(gridDim.y * Nx);
}


void low_pass(MRC* stack_orig, MRC* stack_lp, float lp)
{
    int Nx = stack_orig->getNx();
	int Ny = stack_orig->getNy();
	int Nz = stack_orig->getNz();
	
	float lp_const[2];
	lp_const[0] = float(Nx);
	lp_const[1] = lp;

	int deviceCount = 1;
	cudaStream_t stream[deviceCount];

	cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    
	dim3 grid1(int((Nx / 2 + 1) / deviceProp.maxThreadsPerBlock) +1, Ny);
    dim3 block1(deviceProp.maxThreadsPerBlock);
	
	dim3 grid2(int(Nx / deviceProp.maxThreadsPerBlock) +1, Ny);
    dim3 block2(deviceProp.maxThreadsPerBlock);

	cufftHandle p_fft[deviceCount], p_ifft[deviceCount];
	cufftComplex* image_fft_g[deviceCount];
	float* image_g[deviceCount];

	for (int dev = 0; dev < deviceCount; dev++)
	{
		CHECK(cudaSetDevice(dev));
		CHECK(cudaStreamCreate(&(stream[dev])));

		CHECK(cudaMalloc((float**)&image_g[dev], sizeof(float) * Nx * Ny));
		CHECK(cudaMalloc((cufftComplex**)&image_fft_g[dev], sizeof(cufftComplex) * (Nx / 2 + 1) * Ny));
		CHECK(cudaMemcpyToSymbol(lp_const_g, &lp_const, 2 * sizeof(float)));

		cufftPlan2d(&p_fft[dev], Ny, Nx, CUFFT_R2C);
		cufftSetStream(p_fft[dev], stream[dev]);
		cufftPlan2d(&p_ifft[dev], Ny, Nx, CUFFT_C2R);
        cufftSetStream(p_ifft[dev], stream[dev]);
	}

	float *image_lp[Nz], *image[deviceCount];
	for(int i = 0; i < Nz; i++) image_lp[i] = new float[Nx * Ny];
	for(int i = 0; i < deviceCount; i++) image[i] = new float[Nx * Ny];

	for (int i = 0; i < Nz; i += deviceCount)
	{
        #pragma omp parallel num_threads(deviceCount)
		{
			int dev = omp_get_thread_num();
			int n = i + dev;
			if ((dev < deviceCount) && (n < Nz))
			{
				stack_orig->read2DIm_32bit(image[dev], n);
				CHECK(cudaSetDevice(dev));
				CHECK(cudaMemcpyAsync(image_g[dev], image[dev], sizeof(float) * Nx * Ny, cudaMemcpyHostToDevice, stream[dev]));
				CHECK_CUFFT(cufftExecR2C(p_fft[dev], image_g[dev], image_fft_g[dev]));
				low_pass_gpu <<< grid1, block1, 0, stream[dev] >>> (image_fft_g[dev]);
				CHECK_CUFFT(cufftExecC2R(p_ifft[dev], image_fft_g[dev], image_g[dev]));
				image_normalization_lp_gpu <<< grid2, block2, 0, stream[dev] >>> (image_g[dev]);
				CHECK(cudaMemcpyAsync(image_lp[n], image_g[dev], sizeof(float) * Nx * Ny, cudaMemcpyDeviceToHost, stream[dev]));
			}
		}
	}

	for (int dev = 0; dev < deviceCount; dev++)
	{
		free(image[dev]);
		CHECK(cudaSetDevice(dev));
		CHECK(cudaStreamSynchronize(stream[dev]));
	}

	for (int dev = 0; dev < deviceCount; dev++)
    {
        CHECK(cudaSetDevice(dev));
        cufftDestroy(p_fft[dev]);
		cufftDestroy(p_ifft[dev]);
        CHECK(cudaStreamDestroy(stream[dev]));
        cudaDeviceReset();
    }

    for(int n = 0; n < Nz; n++) stack_lp->write2DIm(image_lp[n], n);
}