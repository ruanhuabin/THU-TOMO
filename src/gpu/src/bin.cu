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

__constant__ int bin_const_g[3];

__global__ void bin_gpu(cufftComplex* image_fft_g, cufftComplex* image_fft_bin_g)
{
	int Nx = bin_const_g[0];
	int Ny = bin_const_g[1];
	int bin = bin_const_g[2];
	
	int Nx_bin = int(ceil(float(Nx) / float(bin)));
	int Ny_bin = int(ceil(float(Ny) / float(bin)));

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;

	if (i < int(Nx_bin / 2 + 1))
    {
		image_fft_bin_g[j * int(Nx_bin / 2 + 1) + i].x = image_fft_g[j * int(Nx / 2 + 1) + i].x;
		image_fft_bin_g[j * int(Nx_bin / 2 + 1) + i].y = image_fft_g[j * int(Nx / 2 + 1) + i].y;
		if((Ny_bin % 2) == 0)
		{
			if(j == 0)
			{
				image_fft_bin_g[int(Ny_bin / 2) * int(Nx_bin / 2 + 1) + i].x = image_fft_g[int(Ny_bin / 2) * int(Nx / 2 + 1) + i].x;
				image_fft_bin_g[int(Ny_bin / 2) * int(Nx_bin / 2 + 1) + i].y = image_fft_g[int(Ny_bin / 2) * int(Nx / 2 + 1) + i].y;
			}
		}
		if(j != 0)
		{
			int j_ = Ny - j;
			image_fft_bin_g[(Ny_bin - 1 - (Ny - 1 - j_)) * int(Nx_bin / 2 + 1) + i].x = image_fft_g[j_ * int(Nx / 2 + 1) + i].x;
			image_fft_bin_g[(Ny_bin - 1 - (Ny - 1 - j_)) * int(Nx_bin / 2 + 1) + i].y = image_fft_g[j_ * int(Nx / 2 + 1) + i].y;
		}
	}
}

__global__ void image_normalization_bin_gpu(float* img)
{
	int Nx = int(bin_const_g[0]);
	int bin = int(bin_const_g[2]);
	int Nx_bin = int(ceil(float(Nx) / float(bin)));

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < Nx_bin) img[blockIdx.y * Nx_bin + x] /= float(gridDim.y * Nx_bin);
}

void bin_image(MRC* stack_orig, MRC* stack_bin, int bin)
{
    int Nx = stack_orig->getNx();
	int Ny = stack_orig->getNy();
	int Nz = stack_orig->getNz();
	
	int Nx_bin = int(ceil(float(Nx) / float(bin)));
	int Ny_bin = int(ceil(float(Ny) / float(bin)));

	int bin_const[3];
	bin_const[0] = Nx;
	bin_const[1] = Ny;
	bin_const[2] = bin;

	int deviceCount = 1;
	cudaStream_t stream[deviceCount];
	
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, 0));

	dim3 grid1(int((Nx_bin / 2 + 1) / deviceProp.maxThreadsPerBlock) +1, int(floor((Ny_bin - 1) / 2)) + 1);
	dim3 block1(deviceProp.maxThreadsPerBlock);

	dim3 grid2(int(Nx_bin / deviceProp.maxThreadsPerBlock) +1, Ny_bin);
	dim3 block2(deviceProp.maxThreadsPerBlock);

	cufftHandle p_fft[deviceCount], p_ifft[deviceCount];
	cufftComplex *image_fft_g[deviceCount], *image_fft_bin_g[deviceCount];
	float *image_g[deviceCount], *image_bin_g[deviceCount];

	for (int dev = 0; dev < deviceCount; dev++)
	{
		CHECK(cudaSetDevice(dev));
		CHECK(cudaStreamCreate(&(stream[dev])));

		CHECK(cudaMalloc((float**)&image_g[dev], sizeof(float) * Nx * Ny));
		CHECK(cudaMalloc((float**)&image_bin_g[dev], sizeof(float) * Nx_bin * Ny_bin));
		CHECK(cudaMalloc((cufftComplex**)&image_fft_g[dev], sizeof(cufftComplex) * (Nx / 2 + 1) * Ny));
		CHECK(cudaMalloc((cufftComplex**)&image_fft_bin_g[dev], sizeof(cufftComplex) * (Nx_bin / 2 + 1) * Ny_bin));
		CHECK(cudaMemcpyToSymbol(bin_const_g, &bin_const, 3 * sizeof(int)));

		cufftPlan2d(&p_fft[dev], Ny, Nx, CUFFT_R2C);
		cufftPlan2d(&p_ifft[dev], Ny_bin, Nx_bin, CUFFT_C2R);
		cufftSetStream(p_fft[dev], stream[dev]);
		cufftSetStream(p_ifft[dev], stream[dev]);
	}

	float* image[deviceCount], *image_bin[Nz];
	for (int i = 0; i < deviceCount; i++) image[i] = new float[Nx * Ny];
	for (int i = 0; i < Nz; i++) image_bin[i] = new float[Nx_bin * Ny_bin];

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
				bin_gpu <<< grid1, block1, 0, stream[dev] >>> (image_fft_g[dev], image_fft_bin_g[dev]);
				CHECK_CUFFT(cufftExecC2R(p_ifft[dev], image_fft_bin_g[dev], image_bin_g[dev]));
				image_normalization_bin_gpu <<< grid2, block2, 0, stream[dev] >>> (image_bin_g[dev]);
				CHECK(cudaMemcpyAsync(image_bin[n], image_bin_g[dev], sizeof(float) * Nx_bin * Ny_bin, cudaMemcpyDeviceToHost, stream[dev]));
			}
		}
	}

	for (int dev = 0; dev < deviceCount; dev++)
	{
		CHECK(cudaSetDevice(dev));
		cufftDestroy(p_fft[dev]);
		cufftDestroy(p_ifft[dev]);
		CHECK(cudaStreamDestroy(stream[dev]));
		cudaDeviceReset();
    }
    
    for(int n = 0; n < Nz; n++) stack_bin->write2DIm(image_bin[n], n);
}