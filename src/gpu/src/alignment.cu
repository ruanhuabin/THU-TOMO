#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sys/time.h>
#include "../../include/mrc.h"
#include "../include/common.h"
using namespace std;

__constant__ int para_const[3];

void standardize_image(float* image, int Nx, int Ny) // standardize to 0-mean & 1-std
{
    double sum = 0.0;
    double sum2 = 0.0;
    // loop: Nx*Ny (whole image)
    for (int i = 0; i < Nx * Ny; i++)
    {
        sum += double(image[i]);
        sum2 += (double(image[i]) * double(image[i]));
    }
    double mean = sum / (Nx * Ny);
    double mean2 = sum2 / (Nx * Ny);
    double std = sqrt(mean2 - mean * mean);
    // loop: Nx*Ny (whole image)
    for (int i = 0; i < Nx * Ny; i++)
    {
        image[i] = float((double(image[i]) - mean) / std);
    }
}

void patch_ref_init(int patch_size_coarse, int patch_Nx_coarse, int patch_Ny_coarse, int Nx, int Ny,
                    int* patch_x_ref_coarse, int* patch_y_ref_coarse)
{
    int patch_size_coarse_half = int(patch_size_coarse / 2);
    float patch_dx_coarse = float(Nx - 1) / float(patch_Nx_coarse + 1);
    float patch_dy_coarse = float(Ny - 1) / float(patch_Ny_coarse + 1);
    float patch_dx_orig_coarse = patch_dx_coarse;
    float patch_dy_orig_coarse = patch_dy_coarse;
    float patch_dx_offset_coarse = 0;
    float patch_dy_offset_coarse = 0;
    if (patch_dx_coarse < patch_size_coarse_half)
    {
        patch_dx_offset_coarse = patch_size_coarse_half - floor(patch_dx_coarse);
        patch_dx_coarse = float(Nx - 1 - patch_size_coarse) / float(patch_Nx_coarse + 1 - 2);
    }
    if (patch_dy_coarse < patch_size_coarse_half)
    {
        patch_dy_offset_coarse = patch_size_coarse_half - floor(patch_dy_coarse);
        patch_dy_coarse = float(Ny - 1 - patch_size_coarse) / float(patch_Ny_coarse + 1 - 2);
    }
    int t = 0;
    for (float x = patch_dx_orig_coarse + patch_dx_offset_coarse; x < Nx - patch_dx_orig_coarse - patch_dx_offset_coarse; x += patch_dx_coarse)
    {
        for (float y = patch_dy_orig_coarse + patch_dy_offset_coarse; y < Ny - patch_dy_orig_coarse - patch_dy_offset_coarse; y += patch_dy_coarse)
        {
            patch_x_ref_coarse[t] = floor(x);
            patch_y_ref_coarse[t] = floor(y);
            t++;
        }
    }
}

__global__ void compute_correlation(float* patch_fix, float* image_move, float* cc, float* cc_de, int* para_xy)
{
    int patch_num = gridDim.x;
    int patch_trans_coarse = (gridDim.x - 1) / 2;
    int patch_size_g = gridDim.z;
    int patch_size_coarse = para_const[2];
    int patch_size_coarse_half = int(patch_size_coarse / 2);

    int patch_x_ref = para_xy[0];
    int patch_y_ref = para_xy[1];
    int stack_orig_nx = para_const[0];
    int stack_orig_ny = para_const[1];

    int jj = blockIdx.x - patch_trans_coarse;
    int ii = blockIdx.y - patch_trans_coarse;
    int j = blockIdx.z;
    int i = threadIdx.x;

    float patch_move_jj_ii = 0;

    if ((j >= patch_size_coarse) || (i >= patch_size_coarse))
    {
        cc[((blockIdx.x * patch_num + blockIdx.y) * patch_size_g + blockIdx.z) * patch_size_g + i] = 0;
        cc_de[((blockIdx.x * patch_num + blockIdx.y) * patch_size_g + blockIdx.z) * patch_size_g + i] = 0;
        return;
    }

    if (patch_x_ref + i - patch_size_coarse_half + ii >= 0 && patch_x_ref + i - patch_size_coarse_half + ii < stack_orig_nx &&
        patch_y_ref + j - patch_size_coarse_half + jj >= 0 && patch_y_ref + j - patch_size_coarse_half + jj < stack_orig_ny)
    {
        patch_move_jj_ii = image_move[(patch_x_ref + i - patch_size_coarse_half + ii) + (patch_y_ref + j - patch_size_coarse_half + jj) * stack_orig_nx];
    }
    cc[((blockIdx.x * patch_num + blockIdx.y) * patch_size_g + blockIdx.z) * patch_size_g + i] = patch_move_jj_ii * patch_fix[j * patch_size_coarse + i];
    cc_de[((blockIdx.x * patch_num + blockIdx.y) * patch_size_g + blockIdx.z) * patch_size_g + i] = patch_move_jj_ii * patch_move_jj_ii;
}

__global__ void reduceUnrolling8(float* cc_map, float* cc_map_reduce, float* cc_de_map, float* cc_de_map_reduce)
{
    int patch_num = gridDim.x;
    int patch_size_coarse = blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.z * blockDim.x * 8 + threadIdx.x;

    float* g_idata_cc = cc_map + (blockIdx.x * patch_num + blockIdx.y) * patch_size_coarse * patch_size_coarse;
    float* g_idata_cc_de = cc_de_map + (blockIdx.x * patch_num + blockIdx.y) * patch_size_coarse * patch_size_coarse;

    float* idata_cc = g_idata_cc + blockIdx.z * blockDim.x * 8;
    float* idata_cc_de = g_idata_cc_de + blockIdx.z * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < patch_size_coarse * patch_size_coarse)
    {
        float a1, a2, a3, a4, b1, b2, b3, b4;
        a1 = g_idata_cc[idx];
        a2 = g_idata_cc[idx + blockDim.x];
        a3 = g_idata_cc[idx + 2 * blockDim.x];
        a4 = g_idata_cc[idx + 3 * blockDim.x];
        b1 = g_idata_cc[idx + 4 * blockDim.x];
        b2 = g_idata_cc[idx + 5 * blockDim.x];
        b3 = g_idata_cc[idx + 6 * blockDim.x];
        b4 = g_idata_cc[idx + 7 * blockDim.x];
        g_idata_cc[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
        a1 = g_idata_cc_de[idx];
        a2 = g_idata_cc_de[idx + blockDim.x];
        a3 = g_idata_cc_de[idx + 2 * blockDim.x];
        a4 = g_idata_cc_de[idx + 3 * blockDim.x];
        b1 = g_idata_cc_de[idx + 4 * blockDim.x];
        b2 = g_idata_cc_de[idx + 5 * blockDim.x];
        b3 = g_idata_cc_de[idx + 6 * blockDim.x];
        b4 = g_idata_cc_de[idx + 7 * blockDim.x];
        g_idata_cc_de[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata_cc[tid] += idata_cc[tid + stride];
            idata_cc_de[tid] += idata_cc_de[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        cc_map_reduce[(blockIdx.x * patch_num + blockIdx.y) * gridDim.z + blockIdx.z] = idata_cc[0];
        cc_de_map_reduce[(blockIdx.x * patch_num + blockIdx.y) * gridDim.z + blockIdx.z] = idata_cc_de[0];
    }
}

__global__ void sum_reduced(float* cc_map_reduce, float* cc_map_sum, float* cc_de_map_reduce, float* cc_de_map_sum)
{
    int patch_num = gridDim.x;
    int patch_size_coarse = blockDim.x;

    int tid = threadIdx.x;

    float* idata_cc = cc_map_reduce + (blockIdx.x * patch_num + blockIdx.y) * patch_size_coarse;
    float* idata_cc_de = cc_de_map_reduce + (blockIdx.x * patch_num + blockIdx.y) * patch_size_coarse;

    for (int stride = patch_size_coarse / 2; stride > 0; stride = stride / 2)
    {
        if (tid < stride)
        {
            idata_cc[tid] += idata_cc[tid + stride];
            idata_cc_de[tid] += idata_cc_de[tid + stride];
        }
        __syncthreads();
    }
    cc_map_sum[blockIdx.x * patch_num + blockIdx.y] = idata_cc[0];
    cc_de_map_sum[blockIdx.x * patch_num + blockIdx.y] = idata_cc_de[0];
}


void coarse_alignment(int ref_n, int patch_size_coarse, int patch_trans_coarse, int patch_Nx_coarse, int patch_Ny_coarse, int* patch_dx_sum_all, int* patch_dy_sum_all,
    MRC* stack_orig, MRC* stack_coarse, string patch_save_path)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaStream_t stream[deviceCount];

    int patch_size_coarse_half = int(patch_size_coarse / 2);
    int patch_size_g = pow(2, int(log(patch_size_coarse) / log(2)) + 1);
    int patch_num_g = (2 * patch_trans_coarse + 1) * (2 * patch_trans_coarse + 1);
    int iteration = int(patch_Nx_coarse * patch_Ny_coarse / deviceCount);
    int para[3];
    para[0] = stack_orig->getNx();
    para[1] = stack_orig->getNy();
    para[2] = patch_size_coarse;

    float* patch_fix[deviceCount], *cc[deviceCount], *cc_de[deviceCount];
    float cc_de_fix[deviceCount], cc_max[deviceCount];
    int* para_xy[deviceCount];
    int cc_max_x[deviceCount], cc_max_y[deviceCount];

    for (int dev = 0; dev < deviceCount; dev++)
    {
        patch_fix[dev] = new float[patch_size_coarse * patch_size_coarse];
        cc[dev] = new float[patch_num_g];
        cc_de[dev] = new float[patch_num_g];
        para_xy[dev] = new int[2];
    }

    float* patch_g[deviceCount], *image_move_g[deviceCount];
    float* cc_g[deviceCount], *cc_de_g[deviceCount], *cc_reduce_g[deviceCount], *cc_de_reduce_g[deviceCount], *cc_sum_g[deviceCount], *cc_de_sum_g[deviceCount];

    for (int dev = 0; dev < deviceCount; dev++)
    {
        CHECK(cudaSetDevice(dev));
        CHECK(cudaStreamCreate(&(stream[dev])));

        cudaMalloc((float**)&patch_g[dev], sizeof(float) * patch_size_coarse * patch_size_coarse);
        cudaMalloc((float**)&image_move_g[dev], sizeof(float) * stack_orig->getNx() * stack_orig->getNy());

        cudaMalloc((float**)&cc_g[dev], sizeof(float) * patch_num_g * patch_size_g * patch_size_g);
        cudaMalloc((float**)&cc_de_g[dev], sizeof(float) * patch_num_g * patch_size_g * patch_size_g);
        cudaMalloc((float**)&cc_reduce_g[dev], sizeof(float) * patch_num_g * patch_size_g);
        cudaMalloc((float**)&cc_de_reduce_g[dev], sizeof(float) * patch_num_g * patch_size_g);
        cudaMalloc((float**)&cc_sum_g[dev], sizeof(float) * patch_num_g);
        cudaMalloc((float**)&cc_de_sum_g[dev], sizeof(float) * patch_num_g);

        cudaMalloc((int**)&para_xy[dev], sizeof(int) * 2);
        cudaMemcpyToSymbol(para_const, para, sizeof(int) * 3);
    }

    dim3 grid1(2 * patch_trans_coarse + 1, 2 * patch_trans_coarse + 1, patch_size_g);
    dim3 block1(patch_size_g);
    dim3 grid2(2 * patch_trans_coarse + 1, 2 * patch_trans_coarse + 1, patch_size_g / 8);
    dim3 block2(patch_size_g);
    dim3 grid3(2 * patch_trans_coarse + 1, 2 * patch_trans_coarse + 1);
    dim3 block3(patch_size_g / 8);

    int patch_x_ref_coarse[patch_Nx_coarse * patch_Ny_coarse];
    int patch_y_ref_coarse[patch_Nx_coarse * patch_Ny_coarse];
    int patch_x_converse_coarse[patch_Nx_coarse * patch_Ny_coarse];
    int patch_y_converse_coarse[patch_Nx_coarse * patch_Ny_coarse];
    int patch_deltaX_coarse[patch_Nx_coarse * patch_Ny_coarse];
    int patch_deltaY_coarse[patch_Nx_coarse * patch_Ny_coarse];

    patch_ref_init(patch_size_coarse, patch_Nx_coarse, patch_Ny_coarse, stack_orig->getNx(), stack_orig->getNy(),
     patch_x_ref_coarse, patch_y_ref_coarse);

    float* image_now = new float[stack_orig->getNx() * stack_orig->getNy()];
    float* image_next = new float[stack_orig->getNx() * stack_orig->getNy()];
    float* image_coarse = new float[stack_orig->getNx() * stack_orig->getNy()];
    int patch_dx_sum = 0;
    int patch_dy_sum = 0;

    // refine from ref_n to Nz
    for (int n = ref_n; n < stack_orig->getNz() - 1; n++)
    {
        // standardize image
        if (n == ref_n)
        {
            stack_orig->read2DIm_32bit(image_now, n);
            stack_orig->read2DIm_32bit(image_next, n + 1);
            standardize_image(image_now, stack_orig->getNx(), stack_orig->getNy());
            standardize_image(image_next, stack_orig->getNx(), stack_orig->getNy());
        }
        else
        {
            memcpy(image_now, image_next, sizeof(float) * stack_orig->getNx() * stack_orig->getNy());
            stack_orig->read2DIm_32bit(image_next, n + 1);
            standardize_image(image_next, stack_orig->getNx(), stack_orig->getNy());
        }

        int t, patch_availN = patch_Nx_coarse * patch_Ny_coarse;
        // forward searching, move: image_next, fix: image_now
        for (int dev = 0; dev < deviceCount; dev++)
        {
            cudaMemcpyAsync(image_move_g[dev], image_next, sizeof(float) * stack_orig->getNx() * stack_orig->getNy(), cudaMemcpyHostToDevice, stream[dev]);
        }

        for (int iter = 0; iter < iteration; iter++)
        {
            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                cc_de_fix[dev] = 0;
                for (int j = 0; j < patch_size_coarse; j++)    // extrack original patch
                {
                    for (int i = 0; i < patch_size_coarse; i++)   // can be be optimized?
                    {
                        patch_fix[dev][j * patch_size_coarse + i] = image_now[(patch_x_ref_coarse[t] + i - patch_size_coarse_half) +
                                                                    (patch_y_ref_coarse[t] + j - patch_size_coarse_half) * stack_orig->getNx()];
                        cc_de_fix[dev] += patch_fix[dev][j * patch_size_coarse + i] * patch_fix[dev][j * patch_size_coarse + i];
                    }
                }
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                CHECK(cudaSetDevice(dev));
                cudaMemcpyAsync(patch_g[dev], patch_fix[dev], sizeof(float) * patch_size_coarse * patch_size_coarse, cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev], patch_x_ref_coarse + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev] + 1, patch_y_ref_coarse + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaSetDevice(dev));
                compute_correlation <<< grid1, block1, 0, stream[dev] >>> (patch_g[dev], image_move_g[dev], cc_g[dev], cc_de_g[dev], para_xy[dev]);
                reduceUnrolling8 <<< grid2, block2, 0, stream[dev] >>> (cc_g[dev], cc_reduce_g[dev], cc_de_g[dev], cc_de_reduce_g[dev]);
                sum_reduced <<< grid3, block3, 0, stream[dev] >>> (cc_reduce_g[dev], cc_sum_g[dev], cc_de_reduce_g[dev], cc_de_sum_g[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                cudaMemcpyAsync(cc[dev], cc_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
                cudaMemcpyAsync(cc_de[dev], cc_de_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                cc_max[dev] = 0;
                cc_max_x[dev] = 0;
                cc_max_y[dev] = 0;
                for (int i = 0; i < patch_num_g; i++)
                {
                    cc[dev][i] = cc[dev][i] / sqrt(cc_de[dev][i] * cc_de_fix[dev]);
                    if (cc[dev][i] > cc_max[dev])
                    {
                        cc_max[dev] = cc[dev][i];
                        cc_max_x[dev] = i % (2 * patch_trans_coarse + 1) - patch_trans_coarse;
                        cc_max_y[dev] = int(i / (2 * patch_trans_coarse + 1)) - patch_trans_coarse;
                    }
                }
                //cout << n << " " << t << " " << cc_max[dev] << " " << cc_max_x[dev] << " " << cc_max_y[dev] << endl;
                patch_deltaX_coarse[t] = cc_max_x[dev];
                patch_deltaY_coarse[t] = cc_max_y[dev];
                patch_x_converse_coarse[t] = patch_x_ref_coarse[t] + cc_max_x[dev];
                patch_y_converse_coarse[t] = patch_y_ref_coarse[t] + cc_max_y[dev];
            }
            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaStreamSynchronize(stream[dev]));
            }
        }


        // backward searching, move: image_now, fix: image_next
        for (int dev = 0; dev < deviceCount; dev++)
        {
            cudaMemcpyAsync(image_move_g[dev], image_now, sizeof(float) * stack_orig->getNx() * stack_orig->getNy(), cudaMemcpyHostToDevice, stream[dev]);
        }

        for (int iter = 0; iter < iteration; iter++)
        {
            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                cc_de_fix[dev] = 0;
                for (int j = 0; j < patch_size_coarse; j++)
                {
                    for (int i = 0; i < patch_size_coarse; i++)    // extract the current patch
                    {
                        if (patch_x_converse_coarse[t] + i - patch_size_coarse_half >= 0 && patch_x_converse_coarse[t] + i - patch_size_coarse_half < stack_orig->getNx() &&
                            patch_y_converse_coarse[t] + j - patch_size_coarse_half >= 0 && patch_y_converse_coarse[t] + j - patch_size_coarse_half < stack_orig->getNy())
                        {
                            patch_fix[dev][j * patch_size_coarse + i] = image_next[(patch_x_converse_coarse[t] + i - patch_size_coarse_half) +
                                                                        (patch_y_converse_coarse[t] + j - patch_size_coarse_half) * stack_orig->getNx()];
                            cc_de_fix[dev] += patch_fix[dev][j * patch_size_coarse + i] * patch_fix[dev][j * patch_size_coarse + i];
                        }
                    }
                }
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                CHECK(cudaSetDevice(dev));
                cudaMemcpyAsync(patch_g[dev], patch_fix[dev], sizeof(float) * patch_size_coarse * patch_size_coarse, cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev], patch_x_converse_coarse + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev] + 1, patch_y_converse_coarse + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaSetDevice(dev));
                compute_correlation <<< grid1, block1, 0, stream[dev] >>> (patch_g[dev], image_move_g[dev], cc_g[dev], cc_de_g[dev], para_xy[dev]);
                reduceUnrolling8 <<< grid2, block2, 0, stream[dev] >>> (cc_g[dev], cc_reduce_g[dev], cc_de_g[dev], cc_de_reduce_g[dev]);
                sum_reduced <<< grid3, block3, 0, stream[dev] >>> (cc_reduce_g[dev], cc_sum_g[dev], cc_de_reduce_g[dev], cc_de_sum_g[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                cudaMemcpyAsync(cc[dev], cc_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
                cudaMemcpyAsync(cc_de[dev], cc_de_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                cc_max[dev] = 0;
                cc_max_x[dev] = 0;
                cc_max_y[dev] = 0;
                for (int i = 0; i < patch_num_g; i++)
                {
                    cc[dev][i] = cc[dev][i] / sqrt(cc_de[dev][i] * cc_de_fix[dev]);
                    if (cc[dev][i] > cc_max[dev])
                    {
                        cc_max[dev] = cc[dev][i];
                        cc_max_x[dev] = i % (2 * patch_trans_coarse + 1) - patch_trans_coarse;
                        cc_max_y[dev] = int(i / (2 * patch_trans_coarse + 1)) - patch_trans_coarse;
                    }
                }
                //cout << n << " " << t << " " << cc_max[dev] << " " << cc_max_x[dev] << " " << cc_max_y[dev] << endl;
                if (abs(patch_deltaX_coarse[t] + cc_max_x[dev]) < 3 && abs(patch_deltaY_coarse[t] + cc_max_y[dev]) < 3)
                {
                    //cout << "accepted patch" << endl;
                }
                else
                {
                    patch_availN--;
                    patch_deltaX_coarse[t] = 0;
                    patch_deltaY_coarse[t] = 0;
                    //cout << "rejected patch" << endl;
                }
            }
            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaStreamSynchronize(stream[dev]));
            }
        }

        int patch_dx_avg = 0, patch_dy_avg = 0;
        // loop: patch_Nx_coarse*patch_Ny_coarse (number of patches)
        for (t = 0; t < patch_Nx_coarse * patch_Ny_coarse; t++)
        {
            patch_dx_avg += patch_deltaX_coarse[t];
            patch_dy_avg += patch_deltaY_coarse[t];
        }
        patch_dx_avg /= patch_availN;
        patch_dy_avg /= patch_availN;
        patch_dx_sum += patch_dx_avg; // 找到与0度图的平移量（一连串平移累加）
        patch_dy_sum += patch_dy_avg;
        patch_dx_sum_all[n + 1] = patch_dx_sum;
        patch_dy_sum_all[n + 1] = patch_dy_sum;

        cout << n << ":" << endl;
        cout << "adjacent: (" << patch_dx_avg << "," << patch_dy_avg << ")" << endl;
        cout << "all: (" << patch_dx_sum << "," << patch_dy_sum << ")" << endl;

        float image_avg = 0;
        for (int j = 0; j < stack_orig->getNy(); j++)
        {
            for (int i = 0; i < stack_orig->getNx(); i++)   // calculate average
            {
                image_avg += image_next[i + j * stack_orig->getNx()];
            }
        }
        image_avg = image_avg / (stack_orig->getNx() * stack_orig->getNy());
        // loop: Nx*Ny (whole image)
        for (int j = 0; j < stack_orig->getNy(); j++)
        {
            for (int i = 0; i < stack_orig->getNx(); i++)
            {
                if (i + patch_dx_sum >= 0 && i + patch_dx_sum < stack_orig->getNx() && j + patch_dy_sum >= 0 && j + patch_dy_sum < stack_orig->getNy())
                {
                    image_coarse[i + j * stack_orig->getNx()] = image_next[(i + patch_dx_sum) + (j + patch_dy_sum) * stack_orig->getNx()];
                }
                else
                {
                    image_coarse[i + j * stack_orig->getNx()] = image_avg;
                }
            }
        }
        if (n == ref_n)
        {
            stack_coarse->write2DIm(image_now, n);
        }
        stack_coarse->write2DIm(image_coarse, n + 1);
    }

    patch_dx_sum = 0;
    patch_dy_sum = 0;

    // refine from ref_n to 0
    for (int n = ref_n; n > 0; n--)
    {
        // standardize image
        if (n == ref_n)
        {
            stack_orig->read2DIm_32bit(image_now, n);
            stack_orig->read2DIm_32bit(image_next, n - 1);
            standardize_image(image_now, stack_orig->getNx(), stack_orig->getNy());
            standardize_image(image_next, stack_orig->getNx(), stack_orig->getNy());
        }
        else
        {
            memcpy(image_now, image_next, sizeof(float) * stack_orig->getNx() * stack_orig->getNy());
            stack_orig->read2DIm_32bit(image_next, n - 1);
            standardize_image(image_next, stack_orig->getNx(), stack_orig->getNy());
        }

        int t, patch_availN = patch_Nx_coarse * patch_Ny_coarse;
        // forward searching, move: image_next, fix: image_now
        for (int dev = 0; dev < deviceCount; dev++)
        {
            cudaMemcpyAsync(image_move_g[dev], image_next, sizeof(float) * stack_orig->getNx() * stack_orig->getNy(), cudaMemcpyHostToDevice, stream[dev]);
        }

        for (int iter = 0; iter < iteration; iter++)
        {
            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                cc_de_fix[dev] = 0;
                for (int j = 0; j < patch_size_coarse; j++)    // extrack original patch
                {
                    for (int i = 0; i < patch_size_coarse; i++)   // can be be optimized?
                    {
                        patch_fix[dev][j * patch_size_coarse + i] = image_now[(patch_x_ref_coarse[t] + i - patch_size_coarse_half) +
                                                                    (patch_y_ref_coarse[t] + j - patch_size_coarse_half) * stack_orig->getNx()];
                        cc_de_fix[dev] += patch_fix[dev][j * patch_size_coarse + i] * patch_fix[dev][j * patch_size_coarse + i];
                    }
                }
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                CHECK(cudaSetDevice(dev));
                cudaMemcpyAsync(patch_g[dev], patch_fix[dev], sizeof(float) * patch_size_coarse * patch_size_coarse, cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev], patch_x_ref_coarse + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev] + 1, patch_y_ref_coarse + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaSetDevice(dev));
                compute_correlation <<< grid1, block1, 0, stream[dev] >>> (patch_g[dev], image_move_g[dev], cc_g[dev], cc_de_g[dev], para_xy[dev]);
                reduceUnrolling8 <<< grid2, block2, 0, stream[dev] >>> (cc_g[dev], cc_reduce_g[dev], cc_de_g[dev], cc_de_reduce_g[dev]);
                sum_reduced <<< grid3, block3, 0, stream[dev] >>> (cc_reduce_g[dev], cc_sum_g[dev], cc_de_reduce_g[dev], cc_de_sum_g[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                cudaMemcpyAsync(cc[dev], cc_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
                cudaMemcpyAsync(cc_de[dev], cc_de_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                cc_max[dev] = 0;
                cc_max_x[dev] = 0;
                cc_max_y[dev] = 0;
                for (int i = 0; i < patch_num_g; i++)
                {
                    cc[dev][i] = cc[dev][i] / sqrt(cc_de[dev][i] * cc_de_fix[dev]);
                    if (cc[dev][i] > cc_max[dev])
                    {
                        cc_max[dev] = cc[dev][i];
                        cc_max_x[dev] = i % (2 * patch_trans_coarse + 1) - patch_trans_coarse;
                        cc_max_y[dev] = int(i / (2 * patch_trans_coarse + 1)) - patch_trans_coarse;
                    }
                }
                //cout << n << " " << t << " " << cc_max[dev] << " " << cc_max_x[dev] << " " << cc_max_y[dev] << endl;
                patch_deltaX_coarse[t] = cc_max_x[dev];
                patch_deltaY_coarse[t] = cc_max_y[dev];
                patch_x_converse_coarse[t] = patch_x_ref_coarse[t] + cc_max_x[dev];
                patch_y_converse_coarse[t] = patch_y_ref_coarse[t] + cc_max_y[dev];
            }
            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaStreamSynchronize(stream[dev]));
            }
        }


        // backward searching, move: image_now, fix: image_next
        for (int dev = 0; dev < deviceCount; dev++)
        {
            cudaMemcpyAsync(image_move_g[dev], image_now, sizeof(float) * stack_orig->getNx() * stack_orig->getNy(), cudaMemcpyHostToDevice, stream[dev]);
        }

        for (int iter = 0; iter < iteration; iter++)
        {
            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                cc_de_fix[dev] = 0;
                for (int j = 0; j < patch_size_coarse; j++)
                {
                    for (int i = 0; i < patch_size_coarse; i++)    // extract the current patch
                    {
                        if (patch_x_converse_coarse[t] + i - patch_size_coarse_half >= 0 && patch_x_converse_coarse[t] + i - patch_size_coarse_half < stack_orig->getNx() &&
                            patch_y_converse_coarse[t] + j - patch_size_coarse_half >= 0 && patch_y_converse_coarse[t] + j - patch_size_coarse_half < stack_orig->getNy())
                        {
                            patch_fix[dev][j * patch_size_coarse + i] = image_next[(patch_x_converse_coarse[t] + i - patch_size_coarse_half) +
                                                                        (patch_y_converse_coarse[t] + j - patch_size_coarse_half) * stack_orig->getNx()];
                            cc_de_fix[dev] += patch_fix[dev][j * patch_size_coarse + i] * patch_fix[dev][j * patch_size_coarse + i];
                        }
                    }
                }
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                CHECK(cudaSetDevice(dev));
                cudaMemcpyAsync(patch_g[dev], patch_fix[dev], sizeof(float) * patch_size_coarse * patch_size_coarse, cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev], patch_x_converse_coarse + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev] + 1, patch_y_converse_coarse + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaSetDevice(dev));
                compute_correlation <<< grid1, block1, 0, stream[dev] >>> (patch_g[dev], image_move_g[dev], cc_g[dev], cc_de_g[dev], para_xy[dev]);
                reduceUnrolling8 <<< grid2, block2, 0, stream[dev] >>> (cc_g[dev], cc_reduce_g[dev], cc_de_g[dev], cc_de_reduce_g[dev]);
                sum_reduced <<< grid3, block3, 0, stream[dev] >>> (cc_reduce_g[dev], cc_sum_g[dev], cc_de_reduce_g[dev], cc_de_sum_g[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                cudaMemcpyAsync(cc[dev], cc_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
                cudaMemcpyAsync(cc_de[dev], cc_de_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx_coarse * patch_Ny_coarse) break;
                cc_max[dev] = 0;
                cc_max_x[dev] = 0;
                cc_max_y[dev] = 0;
                for (int i = 0; i < patch_num_g; i++)
                {
                    cc[dev][i] = cc[dev][i] / sqrt(cc_de[dev][i] * cc_de_fix[dev]);
                    if (cc[dev][i] > cc_max[dev])
                    {
                        cc_max[dev] = cc[dev][i];
                        cc_max_x[dev] = i % (2 * patch_trans_coarse + 1) - patch_trans_coarse;
                        cc_max_y[dev] = int(i / (2 * patch_trans_coarse + 1)) - patch_trans_coarse;
                    }
                }
                //cout << n << " " << t << " " << cc_max[dev] << " " << cc_max_x[dev] << " " << cc_max_y[dev] << endl;
                if (abs(patch_deltaX_coarse[t] + cc_max_x[dev]) < 3 && abs(patch_deltaY_coarse[t] + cc_max_y[dev]) < 3)
                {
                    //cout << "accepted patch" << endl;
                }
                else
                {
                    patch_availN--;
                    patch_deltaX_coarse[t] = 0;
                    patch_deltaY_coarse[t] = 0;
                    //cout << "rejected patch" << endl;
                }
            }
            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaStreamSynchronize(stream[dev]));
            }
        }

        int patch_dx_avg = 0, patch_dy_avg = 0;
        // loop: patch_Nx_coarse*patch_Ny_coarse (number of patches)
        for (t = 0; t < patch_Nx_coarse * patch_Ny_coarse; t++)
        {
            patch_dx_avg += patch_deltaX_coarse[t];
            patch_dy_avg += patch_deltaY_coarse[t];
        }
        patch_dx_avg /= patch_availN;
        patch_dy_avg /= patch_availN;
        patch_dx_sum += patch_dx_avg; // 找到与0度图的平移量（一连串平移累加）
        patch_dy_sum += patch_dy_avg;
        patch_dx_sum_all[n - 1] = patch_dx_sum;
        patch_dy_sum_all[n - 1] = patch_dy_sum;

        cout << n << ":" << endl;
        cout << "adjacent: (" << patch_dx_avg << "," << patch_dy_avg << ")" << endl;
        cout << "all: (" << patch_dx_sum << "," << patch_dy_sum << ")" << endl;

        float image_avg = 0;
        for (int j = 0; j < stack_orig->getNy(); j++)
        {
            for (int i = 0; i < stack_orig->getNx(); i++)   // calculate average
            {
                image_avg += image_next[i + j * stack_orig->getNx()];
            }
        }
        image_avg = image_avg / (stack_orig->getNx() * stack_orig->getNy());
        // loop: Nx*Ny (whole image)
        for (int j = 0; j < stack_orig->getNy(); j++)
        {
            for (int i = 0; i < stack_orig->getNx(); i++)
            {
                if (i + patch_dx_sum >= 0 && i + patch_dx_sum < stack_orig->getNx() && j + patch_dy_sum >= 0 && j + patch_dy_sum < stack_orig->getNy())
                {
                    image_coarse[i + j * stack_orig->getNx()] = image_next[(i + patch_dx_sum) + (j + patch_dy_sum) * stack_orig->getNx()];
                }
                else
                {
                    image_coarse[i + j * stack_orig->getNx()] = image_avg;
                }
            }
        }
        stack_coarse->write2DIm(image_coarse, n - 1);
    }

    FILE* fcoarse = fopen(patch_save_path.c_str(), "w");  // 该文件中的平移量均为相对0度图的平移
    for (int n = 0; n < stack_orig->getNz(); n++)
    {
        fprintf(fcoarse, "%d %d\n", patch_dx_sum_all[n], patch_dy_sum_all[n]);
    }
    
    fflush(fcoarse);
    fclose(fcoarse);
    
    cout << "Finish coarse alignment!" << endl;
    cout << "Result saved in: " << patch_save_path << endl;

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaFree(patch_g[dev]);
        cudaFree(image_move_g[dev]);
        cudaFree(cc_g[dev]);
        cudaFree(cc_de_g[dev]);
        cudaFree(cc_reduce_g[dev]);
        cudaFree(cc_de_reduce_g[dev]);
        cudaFree(cc_sum_g[dev]);
        cudaFree(cc_de_sum_g[dev]);
        cudaFree(para_xy[dev]);
    }
    for (int dev = 0; dev < deviceCount; dev++)
    {
        CHECK(cudaStreamDestroy(stream[dev]));
    }
}



void patch_tracking(int ref_n, int patch_size, int patch_trans, int patch_Nx, int patch_Ny, MRC* stack_orig, string patch_save_path)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaStream_t stream[deviceCount];

    int patch_size_half = int(patch_size / 2);
    int patch_size_g = pow(2, int(log(patch_size) / log(2)) + 1);
    int patch_num_g = (2 * patch_trans + 1) * (2 * patch_trans + 1);
    int iteration = int(patch_Nx * patch_Ny / deviceCount);
    int para[3];
    para[0] = stack_orig->getNx();
    para[1] = stack_orig->getNy();
    para[2] = patch_size;

    float* patch_fix[deviceCount], *cc[deviceCount], *cc_de[deviceCount];
    float cc_de_fix[deviceCount], cc_max[deviceCount];
    int* para_xy[deviceCount];
    int cc_max_x[deviceCount], cc_max_y[deviceCount];

    for (int dev = 0; dev < deviceCount; dev++)
    {
        patch_fix[dev] = new float[patch_size * patch_size];
        cc[dev] = new float[patch_num_g];
        cc_de[dev] = new float[patch_num_g];
        para_xy[dev] = new int[2];
    }

    float* patch_g[deviceCount], *image_move_g[deviceCount];
    float* cc_g[deviceCount], *cc_de_g[deviceCount], *cc_reduce_g[deviceCount], *cc_de_reduce_g[deviceCount], *cc_sum_g[deviceCount], *cc_de_sum_g[deviceCount];

    for (int dev = 0; dev < deviceCount; dev++)
    {
        CHECK(cudaSetDevice(dev));
        CHECK(cudaStreamCreate(&(stream[dev])));

        cudaMalloc((float**)&patch_g[dev], sizeof(float) * patch_size * patch_size);
        cudaMalloc((float**)&image_move_g[dev], sizeof(float) * stack_orig->getNx() * stack_orig->getNy());

        cudaMalloc((float**)&cc_g[dev], sizeof(float) * patch_num_g * patch_size_g * patch_size_g);
        cudaMalloc((float**)&cc_de_g[dev], sizeof(float) * patch_num_g * patch_size_g * patch_size_g);
        cudaMalloc((float**)&cc_reduce_g[dev], sizeof(float) * patch_num_g * patch_size_g);
        cudaMalloc((float**)&cc_de_reduce_g[dev], sizeof(float) * patch_num_g * patch_size_g);
        cudaMalloc((float**)&cc_sum_g[dev], sizeof(float) * patch_num_g);
        cudaMalloc((float**)&cc_de_sum_g[dev], sizeof(float) * patch_num_g);

        cudaMalloc((int**)&para_xy[dev], sizeof(int) * 2);
        cudaMemcpyToSymbol(para_const, para, sizeof(int) * 3);
    }

    dim3 grid1(2 * patch_trans + 1, 2 * patch_trans + 1, patch_size_g);
    dim3 block1(patch_size_g);
    dim3 grid2(2 * patch_trans + 1, 2 * patch_trans + 1, patch_size_g / 8);
    dim3 block2(patch_size_g);
    dim3 grid3(2 * patch_trans + 1, 2 * patch_trans + 1);
    dim3 block3(patch_size_g / 8);

    int patch_deltaX[patch_Nx * patch_Ny];
    int patch_deltaY[patch_Nx * patch_Ny];

    int patch_x[stack_orig->getNz()][patch_Nx*patch_Ny];
    int patch_y[stack_orig->getNz()][patch_Nx*patch_Ny];
    bool patch_avail[stack_orig->getNz()][patch_Nx*patch_Ny];

    for (int i = 0; i < patch_Nx * patch_Ny; i++)
    {
        patch_avail[stack_orig->getNz() - 1][i] = 0;
    }

    patch_ref_init(patch_size, patch_Nx, patch_Ny, stack_orig->getNx(), stack_orig->getNy(), patch_x[ref_n], patch_y[ref_n]);

    float* image_now = new float[stack_orig->getNx() * stack_orig->getNy()];
    float* image_next = new float[stack_orig->getNx() * stack_orig->getNy()];
    float* image = new float[stack_orig->getNx() * stack_orig->getNy()];

    // refine from ref_n to Nz
    for (int n = ref_n; n < stack_orig->getNz() - 1; n++)
    {
        // standardize image
        if (n == ref_n)
        {
            stack_orig->read2DIm_32bit(image_now, n);
            stack_orig->read2DIm_32bit(image_next, n + 1);
        }
        else
        {
            memcpy(image_now, image_next, sizeof(float) * stack_orig->getNx() * stack_orig->getNy());
            stack_orig->read2DIm_32bit(image_next, n + 1);
        }

        int t;
        // forward searching, move: image_next, fix: image_now
        for (int dev = 0; dev < deviceCount; dev++)
        {
            cudaMemcpyAsync(image_move_g[dev], image_next, sizeof(float) * stack_orig->getNx() * stack_orig->getNy(), cudaMemcpyHostToDevice, stream[dev]);
        }

        for (int iter = 0; iter < iteration; iter++)
        {
            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                cc_de_fix[dev] = 0;
                for (int j = 0; j < patch_size; j++)    // extrack original patch
                {
                    for (int i = 0; i < patch_size; i++)   // can be be optimized?
                    {
                        if (patch_x[n][t] + i - patch_size_half >= 0 && patch_x[n][t] + i - patch_size_half < stack_orig->getNx() &&
                            patch_y[n][t] + j - patch_size_half >= 0 && patch_y[n][t] + j - patch_size_half < stack_orig->getNy())
                        {
                            patch_fix[dev][j * patch_size + i] = image_now[(patch_x[n][t] + i - patch_size_half) +
                                                                (patch_y[n][t] + j - patch_size_half) * stack_orig->getNx()];
                            cc_de_fix[dev] += patch_fix[dev][j * patch_size + i] * patch_fix[dev][j * patch_size + i];
                        }
                    }
                }

            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                CHECK(cudaSetDevice(dev));
                cudaMemcpyAsync(patch_g[dev], patch_fix[dev], sizeof(float) * patch_size * patch_size, cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev], patch_x[n] + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev] + 1, patch_y[n] + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaSetDevice(dev));
                compute_correlation << < grid1, block1, 0, stream[dev] >> > (patch_g[dev], image_move_g[dev], cc_g[dev], cc_de_g[dev], para_xy[dev]);
                reduceUnrolling8 << < grid2, block2, 0, stream[dev] >> > (cc_g[dev], cc_reduce_g[dev], cc_de_g[dev], cc_de_reduce_g[dev]);
                sum_reduced << < grid3, block3, 0, stream[dev] >> > (cc_reduce_g[dev], cc_sum_g[dev], cc_de_reduce_g[dev], cc_de_sum_g[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                cudaMemcpyAsync(cc[dev], cc_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
                cudaMemcpyAsync(cc_de[dev], cc_de_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                cc_max[dev] = 0;
                cc_max_x[dev] = 0;
                cc_max_y[dev] = 0;
                for (int i = 0; i < patch_num_g; i++)
                {
                    cc[dev][i] = cc[dev][i] / sqrt(cc_de[dev][i] * cc_de_fix[dev]);
                    if (cc[dev][i] > cc_max[dev])
                    {
                        cc_max[dev] = cc[dev][i];
                        cc_max_x[dev] = i % (2 * patch_trans + 1) - patch_trans;
                        cc_max_y[dev] = int(i / (2 * patch_trans + 1)) - patch_trans;
                    }
                }
                //cout << n << " " << t << " " << cc_max[dev] << " " << cc_max_x[dev] << " " << cc_max_y[dev] << endl;
                patch_deltaX[t] = cc_max_x[dev];
                patch_deltaY[t] = cc_max_y[dev];
                patch_x[n + 1][t] = patch_x[n][t] + cc_max_x[dev];
                patch_y[n + 1][t] = patch_y[n][t] + cc_max_y[dev];
            }
            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaStreamSynchronize(stream[dev]));
            }
        }


        // backward searching, move: image_now, fix: image_next
        for (int dev = 0; dev < deviceCount; dev++)
        {
            cudaMemcpyAsync(image_move_g[dev], image_now, sizeof(float) * stack_orig->getNx() * stack_orig->getNy(), cudaMemcpyHostToDevice, stream[dev]);
        }

        for (int iter = 0; iter < iteration; iter++)
        {
            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                cc_de_fix[dev] = 0;
                for (int j = 0; j < patch_size; j++)
                {
                    for (int i = 0; i < patch_size; i++)    // extract the current patch
                    {
                        if (patch_x[n + 1][t] + i - patch_size_half >= 0 && patch_x[n + 1][t] + i - patch_size_half < stack_orig->getNx() &&
                            patch_y[n + 1][t] + j - patch_size_half >= 0 && patch_y[n + 1][t] + j - patch_size_half < stack_orig->getNy())
                        {
                            patch_fix[dev][j * patch_size + i] = image_next[(patch_x[n + 1][t] + i - patch_size_half) +
                                (patch_y[n + 1][t] + j - patch_size_half) * stack_orig->getNx()];
                            cc_de_fix[dev] += patch_fix[dev][j * patch_size + i] * patch_fix[dev][j * patch_size + i];
                        }
                    }
                }
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                CHECK(cudaSetDevice(dev));
                cudaMemcpyAsync(patch_g[dev], patch_fix[dev], sizeof(float) * patch_size * patch_size, cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev], patch_x[n + 1] + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev] + 1, patch_y[n + 1] + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaSetDevice(dev));
                compute_correlation << < grid1, block1, 0, stream[dev] >> > (patch_g[dev], image_move_g[dev], cc_g[dev], cc_de_g[dev], para_xy[dev]);
                reduceUnrolling8 << < grid2, block2, 0, stream[dev] >> > (cc_g[dev], cc_reduce_g[dev], cc_de_g[dev], cc_de_reduce_g[dev]);
                sum_reduced << < grid3, block3, 0, stream[dev] >> > (cc_reduce_g[dev], cc_sum_g[dev], cc_de_reduce_g[dev], cc_de_sum_g[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                cudaMemcpyAsync(cc[dev], cc_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
                cudaMemcpyAsync(cc_de[dev], cc_de_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                cc_max[dev] = 0;
                cc_max_x[dev] = 0;
                cc_max_y[dev] = 0;
                for (int i = 0; i < patch_num_g; i++)
                {
                    cc[dev][i] = cc[dev][i] / sqrt(cc_de[dev][i] * cc_de_fix[dev]);
                    if (cc[dev][i] > cc_max[dev])
                    {
                        cc_max[dev] = cc[dev][i];
                        cc_max_x[dev] = i % (2 * patch_trans + 1) - patch_trans;
                        cc_max_y[dev] = int(i / (2 * patch_trans + 1)) - patch_trans;
                    }
                }
                //cout << n << " " << t << " " << cc_max[dev] << " " << cc_max_x[dev] << " " << cc_max_y[dev] << endl;
                if (abs(patch_deltaX[t] + cc_max_x[dev]) < 3 && abs(patch_deltaY[t] + cc_max_y[dev]) < 3)
                {
                    patch_avail[n][t] = true;
                    //cout << "accepted patch" << endl;
                }
                else
                {
                    patch_avail[n][t] = false;
                    //cout << "rejected patch" << endl;
                }
            }
            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaStreamSynchronize(stream[dev]));
            }
        }
    }

    //refine from ref_n to 0
    for (int n = ref_n; n > 0; n--)
    {
        // standardize image
        if (n == ref_n)
        {
            stack_orig->read2DIm_32bit(image_now, n);
            stack_orig->read2DIm_32bit(image_next, n - 1);
        }
        else
        {
            memcpy(image_now, image_next, sizeof(float) * stack_orig->getNx() * stack_orig->getNy());
            stack_orig->read2DIm_32bit(image_next, n - 1);
        }

        int t;
        // forward searching, move: image_next, fix: image_now
        for (int dev = 0; dev < deviceCount; dev++)
        {
            cudaMemcpyAsync(image_move_g[dev], image_next, sizeof(float) * stack_orig->getNx() * stack_orig->getNy(), cudaMemcpyHostToDevice, stream[dev]);
        }

        for (int iter = 0; iter < iteration; iter++)
        {
            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                cc_de_fix[dev] = 0;
                for (int j = 0; j < patch_size; j++)    // extrack original patch
                {
                    for (int i = 0; i < patch_size; i++)   // can be be optimized?
                    {
                        if (patch_x[n][t] + i - patch_size_half >= 0 && patch_x[n][t] + i - patch_size_half < stack_orig->getNx() &&
                            patch_y[n][t] + j - patch_size_half >= 0 && patch_y[n][t] + j - patch_size_half < stack_orig->getNy())
                        {
                            patch_fix[dev][j * patch_size + i] = image_now[(patch_x[n][t] + i - patch_size_half) +
                                                                 (patch_y[n][t] + j - patch_size_half) * stack_orig->getNx()];
                            cc_de_fix[dev] += patch_fix[dev][j * patch_size + i] * patch_fix[dev][j * patch_size + i];
                        }
                    }
                }
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                CHECK(cudaSetDevice(dev));
                cudaMemcpyAsync(patch_g[dev], patch_fix[dev], sizeof(float) * patch_size * patch_size, cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev], patch_x[n] + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev] + 1, patch_y[n] + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaSetDevice(dev));
                compute_correlation << < grid1, block1, 0, stream[dev] >> > (patch_g[dev], image_move_g[dev], cc_g[dev], cc_de_g[dev], para_xy[dev]);
                reduceUnrolling8 << < grid2, block2, 0, stream[dev] >> > (cc_g[dev], cc_reduce_g[dev], cc_de_g[dev], cc_de_reduce_g[dev]);
                sum_reduced << < grid3, block3, 0, stream[dev] >> > (cc_reduce_g[dev], cc_sum_g[dev], cc_de_reduce_g[dev], cc_de_sum_g[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                cudaMemcpyAsync(cc[dev], cc_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
                cudaMemcpyAsync(cc_de[dev], cc_de_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                cc_max[dev] = 0;
                cc_max_x[dev] = 0;
                cc_max_y[dev] = 0;
                for (int i = 0; i < patch_num_g; i++)
                {
                    cc[dev][i] = cc[dev][i] / sqrt(cc_de[dev][i] * cc_de_fix[dev]);
                    if (cc[dev][i] > cc_max[dev])
                    {
                        cc_max[dev] = cc[dev][i];
                        cc_max_x[dev] = i % (2 * patch_trans + 1) - patch_trans;
                        cc_max_y[dev] = int(i / (2 * patch_trans + 1)) - patch_trans;
                    }
                }
                //cout << n << " " << t << " " << cc_max[dev] << " " << cc_max_x[dev] << " " << cc_max_y[dev] << endl;
                patch_deltaX[t] = cc_max_x[dev];
                patch_deltaY[t] = cc_max_y[dev];
                patch_x[n - 1][t] = patch_x[n][t] + cc_max_x[dev];
                patch_y[n - 1][t] = patch_y[n][t] + cc_max_y[dev];
            }
            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaStreamSynchronize(stream[dev]));
            }
        }


        // backward searching, move: image_now, fix: image_next
        for (int dev = 0; dev < deviceCount; dev++)
        {
            cudaMemcpyAsync(image_move_g[dev], image_now, sizeof(float) * stack_orig->getNx() * stack_orig->getNy(), cudaMemcpyHostToDevice, stream[dev]);
        }

        for (int iter = 0; iter < iteration; iter++)
        {
            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                cc_de_fix[dev] = 0;
                for (int j = 0; j < patch_size; j++)
                {
                    for (int i = 0; i < patch_size; i++)    // extract the current patch
                    {
                        if (patch_x[n - 1][t] + i - patch_size_half >= 0 && patch_x[n - 1][t] + i - patch_size_half < stack_orig->getNx() &&
                            patch_y[n - 1][t] + j - patch_size_half >= 0 && patch_y[n - 1][t] + j - patch_size_half < stack_orig->getNy())
                        {
                            patch_fix[dev][j * patch_size + i] = image_next[(patch_x[n - 1][t] + i - patch_size_half) +
                                (patch_y[n - 1][t] + j - patch_size_half) * stack_orig->getNx()];
                            cc_de_fix[dev] += patch_fix[dev][j * patch_size + i] * patch_fix[dev][j * patch_size + i];
                        }
                    }
                }
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                CHECK(cudaSetDevice(dev));
                cudaMemcpyAsync(patch_g[dev], patch_fix[dev], sizeof(float) * patch_size * patch_size, cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev], patch_x[n - 1] + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
                cudaMemcpyAsync(para_xy[dev] + 1, patch_y[n - 1] + t, sizeof(int), cudaMemcpyHostToDevice, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaSetDevice(dev));
                compute_correlation << < grid1, block1, 0, stream[dev] >> > (patch_g[dev], image_move_g[dev], cc_g[dev], cc_de_g[dev], para_xy[dev]);
                reduceUnrolling8 << < grid2, block2, 0, stream[dev] >> > (cc_g[dev], cc_reduce_g[dev], cc_de_g[dev], cc_de_reduce_g[dev]);
                sum_reduced << < grid3, block3, 0, stream[dev] >> > (cc_reduce_g[dev], cc_sum_g[dev], cc_de_reduce_g[dev], cc_de_sum_g[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                cudaMemcpyAsync(cc[dev], cc_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
                cudaMemcpyAsync(cc_de[dev], cc_de_sum_g[dev], sizeof(float) * patch_num_g, cudaMemcpyDeviceToHost, stream[dev]);
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                t = iter * deviceCount + dev;
                if (t >= patch_Nx * patch_Ny) break;
                cc_max[dev] = 0;
                cc_max_x[dev] = 0;
                cc_max_y[dev] = 0;
                for (int i = 0; i < patch_num_g; i++)
                {
                    cc[dev][i] = cc[dev][i] / sqrt(cc_de[dev][i] * cc_de_fix[dev]);
                    if (cc[dev][i] > cc_max[dev])
                    {
                        cc_max[dev] = cc[dev][i];
                        cc_max_x[dev] = i % (2 * patch_trans + 1) - patch_trans;
                        cc_max_y[dev] = int(i / (2 * patch_trans + 1)) - patch_trans;
                    }
                }
                //cout << n << " " << t << " " << cc_max[dev] << " " << cc_max_x[dev] << " " << cc_max_y[dev] << endl;
                if (abs(patch_deltaX[t] + cc_max_x[dev]) < 3 && abs(patch_deltaY[t] + cc_max_y[dev]) < 3)
                {
                    patch_avail[n - 1][t] = true;
                    //cout << "accepted patch" << endl;
                }
                else
                {
                    patch_avail[n - 1][t] = false;
                    //cout << "rejected patch" << endl;
                }
            }
            for (int dev = 0; dev < deviceCount; dev++)
            {
                CHECK(cudaStreamSynchronize(stream[dev]));
            }
        }
    }

    // write out coarse translation
    FILE* fpatch = fopen(patch_save_path.c_str(), "w");
    // loop: Nz (number of images)
    for (int n = 0; n < stack_orig->getNz(); n++)   // patch_avail
    {
        // loop: patch_Nx*patch_Ny (number of patches)
        for (int t = 0; t < patch_Nx * patch_Ny; t++)
        {
            fprintf(fpatch, "%d ", patch_avail[n][t]);
        }
        fprintf(fpatch, "\n");
    }
    // loop: Nz (number of images)
    for (int n = 0; n < stack_orig->getNz(); n++)
    {
        // loop: patch_Nx*patch_Ny (number of patches)
        for (int t = 0; t < patch_Nx * patch_Ny; t++)
        {
            fprintf(fpatch, "%d ", patch_x[n][t]);
        }
        fprintf(fpatch, "\n");
    }
    // loop: Nz (number of images)
    for (int n = 0; n < stack_orig->getNz(); n++)
    {
        // loop: patch_Nx*patch_Ny (number of patches)
        for (int t = 0; t < patch_Nx * patch_Ny; t++)
        {
            fprintf(fpatch, "%d ", patch_y[n][t]);
        }
        fprintf(fpatch, "\n");
    }
    fflush(fpatch);
    fclose(fpatch);

    cout << "Finish patch tracking!" << endl;
    cout << "Result saved in: " << patch_save_path << endl;

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaFree(patch_g[dev]);
        cudaFree(image_move_g[dev]);
        cudaFree(cc_g[dev]);
        cudaFree(cc_de_g[dev]);
        cudaFree(cc_reduce_g[dev]);
        cudaFree(cc_de_reduce_g[dev]);
        cudaFree(cc_sum_g[dev]);
        cudaFree(cc_de_sum_g[dev]);
        cudaFree(para_xy[dev]);
    }
    for (int dev = 0; dev < deviceCount; dev++)
    {
        CHECK(cudaStreamDestroy(stream[dev]));
    }
}