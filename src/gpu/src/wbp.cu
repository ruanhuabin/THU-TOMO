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
#include "../../include/CTF.h"
using namespace std;

__constant__ int Nx_g[1];
__constant__ float ctf_const_g[4];
__constant__ float ctf_nconst_g[4];
__constant__ float recon_const_g[3];
__constant__ bool flip_const_g[1];

__global__ void cft_correction_gpu(cufftComplex* img_fft, cufftComplex* img_ctf, float* z_offset)
{
    int x_norm = blockIdx.x * blockDim.x + threadIdx.x;
    int y_norm = (blockIdx.y >= int(ceil(float(gridDim.y + 1) / 2))) ? (blockIdx.y - gridDim.y) : blockIdx.y;

    float x_real = float(x_norm) / float(Nx_g[0]) * (1 / ctf_const_g[0]);
    float y_real = float(y_norm) / float(gridDim.y) * (1 / ctf_const_g[0]);

    float alpha;
    alpha = (y_norm < 0) ? -M_PI_2 : ((y_norm == 0) ? 0 : M_PI_2);
    alpha = (x_norm == 0) ? alpha : atan(y_real / x_real);

    float freq2 = x_real * x_real + y_real * y_real;
    float df_now = ((ctf_nconst_g[0] + ctf_nconst_g[1] - 2 * z_offset[0] * ctf_const_g[0]) + (ctf_nconst_g[0] - ctf_nconst_g[1]) * cos(2 * (alpha - ctf_nconst_g[2]))) / 2.0;
    float chi = M_PI * ctf_const_g[3] * df_now * freq2 - M_PI_2 * ctf_const_g[1] * ctf_const_g[3] * ctf_const_g[3] * ctf_const_g[3] * freq2 * freq2 + ctf_nconst_g[3];
    float ctf_now = sqrt(1 - ctf_const_g[2] * ctf_const_g[2]) * sin(chi) + ctf_const_g[2] * cos(chi);

    ctf_now = flip_const_g[0] ? -ctf_now : ctf_now;
    ctf_now = (ctf_now > 0) ? 1.0 : -1.0;
    if (x_norm < int(Nx_g[0] / 2 + 1))
    {
        img_ctf[blockIdx.y * (Nx_g[0] / 2 + 1) + x_norm].x = img_fft[blockIdx.y * (Nx_g[0] / 2 + 1) + x_norm].x * ctf_now;
        img_ctf[blockIdx.y * (Nx_g[0] / 2 + 1) + x_norm].y = img_fft[blockIdx.y * (Nx_g[0] / 2 + 1) + x_norm].y * ctf_now;
    }
}

__global__ void image_normalization_gpu(float* img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < Nx_g[0]) img[blockIdx.y * Nx_g[0] + x] /= float(gridDim.y * Nx_g[0]);
}

__global__ void reconstruct_gpu(float* recon_stack, float* recon_prev, float* recon_now)
{
    float x_orig_offset = float(gridDim.x) / 2.0;
    float z_orig_offset = float(blockDim.x) / 2.0;
    float x_orig = (float(blockIdx.x) - x_orig_offset) * cos(recon_const_g[2]) - (float(threadIdx.x) - z_orig_offset) * sin(recon_const_g[2]) + x_orig_offset;
    float z_orig = (float(blockIdx.x) - x_orig_offset) * sin(recon_const_g[2]) + (float(threadIdx.x) - z_orig_offset) * cos(recon_const_g[2]) + z_orig_offset;
    float coeff = x_orig - floor(x_orig);
    int n_z = floor(((z_orig - z_orig_offset) + int(recon_const_g[0] / 2)) / recon_const_g[1]);

    if (floor(x_orig) >= 0 && ceil(x_orig) < gridDim.x && n_z >= 0 && n_z < int(recon_const_g[0] / recon_const_g[1]) + 1)
    {
        recon_now[blockIdx.x + threadIdx.x * gridDim.x + blockIdx.y * gridDim.x * blockDim.x] = recon_prev[blockIdx.x + threadIdx.x * gridDim.x + blockIdx.y * gridDim.x * blockDim.x] + 
        (1 - coeff) * recon_stack[n_z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + int(floor(x_orig))] + coeff * recon_stack[n_z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + int(ceil(x_orig))];
    }
    else recon_now[blockIdx.x + threadIdx.x * gridDim.x + blockIdx.y * gridDim.x * blockDim.x] = recon_prev[blockIdx.x + threadIdx.x * gridDim.x + blockIdx.y * gridDim.x * blockDim.x];
}

void weight_back_projection(MRC* stack_orig, MRC* stack_recon, std::string temp_save_path, int h_tilt_max, int defocus_step,
                            int h, float pix, float Cs, float volt, float w_cos, float psi_deg, int batch_recon, int batch_write,
                            bool flip_contrast, CTF *ctf_para, float *theta, bool ram)
{
    float ctf_const[8];
    float recon_const[3];
    recon_const[0] = h_tilt_max;
    recon_const[1] = defocus_step;
    
    ctf_const[0] = ctf_para[0].getPixelSize();
    ctf_const[1] = ctf_para[0].getCs();
    ctf_const[2] = ctf_para[0].getW();
    ctf_const[3] = ctf_para[0].getLambda();
    
    int y_recon = int(batch_recon / batch_write);

    int Nx = stack_orig->getNx();
    int Ny = stack_orig->getNy();
    int Nz = stack_orig->getNz();

    float* recon_empty_xyz = new float[Nx * h * y_recon];
    for (int i = 0; i < Nx * h * y_recon; i++) recon_empty_xyz[i] = 0.0;

    //======================================================================================================================================
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaStream_t stream[deviceCount];
    cudaStream_t stream_recon[deviceCount * batch_write];

    int N_zz = int(h_tilt_max / defocus_step) + 1;
    float* image[deviceCount];
    for (int dev = 0; dev < deviceCount; dev++) image[dev] = new float[Nx * Ny];
    
    MRC *stack_temp;
    float *recon_temp;
    float *recon_now[deviceCount * batch_write], *recon_prev[deviceCount * batch_write];

    float *image_recon;

    if(!ram)
    {
        stack_temp = new MRC[deviceCount * batch_write];
        for(int i = 0; i < deviceCount * batch_write; i++)
        {
            string output_mrc = temp_save_path;
            char* idx = new char;
            sprintf(idx, "%d", i);
            output_mrc.append(idx);
            stack_temp[i] = MRC(output_mrc.c_str(),"wb+");
            stack_temp[i].createMRC_empty(Nx, h, ceil(ceil(float(Ny) / float(batch_recon)) / float(deviceCount)) * batch_recon / batch_write, 2);
        }
        
        recon_temp = new float[Nx * h];
        for (int dev = 0; dev < deviceCount; dev++)
        {
            for (int i = 0; i < batch_write; i++)
            {
                recon_now[dev * batch_write + i] = new float[y_recon * Nx * h];
                recon_prev[dev * batch_write + i] = new float[y_recon * Nx * h];
            }
        }
    }
    else
    {
        image_recon = new float[size_t(Ny) * size_t(h) * size_t(Nx)];
    }

    cufftHandle p[deviceCount];
    cufftComplex* image_fft[Nz];

    cufftComplex* image_fft_g[deviceCount], *image_ctf_g[deviceCount];
    float* image_g[deviceCount], *z_offset_g[deviceCount];
    float* stack_corrected[int(h_tilt_max / defocus_step) + 1];
    for (int i = 0; i < N_zz; i++) stack_corrected[i] = new float[Nx * Ny];
    float* recon_stack_g[deviceCount * batch_write], *recon_now_g[deviceCount * batch_write], *recon_prev_g[deviceCount * batch_write];

    for (int dev = 0; dev < deviceCount; dev++)
    {
        CHECK(cudaSetDevice(dev));
        CHECK(cudaStreamCreate(&(stream[dev])));
        
        CHECK(cudaMalloc((float**)&image_g[dev], sizeof(float) * Nx * Ny));
        CHECK(cudaMalloc((float**)&z_offset_g[dev], sizeof(float)));

        CHECK(cudaMalloc((cufftComplex**)&image_fft_g[dev], sizeof(cufftComplex) * (Nx / 2 + 1) * Ny));
        CHECK(cudaMalloc((cufftComplex**)&image_ctf_g[dev], sizeof(cufftComplex) * (Nx / 2 + 1) * Ny));

        CHECK(cudaMemcpyToSymbol(ctf_const_g, ctf_const, sizeof(float) * 4));
        CHECK(cudaMemcpyToSymbol(flip_const_g, &flip_contrast, sizeof(bool)));
        CHECK(cudaMemcpyToSymbol(Nx_g, &Nx, sizeof(int)));

        cufftPlan2d(&p[dev], Ny, Nx, CUFFT_R2C);
        cufftSetStream(p[dev], stream[dev]);
        
        for(int i = 0; i < batch_write; i++)
        {
            CHECK(cudaStreamCreate(&(stream_recon[dev * batch_write + i])));
            CHECK(cudaMalloc((float**)&recon_stack_g[dev * batch_write + i], sizeof(float) * Nx * N_zz * y_recon));
            CHECK(cudaMalloc((float**)&recon_now_g[dev * batch_write + i], sizeof(float) * Nx * h * y_recon));
            CHECK(cudaMalloc((float**)&recon_prev_g[dev * batch_write + i], sizeof(float) * Nx * h * y_recon));
        }
    }

    for (int i = 0; i < Nz; i += deviceCount)
    {
        #pragma omp parallel num_threads(deviceCount)
        {
            int dev = omp_get_thread_num();
            int n = i + dev;
            if ((dev < deviceCount) && (n < Nz))
            {
                image_fft[n] = new cufftComplex[(Nx / 2 + 1) * Ny];
                stack_orig->read2DIm_32bit(image[dev], n);
                CHECK(cudaSetDevice(dev));
                CHECK(cudaMemcpyAsync(image_g[dev], image[dev], sizeof(float) * Nx * Ny, cudaMemcpyHostToDevice, stream[dev]));
                CHECK_CUFFT(cufftExecR2C(p[dev], image_g[dev], image_fft_g[dev]));
                CHECK(cudaMemcpyAsync(image_fft[n], image_fft_g[dev], sizeof(cufftComplex) * (Nx / 2 + 1) * Ny, cudaMemcpyDeviceToHost, stream[dev]));
            }
        }
    }
    for (int dev = 0; dev < deviceCount; dev++)
    {
        free(image[dev]);
        CHECK(cudaSetDevice(dev));
        CHECK(cudaStreamSynchronize(stream[dev]));
    }

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    dim3 grid1(int((Nx / 2 + 1) / deviceProp.maxThreadsPerBlock) +1, Ny);
    dim3 block1(deviceProp.maxThreadsPerBlock);

    dim3 grid2(int(Nx / deviceProp.maxThreadsPerBlock) +1, Ny);
    dim3 block2(deviceProp.maxThreadsPerBlock);

    dim3 grid3(Nx, y_recon);
    dim3 block3(h);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        CHECK(cudaSetDevice(dev));
        cufftDestroy(p[dev]);
        cufftPlan2d(&p[dev], Ny, Nx, CUFFT_C2R);
        cufftSetStream(p[dev], stream[dev]);
    }

    for (int n = 0; n < Nz; n++)
    {
        ctf_const[4] = ctf_para[n].getDefocus1();
        ctf_const[5] = ctf_para[n].getDefocus2();
        ctf_const[6] = ctf_para[n].getAstigmatism();
        ctf_const[7] = ctf_para[n].getPhaseShift();
        recon_const[2] = theta[n] / 180 * M_PI;

        for (int dev = 0; dev < deviceCount; dev++)
        {
            CHECK(cudaSetDevice(dev));
            CHECK(cudaMemcpyToSymbol(ctf_nconst_g, ctf_const + 4, 4 * sizeof(float)));
            CHECK(cudaMemcpyToSymbol(recon_const_g, recon_const, sizeof(float) * 3));
        }
        //for (int dev = 0; dev < deviceCount; dev++)
        #pragma omp parallel num_threads(deviceCount) 
        {
            int dev = omp_get_thread_num();
            CHECK(cudaSetDevice(dev));
            CHECK(cudaMemcpyAsync(image_fft_g[dev], image_fft[n], sizeof(cufftComplex) * (Nx / 2 + 1) * Ny, cudaMemcpyHostToDevice, stream[dev]));
            for(int i = 0; i < batch_write; i++) CHECK(cudaMemcpyAsync(recon_prev_g[dev * batch_write + i], recon_empty_xyz, sizeof(float) * y_recon * h * Nx, cudaMemcpyHostToDevice, stream[dev]));
        }

        for (int zz = -int(h_tilt_max / 2); zz < int(h_tilt_max / 2); zz += deviceCount * defocus_step)
        {
            //for (int dev = 0; dev < deviceCount; dev++)
            #pragma omp parallel num_threads(deviceCount) 
            {
                int dev = omp_get_thread_num();
                int zz_d = zz + dev * defocus_step;
                if ((dev < deviceCount) && (zz_d < int(h_tilt_max / 2)))
                {
                    CHECK(cudaSetDevice(dev));
                    float z_offset = float(zz_d) + float(defocus_step - 1) / 2;
                    int n_z = (zz_d + int(h_tilt_max / 2)) / defocus_step;
                    CHECK(cudaMemcpyAsync(z_offset_g[dev], &z_offset, sizeof(float), cudaMemcpyHostToDevice, stream[dev]));
                    cft_correction_gpu <<< grid1, block1, 0, stream[dev] >>> (image_fft_g[dev], image_ctf_g[dev], z_offset_g[dev]);
                    CHECK_CUFFT(cufftExecC2R(p[dev], image_ctf_g[dev], image_g[dev]));
                    image_normalization_gpu <<< grid2, block2, 0, stream[dev] >>> (image_g[dev]);
                    CHECK(cudaMemcpyAsync(stack_corrected[n_z], image_g[dev], sizeof(float) * Nx * Ny, cudaMemcpyDeviceToHost, stream[dev]));
                }
            }
        }
        for (int dev = 0; dev < deviceCount; dev++)
        {
            CHECK(cudaSetDevice(dev));
            CHECK(cudaStreamSynchronize(stream[dev]));
        }

        for (int batch = 0; batch < ceil(float(Ny) / float(batch_recon)); batch += deviceCount)
        {
            if(!ram)
            {
                #pragma omp parallel num_threads(deviceCount * batch_write) 
                {
                    int th = omp_get_thread_num();
                    int dev = int(th / batch_write);
                    int i = th % batch_write;
                    int j = batch + dev;
                    int j_file = int(float(batch) / float(deviceCount));
                    if (j < ceil(float(Ny) / float(batch_recon)))
                    {
                        CHECK(cudaSetDevice(dev));
                        int batch_current = (batch_recon < (Ny - j * batch_recon))? batch_recon : (Ny - j * batch_recon);
                        int y_current = (y_recon < (batch_current - i * y_recon))? y_recon : (batch_current - i * y_recon);
                        if(y_current > 0)
                        {       
                            for (int k = 0; k < N_zz; k++)
                            {
                                CHECK(cudaMemcpyAsync(recon_stack_g[dev * batch_write + i] + k * y_recon * Nx, stack_corrected[k] + Nx * (j * batch_recon + i * y_recon), sizeof(float) * Nx * y_current, cudaMemcpyHostToDevice, stream_recon[dev * batch_write + i]));
                            }
                            if (n > 0)
                            {
                                    
                                for(int r = 0; r < y_current; r++)
                                {
                                    stack_temp[dev * batch_write + i].read2DIm_32bit(recon_prev[dev * batch_write + i] + r * h * Nx, j_file * y_recon + r);
                                }
                                CHECK(cudaMemcpyAsync(recon_prev_g[dev * batch_write + i], recon_prev[dev * batch_write + i], sizeof(float) * y_current * h * Nx, cudaMemcpyHostToDevice, stream_recon[dev * batch_write+ i]));
                            }
                            reconstruct_gpu <<< grid3, block3, 0, stream_recon[dev * batch_write + i] >>> (recon_stack_g[dev * batch_write + i], recon_prev_g[dev * batch_write + i], recon_now_g[dev * batch_write + i]);
                            CHECK(cudaMemcpyAsync(recon_now[dev * batch_write + i], recon_now_g[dev * batch_write + i], sizeof(float) * y_current * h * Nx, cudaMemcpyDeviceToHost, stream_recon[dev * batch_write + i]));
                            CHECK(cudaStreamSynchronize(stream_recon[dev * batch_write + i]));
                            for(int w = 0; w < y_current; w++)
                            {
                                stack_temp[dev * batch_write + i].write2DIm(recon_now[dev * batch_write + i] + w * h * Nx, j_file * y_recon + w);
                            }
                        }
                    }
                }
            }
            else
            {
                #pragma omp parallel num_threads(deviceCount * batch_write) 
                {
                    int th = omp_get_thread_num();
                    int dev = int(th / batch_write);
                    int i = th % batch_write;
                    int j = batch + dev;
                    if ((dev < deviceCount) && (j < ceil(float(Ny) / float(batch_recon))))
                    {
                        CHECK(cudaSetDevice(dev));
                        int batch_current = (batch_recon < (Ny - j * batch_recon))? batch_recon : (Ny - j * batch_recon);
                        int y_current = (y_recon < (batch_current - i * y_recon))? y_recon : (batch_current - i * y_recon);
                        if(y_current > 0)
                        {                        
                            for (int k = 0; k < N_zz; k++)
                            {
                                CHECK(cudaMemcpyAsync(recon_stack_g[dev * batch_write + i] + k * y_recon * Nx, stack_corrected[k] + Nx * (j * batch_recon + i * y_recon), sizeof(float) * Nx * y_current, cudaMemcpyHostToDevice, stream_recon[dev * batch_write + i]));
                            }
                            if (n > 0)
                            {
                                CHECK(cudaMemcpyAsync(recon_prev_g[dev * batch_write + i], image_recon + size_t(h) * size_t(Nx) * size_t(j * batch_recon + i * y_recon), sizeof(float) * y_current * h * Nx, cudaMemcpyHostToDevice, stream_recon[dev * batch_write+ i]));
                            }
                            reconstruct_gpu <<< grid3, block3, 0, stream_recon[dev * batch_write + i] >>> (recon_stack_g[dev * batch_write + i], recon_prev_g[dev * batch_write + i], recon_now_g[dev * batch_write + i]);
                            CHECK(cudaMemcpyAsync(image_recon + size_t(h) * size_t(Nx) * size_t(j * batch_recon + i * y_recon), recon_now_g[dev * batch_write + i], sizeof(float) * y_current * h * Nx, cudaMemcpyDeviceToHost, stream_recon[dev * batch_write + i]));
                        }
                    }
                }
            }
        }

        // for (int batch = 0; batch < ceil(float(Ny) / float(batch_recon)); batch += deviceCount)
        // {
        //     if(!ram)
        //     {
        //         #pragma omp parallel num_threads(deviceCount) 
        //         {
        //             int dev = omp_get_thread_num();
        //             int j = batch + dev;
        //             int j_file = int(float(batch) / float(deviceCount));
        //             if (j < ceil(float(Ny) / float(batch_recon)))
        //             {
        //                 CHECK(cudaSetDevice(dev));
        //                 int batch_current = (batch_recon < (Ny - j * batch_recon))? batch_recon : (Ny - j * batch_recon);
        //                 for(int i = 0; i < batch_write; i++)
        //                 //#pragma omp parallel num_threads(batch_write) 
        //                 {
        //                     //int i = omp_get_thread_num();
        //                     int y_current = (y_recon < (batch_current - i * y_recon))? y_recon : (batch_current - i * y_recon);
        //                     if(y_current > 0)
        //                     {       
        //                         for (int k = 0; k < N_zz; k++)
        //                         {
        //                             CHECK(cudaMemcpyAsync(recon_stack_g[dev * batch_write + i] + k * y_recon * Nx, stack_corrected[k] + Nx * (j * batch_recon + i * y_recon), sizeof(float) * Nx * y_current, cudaMemcpyHostToDevice, stream_recon[dev * batch_write + i]));
        //                         }
        //                         if (n > 0)
        //                         {
                                    
        //                             for(int r = 0; r < y_current; r++)
        //                             {
        //                                 stack_temp[dev * batch_write + i].read2DIm_32bit(recon_prev[dev * batch_write + i] + r * h * Nx, j_file * y_recon + r);
        //                             }
        //                             CHECK(cudaMemcpyAsync(recon_prev_g[dev * batch_write + i], recon_prev[dev * batch_write + i], sizeof(float) * y_current * h * Nx, cudaMemcpyHostToDevice, stream_recon[dev * batch_write+ i]));
        //                         }
        //                         reconstruct_gpu <<< grid3, block3, 0, stream_recon[dev * batch_write + i] >>> (recon_stack_g[dev * batch_write + i], recon_prev_g[dev * batch_write + i], recon_now_g[dev * batch_write + i]);
        //                         CHECK(cudaMemcpyAsync(recon_now[dev * batch_write + i], recon_now_g[dev * batch_write + i], sizeof(float) * y_current * h * Nx, cudaMemcpyDeviceToHost, stream_recon[dev * batch_write + i]));
        //                         CHECK(cudaStreamSynchronize(stream_recon[dev * batch_write + i]));
        //                         for(int w = 0; w < y_current; w++)
        //                         {
        //                             stack_temp[dev * batch_write + i].write2DIm(recon_now[dev * batch_write + i] + w * h * Nx, j_file * y_recon + w);
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //     }
        //     else
        //     {
        //         #pragma omp parallel num_threads(deviceCount) 
        //         {
        //             int dev = omp_get_thread_num();
        //             int j = batch + dev;
        //             if ((dev < deviceCount) && (j < ceil(float(Ny) / float(batch_recon))))
        //             {
        //                 CHECK(cudaSetDevice(dev));
        //                 int batch_current = (batch_recon < (Ny - j * batch_recon))? batch_recon : (Ny - j * batch_recon);
        //                 for(int i=0; i < batch_write; i++)
        //                 //#pragma omp parallel num_threads(batch_write) 
        //                 {
        //                     //int i = omp_get_thread_num();
        //                     int y_current = (y_recon < (batch_current - i * y_recon))? y_recon : (batch_current - i * y_recon);
        //                     if(y_current > 0)
        //                     {                        
        //                         for (int k = 0; k < N_zz; k++)
        //                         {
        //                             CHECK(cudaMemcpyAsync(recon_stack_g[dev * batch_write + i] + k * y_recon * Nx, stack_corrected[k] + Nx * (j * batch_recon + i * y_recon), sizeof(float) * Nx * y_current, cudaMemcpyHostToDevice, stream_recon[dev * batch_write + i]));
        //                         }
        //                         if (n > 0)
        //                         {
        //                             CHECK(cudaMemcpyAsync(recon_prev_g[dev * batch_write + i], image_recon + size_t(h) * size_t(Nx) * size_t(j * batch_recon + i * y_recon), sizeof(float) * y_current * h * Nx, cudaMemcpyHostToDevice, stream_recon[dev * batch_write+ i]));
        //                         }
        //                         reconstruct_gpu <<< grid3, block3, 0, stream_recon[dev * batch_write + i] >>> (recon_stack_g[dev * batch_write + i], recon_prev_g[dev * batch_write + i], recon_now_g[dev * batch_write + i]);
        //                         CHECK(cudaMemcpyAsync(image_recon + size_t(h) * size_t(Nx) * size_t(j * batch_recon + i * y_recon), recon_now_g[dev * batch_write + i], sizeof(float) * y_current * h * Nx, cudaMemcpyDeviceToHost, stream_recon[dev * batch_write + i]));
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
        cout << "n = " << n << " finished." << endl;
    }

    if(!ram)
    {
        for (int j = 0; j < ceil(float(Ny) / float(batch_recon)); j++)
        {
            int dev = j % deviceCount;
            int j_file = int((j - dev) / deviceCount);
            int batch_current = min(batch_recon, Ny - j * batch_recon);
            for (int w = 0; w < batch_write; w++)
            {
                int y_current = (y_recon < (batch_current - w * y_recon))? y_recon : (batch_current - w * y_recon);
                for (int y_file = 0; y_file < y_current; y_file++)
                {
                    stack_temp[dev * batch_write + w].read2DIm_32bit(recon_temp, j_file * y_recon + y_file);
                    stack_recon->write2DIm(recon_temp, y_file + w * y_recon + j * batch_recon);
                }
                
            }
        }
        stack_recon->computeHeader_omp(pix, false, 12);
        stack_recon->close();

        for(int i = 0; i < deviceCount * batch_write; i++)
        {
            stack_temp[i].close();
            string output_mrc = temp_save_path;
            char* idx = new char;
            sprintf(idx, "%d", i);
            output_mrc.append(idx);
            remove(output_mrc.c_str());
        }
    }
    else
    {
        for (int j = 0; j < Ny; j++)
        {
            stack_recon->write2DIm(image_recon + size_t(j) * size_t(Nx) * size_t(h), j);
        }
        stack_recon->computeHeader_omp(pix, false, 12);
        stack_recon->close();
    }
    
    for (int dev = 0; dev < deviceCount; dev++)
    {
        CHECK(cudaSetDevice(dev));
        cufftDestroy(p[dev]);
        CHECK(cudaStreamDestroy(stream[dev]));
        for(int i = 0; i < batch_write; i++) CHECK(cudaStreamDestroy(stream_recon[dev * batch_write + i]));
        cudaDeviceReset();
    }
    cout << "Done!" << endl;
}
