#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "volumeProj.cuh"

#include <iostream>
#include <stdio.h>
#include <vector>

#include "common.h"



namespace cuTomo {

using std::vector;

int cudaInit(
    vector<int>& iGPU,
    vector<void*>& stream
);

void cudaEndUp(
    vector<int>& iGPU,
    vector<void*>& stream
);

void volumeProj(
    bool skip_ctfcorrection,
    bool skip_3dctf,
    //MRC stack_orig,
    int stack_orig_Nx,
    int stack_orig_Ny,
    //CTF ctf_para_n,     //"n" refers to the n-th image out of Nz overall
    float *stack_recon[],
    float *proj_now,
    float psi_rad,
    float theta_rad,      
    int h_tilt_max,
    int h,
    //int threads,         //To be removed
    //fftwf_plan *plan_fft,
    //fftwf_plan *plan_ifft,
    // float *proj_omp[],
    // float *bufc[],
    float x_offset,
    float y_offset,
    float z_offset,
    vector<int>& iGPU,
    vector<void*>& stream
);



}

