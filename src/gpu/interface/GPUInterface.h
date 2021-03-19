#include "TomoConfig.h"

#ifdef GPU_VERSION

#include <vector>
#include "mrc.h"
#include "CTF.h"
#include "common.h"
#include "GPUKernel.cuh"
#include "sumArray.h"

// #define GPU_VERSION

void sumArray(int nElem);

int InitGPU(    
    std::vector<int>& iGPU,
    std::vector<void*>& stream);

void freeGPU(
    std::vector<int>& iGPU,
    std::vector<void*>& stream);

void volumeProjG(
    bool skip_ctfcorrection,
    bool skip_3dctf,
    MRC& stack_orig,
    CTF& ctf_para_n,     //"n" refers to the n-th image out of Nz overall
    float *stack_recon[],
    float *proj_now,
    float psi_rad,
    float theta_rad,      
    int h_tilt_max,
    int h,
    //int threads,         //To be removed
    //fftwf_plan *plan_fft,
    //fftwf_plan *plan_ifft,
    float *proj_omp[],
    float *bufc[],
    float x_offset,
    float y_offset,
    float z_offset,
    std::vector<int>& iGPU,
    std::vector<void*>& stream
);

#endif
