#include "GPUInterface.h"

void sumArray(int nElem)
{
    cuTomo::sumArray(nElem);
}

//需要将gpu/src也加入include path才能找到namespace cuTomo
//可能是VSCode的问题。
int InitGPU(    
	std::vector<int>& iGPU,
    std::vector<void*>& stream)
{
	//调用GPUKernel.cu中的函数
	return cuTomo::cudaInit(iGPU, stream);
}

void freeGPU(
    std::vector<int>& iGPU,
    std::vector<void*>& stream)
{
	cuTomo::cudaEndUp(iGPU,stream);
}

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
    std::vector<void *>& stream
)
{
    cuTomo::volumeProj(
        skip_ctfcorrection,
        skip_3dctf,
        stack_orig.getNx(),
        stack_orig.getNy(),
        //ctf_para_n,     //"n" refers to the n-th image out of Nz overall
        stack_recon,
        proj_now,
        psi_rad,
        theta_rad,      
        h_tilt_max,
        h,
        // proj_omp,
        // bufc,
        x_offset,
        y_offset,
        z_offset,
        iGPU,
        stream
    );
    // if(!skip_ctfcorrection)
    // {
    //     cuTomo::ctfModulationG(
    //         stack_recon
    //     );
    // }
}
