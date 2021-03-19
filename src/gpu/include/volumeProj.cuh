#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
//#include <stdio.h>

namespace cuTomo
{
__global__ void	kernel_volumeProj(
    float *devStackRecon,   //[in]
    float *devProj,
    int stack_orig_Nx,
    int stack_orig_Ny,
    float psi_rad,
    float theta_rad,      
    int h_tilt_max,
    int h,
    // float *proj_omp[],
    // float *bufc[],
    float x_offset,
    float y_offset,
    float z_offset
);

__global__ void kernel_projAccu(
    float *devProj,
    float *tmpProjSum,
    int stack_orig_Nx,
    int stack_orig_Ny,
    int h_tilt_max
);


}