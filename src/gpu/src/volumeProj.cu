#include "volumeProj.cuh"

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
)
{

    //注：这里调用的是sinfgle precision的CUDA Math API —— floorf,ceilf,sinf,cosf。
    //不能直接使用CPU版本的function API。
    //如果后续需要其他精度，则需要对这些API的调用进行修改。

    for(int blkid = blockIdx.x; blkid < h_tilt_max; blkid += gridDim.x)
    {
        for(int tidX = threadIdx.x; tidX < stack_orig_Nx; tidX += blockDim.x)
        {
            for(int tidY = threadIdx.y; tidY < stack_orig_Ny; tidY += blockDim.y)
            {
                float x_now = tidX - x_offset;
                float y_now = tidY - y_offset;
                float z_now = blkid - z_offset;   // move origin to the center

                float x_psi=x_now*cosf(-psi_rad)-y_now*sinf(-psi_rad);
                float y_psi=x_now*sinf(-psi_rad)+y_now*cosf(-psi_rad);
                float z_psi=z_now;  // rotate tilt axis to y-axis

                float x_tlt=x_psi*cosf(-theta_rad)-z_psi*sinf(-theta_rad);
                float y_tlt=y_psi;
                float z_tlt=x_psi*sinf(-theta_rad)+z_psi*cosf(-theta_rad);  // tilt

                float x_final=x_tlt*cosf(psi_rad)-y_tlt*sinf(psi_rad)+x_offset;
                float y_final=x_tlt*sinf(psi_rad)+y_tlt*cosf(psi_rad)+y_offset;
                float z_final=z_tlt+z_offset-int(float(h_tilt_max-h)/2.0);    // rotate back

                float coeff_x=x_final-floorf(x_final);
                float coeff_y=y_final-floorf(y_final);
                float coeff_z=z_final-floorf(z_final);

                float tmp_y_1_1,tmp_y_1_2,tmp_y_2_1,tmp_y_2_2;
                float tmp_z_1,tmp_z_2;

                //TODO:把devStackRecon改成二维。
                
                if(floorf(x_final)>=0 && ceilf(x_final)<stack_orig_Nx && floorf(y_final)>=0 && ceilf(y_final)<stack_orig_Ny && floorf(z_final)>=0 && ceilf(z_final)<h)
                {
                    //slice_now[omp_get_thread_num()][j*stack_orig.getNx()+i]=(1-coeff_x)*(1-coeff_y)*(1-coeff_z)*stack_recon[int(floorf(z_final))][int(floorf(y_final))*stack_orig.getNx()+int(floorf(x_final))]+(1-coeff_x)*(1-coeff_y)*(coeff_z)*stack_recon[int(ceilf(z_final))][int(floorf(y_final))*stack_orig.getNx()+int(floorf(x_final))]+(1-coeff_x)*(coeff_y)*(1-coeff_z)*stack_recon[int(floorf(z_final))][int(ceilf(y_final))*stack_orig.getNx()+int(floorf(x_final))]+(1-coeff_x)*(coeff_y)*(coeff_z)*stack_recon[int(ceilf(z_final))][int(ceilf(y_final))*stack_orig.getNx()+int(floorf(x_final))]+(coeff_x)*(1-coeff_y)*(1-coeff_z)*stack_recon[int(floorf(z_final))][int(floorf(y_final))*stack_orig.getNx()+int(ceilf(x_final))]+(coeff_x)*(1-coeff_y)*(coeff_z)*stack_recon[int(ceilf(z_final))][int(floorf(y_final))*stack_orig.getNx()+int(ceilf(x_final))]+(coeff_x)*(coeff_y)*(1-coeff_z)*stack_recon[int(floorf(z_final))][int(ceilf(y_final))*stack_orig.getNx()+int(ceilf(x_final))]+(coeff_x)*(coeff_y)*(coeff_z)*stack_recon[int(ceilf(z_final))][int(ceilf(y_final))*stack_orig.getNx()+int(ceilf(x_final))];
                    tmp_y_1_1=
                        (1-coeff_x)
                        // *stack_recon[int(floorf(z_final))]
                        //             [int(floorf(y_final))*stack_orig.getNx()+int(floorf(x_final))]
                        *devStackRecon[int(floorf(z_final)) * stack_orig_Nx * stack_orig_Ny
                                    + int(floorf(y_final)) * stack_orig_Nx + int(floorf(x_final))]
                        +(coeff_x)
                        // *stack_recon[int(floorf(z_final))]
                        //             [int(floorf(y_final))*stack_orig.getNx()+int(ceilf(x_final))];
                        *devStackRecon[int(floorf(z_final)) * stack_orig_Nx * stack_orig_Ny
                                    +int(floorf(y_final)) * stack_orig_Nx + int(ceilf(x_final))];
                    
                    tmp_y_1_2=
                        (1-coeff_x)
                        // *stack_recon[int(floorf(z_final))]
                        //             [int(ceilf(y_final))*stack_orig.getNx()+int(floorf(x_final))]
                        *devStackRecon[int(floorf(z_final)) * stack_orig_Nx * stack_orig_Ny
                                    + int(ceilf(y_final)) * stack_orig_Nx + int(floorf(x_final))]
                        +(coeff_x)
                        // *stack_recon[int(floorf(z_final))]
                        //             [int(ceilf(y_final))*stack_orig.getNx()+int(ceilf(x_final))];
                        *devStackRecon[int(floorf(z_final)) * stack_orig_Nx * stack_orig_Ny
                                    + int(ceilf(y_final)) * stack_orig_Nx + int(ceilf(x_final))];

                    tmp_y_2_1=
                        (1-coeff_x)
                        // *stack_recon[int(ceilf(z_final))]
                        //             [int(floorf(y_final))*stack_orig.getNx()+int(floorf(x_final))]
                        *devStackRecon[int(ceilf(z_final)) * stack_orig_Nx * stack_orig_Ny
                                    + int(floorf(y_final)) * stack_orig_Nx + int(floorf(x_final))]
                        +(coeff_x)
                        // *stack_recon[int(ceilf(z_final))]
                        //             [int(floorf(y_final))*stack_orig.getNx()+int(ceilf(x_final))];
                        *devStackRecon[int(ceilf(z_final)) * stack_orig_Nx * stack_orig_Ny
                                    + int(floorf(y_final)) * stack_orig_Nx + int(ceilf(x_final))]; 
                                           
                    tmp_y_2_2=
                        (1-coeff_x)
                        // *stack_recon[int(ceilf(z_final))]
                        //             [int(ceilf(y_final))*stack_orig.getNx()+int(floorf(x_final))]
                        *devStackRecon[int(ceilf(z_final)) * stack_orig_Nx * stack_orig_Ny
                                    + int(ceilf(y_final)) * stack_orig_Nx + int(floorf(x_final))]
                        +(coeff_x)
                        // *stack_recon[int(ceilf(z_final))]
                        //             [int(ceilf(y_final))*stack_orig.getNx()+int(ceilf(x_final))];
                        *devStackRecon[int(ceilf(z_final)) * stack_orig_Nx * stack_orig_Ny
                                    + int(ceilf(y_final)) * stack_orig_Nx + int(ceilf(x_final))];


                    tmp_z_1=(1-coeff_y)*tmp_y_1_1+(coeff_y)*tmp_y_1_2;
                    tmp_z_2=(1-coeff_y)*tmp_y_2_1+(coeff_y)*tmp_y_2_2;

                    //slice_now[omp_get_thread_num()][j*stack_orig.getNx()+i]=(1-coeff_z)*tmp_z_1+(coeff_z)*tmp_z_2;
                    //注意 这里需要取余,然后变成累加,与CPU版本的算法有所不同.
                    devProj[(blkid%gridDim.x) * stack_orig_Nx * stack_orig_Ny
                    + tidY * stack_orig_Nx + tidX ] += (1-coeff_z)*tmp_z_1+(coeff_z)*tmp_z_2;
                }
                else
                {
                    // slice_now[omp_get_thread_num()][j*stack_orig.getNx()+i]=0.0;
                    devProj[(blkid%gridDim.x) * stack_orig_Nx * stack_orig_Ny
                    + tidY * stack_orig_Nx + tidX ] += 0.0;
                }
            }
        }
    }   
}

//ToOpti:用更快的规约算法
//TODO:多线程同时写是否安全？
__global__ void kernel_projAccu(
    float *devProj,
    float *tmpProjSum,
    int stack_orig_Nx,
    int stack_orig_Ny,
    int h_tilt_max  //To Remove 这个参数是没用的
)
{
    for(int tidX = threadIdx.x; tidX < stack_orig_Nx; tidX += blockDim.x)
    {
        for(int tidY = threadIdx.y; tidY < stack_orig_Ny; tidY += blockDim.y)
        {
            tmpProjSum[tidY * blockDim.x + tidX] += devProj[blockIdx.x * stack_orig_Nx * stack_orig_Ny + tidY * blockDim.x + tidX];
        }
    }
}


}