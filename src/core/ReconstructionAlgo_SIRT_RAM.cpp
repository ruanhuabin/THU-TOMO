/*******************************************************************
 *       Filename:  ReconstructionAlgo.cpp                                     
 *                                                                 
 *    Description:                                        
 *                                                                 
 *        Version:  1.0                                            
 *        Created:  07/07/2020 05:40:48 PM                                 
 *       Revision:  none                                           
 *       Compiler:  gcc                                           
 *                                                                 
 *         Author:  Ruan Huabin                                      
 *          Email:  ruanhuabin@tsinghua.edu.cn                                        
 *        Company:  Dep. of CS, Tsinghua Unversity                                      
 *                                                                 
 *******************************************************************/
#include "ReconstructionAlgo_SIRT_RAM.h"
#include "mrc.h"
#include "CTF.h"
#include "math.h"
#include "fftw3.h"
#include "omp.h"
#include "util.h"

static void buf2fft(float *buf, float *fft, int nx, int ny)
{
    int nxb=nx+2-nx%2;
    int i;
    for(i=0;i<(nx+2-nx%2)*ny;i++)
    {
        fft[i]=0.0;
    }
    for(i=0;i<ny;i++)
    {
        memcpy(fft+i*nxb,buf+i*nx,sizeof(float)*nx);
    }
}

static void fft2buf(float *buf, float *fft, int nx, int ny)
{
    int nxb=nx+2-nx%2;
    int i;
    for(i=0;i<nx*ny;i++)
    {
        buf[i]=0.0;
    }
    for(i=0;i<ny;i++)
    {
        memcpy(buf+i*nx,fft+i*nxb,sizeof(float)*nx);
    }
}

static void rotate_image(float *image_orig,float *image_rot,int Nx,int Ny,int Nx_orig_offset,int Ny_orig_offset,float psi_rad)   // counter-clockwise
{
    int i,j;
    float x_rot,y_rot;
    float x,y;
    float coeff_x,coeff_y;
    float tmp_y_1,tmp_y_2;
    // loop: Nx*Ny (whole image)
    for(j=0;j<Ny;j++)
    {
        for(i=0;i<Nx;i++)
        {
            x=i+Nx_orig_offset;
            y=j+Ny_orig_offset;
            x_rot=x*cos(-psi_rad)-y*sin(-psi_rad)-Nx_orig_offset;
            y_rot=x*sin(-psi_rad)+y*cos(-psi_rad)-Ny_orig_offset;
            if(floor(x_rot)>=0 && ceil(x_rot)<Nx && floor(y_rot)>=0 && ceil(y_rot)<Ny)
            {
                coeff_x=x_rot-floor(x_rot);
                coeff_y=y_rot-floor(y_rot);
                //image_rot[i+j*Nx]=(1-coeff_x)*(1-coeff_y)*image_orig[int(floor(x_rot)+floor(y_rot)*Nx)]+(coeff_x)*(1-coeff_y)*image_orig[int(ceil(x_rot)+floor(y_rot)*Nx)]+(1-coeff_x)*(coeff_y)*image_orig[int(floor(x_rot)+ceil(y_rot)*Nx)]+(coeff_x)*(coeff_y)*image_orig[int(ceil(x_rot)+ceil(y_rot)*Nx)];
                tmp_y_1=(1-coeff_x)*image_orig[int(floor(x_rot)+floor(y_rot)*Nx)]+(coeff_x)*image_orig[int(ceil(x_rot)+floor(y_rot)*Nx)];
                tmp_y_2=(1-coeff_x)*image_orig[int(floor(x_rot)+ceil(y_rot)*Nx)]+(coeff_x)*image_orig[int(ceil(x_rot)+ceil(y_rot)*Nx)];
                image_rot[i+j*Nx]=(1-coeff_y)*tmp_y_1+(coeff_y)*tmp_y_2;
            }
            else
            {
                image_rot[i+j*Nx]=0;
            }
        }
    }
}

static void ctf_modulation(float *image,int Nx,int Ny,CTF ctf,float z_offset)  // z_offset in pixels
{
    fftwf_plan plan_fft,plan_ifft;
    float *bufc=new float[(Nx+2-Nx%2)*Ny];
    plan_fft=fftwf_plan_dft_r2c_2d(Ny,Nx,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);
    plan_ifft=fftwf_plan_dft_c2r_2d(Ny,Nx,reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);
    buf2fft(image,bufc,Nx,Ny);
    fftwf_execute(plan_fft);

    // loop: Nx*Ny (whole image)
    for(int j=0;j<Ny;j++)
    {
        for(int i=0;i<(Nx+2-Nx%2);i+=2)
        {
            float ctf_now=ctf.computeCTF2D(i/2,j,Nx,Ny,false,false,z_offset);
            bufc[i+j*(Nx+2-Nx%2)]*=ctf_now;
            bufc[(i+1)+j*(Nx+2-Nx%2)]*=ctf_now;
        }
    }

    fftwf_execute(plan_ifft);
    fft2buf(image,bufc,Nx,Ny);

    // loop: Nx*Ny (whole image)
    for(int i=0;i<Nx*Ny;i++)   // normalization
    {
        image[i]=image[i]/(Nx*Ny);
    }
    fftwf_destroy_plan(plan_fft);
	fftwf_destroy_plan(plan_ifft);
    delete [] bufc;
}

static void ctf_modulation_omp(float *image,int Nx,int Ny,CTF ctf,float z_offset,fftwf_plan plan_fft,fftwf_plan plan_ifft,float *bufc)  // z_offset in pixels
{
    buf2fft(image,bufc,Nx,Ny);
    fftwf_execute(plan_fft);

    // loop: Nx*Ny (whole image)
    for(int j=0;j<Ny;j++)
    {
        for(int i=0;i<(Nx+2-Nx%2);i+=2)
        {
            float ctf_now=ctf.computeCTF2D(i/2,j,Nx,Ny,false,false,z_offset);
            bufc[i+j*(Nx+2-Nx%2)]*=ctf_now;
            bufc[(i+1)+j*(Nx+2-Nx%2)]*=ctf_now;
        }
    }

    fftwf_execute(plan_ifft);
    fft2buf(image,bufc,Nx,Ny);

    // loop: Nx*Ny (whole image)
    for(int i=0;i<Nx*Ny;i++)   // normalization
    {
        image[i]=image[i]/(Nx*Ny);
    }
}



ReconstructionAlgo_SIRT_RAM::~ReconstructionAlgo_SIRT_RAM()
{

}

void ReconstructionAlgo_SIRT_RAM::doReconstruction(map<string, string> &inputPara, map<string, string> &outputPara)
{
    cout<<"Run doReconstruction() in ReconstructionAlgo_SIRT_RAM"<<endl;

    // Input
    map<string,string>::iterator it=inputPara.find("path");
    string path;
    if(it!=inputPara.end())
    {
        path=it->second;
        cout << "File path: " << path << endl;
    }
    else
    {
        cout << "No specifit file path, set default: ./" << endl;
        path="./";
    }

    it=inputPara.find("input_mrc");
    string input_mrc;
    if(it!=inputPara.end())
    {
        input_mrc=path+"/"+it->second;
        cout << "Input file name: " << input_mrc << endl;
    }
    else
    {
        cerr << "No input file name!" << endl;
        abort();
    }
    MRC stack_orig(input_mrc.c_str(),"rb");
    if(!stack_orig.hasFile())
    {
        cerr << "Cannot open input mrc stack!" << endl;
        abort();
    }

    it=inputPara.find("output_mrc");
    string output_mrc;
    if(it!=inputPara.end())
    {
        output_mrc=path+"/"+it->second;
        cout << "Output file name: " << output_mrc << endl;
    }
    else
    {
        cout << "No output file name, set default: tomo.rec" << endl;
        output_mrc="tomo.rec";
    }

    it=inputPara.find("prfx");
    string prfx;
    if(it!=inputPara.end())
    {
        prfx=path+"/"+it->second;
        cout << "Prefix: " << prfx << endl;
    }
    else
    {
        cout << "No prfx, set default: tomo" << endl;
        prfx=path+"/"+"tomo";
    }

    it=inputPara.find("h");
    int h;
    if(it!=inputPara.end())
    {
        h=atoi(it->second.c_str());
        cout << "Reconstruction height: " << h << endl;
    }
    else
    {
        h=int(stack_orig.getNx()/4);
        cout << "No height for reconstruction, set default (Nx/4): " << h << endl;
    }

    bool skip_ctfcorrection,skip_3dctf;
    it=inputPara.find("skip_ctfcorrection");
    if(it!=inputPara.end())
    {
        skip_ctfcorrection=atoi(it->second.c_str());
        cout << "Skip CTF correction: " << skip_ctfcorrection << endl;
    }
    else
    {
        cout << "No skip_ctfcorrection, set default: 0" << endl;
        skip_ctfcorrection=0;
    }
    it=inputPara.find("skip_3dctf");
    if(it!=inputPara.end())
    {
        skip_3dctf=atoi(it->second.c_str());
        cout << "Skip 3D-CTF: " << skip_3dctf << endl;
    }
    else
    {
        cout << "No skip_3dctf, set default: 0 (Perform 3D-CTF correction)" << endl;
        skip_3dctf=0;
    }

    it=inputPara.find("input_tlt");
    string input_tlt;
    if(it!=inputPara.end())
    {
        input_tlt=path+"/"+it->second;
        cout << "Input tlt file name: " << input_tlt << endl;
    }
    else
    {
        cerr << "No input tlt file name!" << endl;
        abort();
    }
    FILE *ftlt=fopen(input_tlt.c_str(),"r");
    if(ftlt==NULL)
    {
        cerr << "Cannot open tlt file!" << endl;
        abort();
    }
    float theta[stack_orig.getNz()];
    float theta_max=0.0;
    for(int n=0;n<stack_orig.getNz();n++)
    {
        fscanf(ftlt,"%f",&theta[n]);
        if(fabs(theta[n])>theta_max)
        {
            theta_max=fabs(theta[n]);
        }
    }
    fflush(ftlt);
    fclose(ftlt);

    bool unrotated_stack=false;
    it=inputPara.find("unrotated_stack");
    if(it!=inputPara.end())
    {
        unrotated_stack=atoi(it->second.c_str());
        cout << "Input unrotated stack: " << unrotated_stack << endl;
    }
    else
    {
        cout << "No unrotated stack, input rotated stack" << endl;
        unrotated_stack=0;
    }

    int h_tilt_max=int(ceil(float(stack_orig.getNx())*sin(theta_max/180*M_PI)+float(h)*cos(theta_max/180*M_PI)))+1;    // the maximum height after tilt

    it=inputPara.find("j");
    int threads;
    if(it!=inputPara.end())
    {
        threads=atoi(it->second.c_str());
        cout << "Threads: " << threads << endl;
    }
    else
    {
        cout << "No thread num, set default: 1" << endl;
        threads=1;
    }

    string path_psi=path+"/"+"psi.txt";
    FILE *fpsi=fopen(path_psi.c_str(),"r");
    if(!fpsi)
    {
        cerr << "No psi found!" << endl;
        abort();
    }
    float psi_deg,psi_rad;
    fscanf(fpsi,"%f",&psi_deg);
    psi_rad=psi_deg*M_PI/180;
    fflush(fpsi);
    fclose(fpsi);
    cout << "psi: " << psi_deg << endl;

    CTF ctf_para[stack_orig.getNz()];
    float Cs,pix,volt,w_cos;
    if(!skip_ctfcorrection) // read in defocus file for CTF correction
    {
        it=inputPara.find("Cs");
        if(it!=inputPara.end())
        {
            Cs=atof(it->second.c_str());
            cout << "Cs (mm): " << Cs << endl;
        }
        else
        {
            cerr << "No Cs!" << endl;
            abort();
        }
        it=inputPara.find("pixel_size");
        if(it!=inputPara.end())
        {
            pix=atof(it->second.c_str());
            cout << "Pixel size (A): " << pix << endl;
        }
        else
        {
            cerr << "No pixel size!" << endl;
            abort();
        }
        it=inputPara.find("voltage");
        if(it!=inputPara.end())
        {
            volt=atof(it->second.c_str());
            cout << "Accelerating voltage (kV): " << volt << endl;
        }
        else
        {
            cerr << "No accelerating voltage!" << endl;
            abort();
        }
        it=inputPara.find("w");
        if(it!=inputPara.end())
        {
            w_cos=atof(it->second.c_str());
            cout << "Amplitude contrast: " << w_cos << endl;
        }
        else
        {
            cerr << "No amplitude contrast!" << endl;
            abort();
        }

        it=inputPara.find("defocus_file");
        string defocus_file;
        if(it!=inputPara.end())
        {
            defocus_file=path+"/"+it->second;
            cout << "Defocus file name: " << defocus_file << endl;
        }
        else
        {
            cout << "No defocus file name, set default: defocus_file.txt" << endl;
            defocus_file="defocus_file.txt";
        }
        FILE *fdefocus=fopen(defocus_file.c_str(),"r");
        if(!fdefocus)
        {
            cerr << "Cannot open defocus file!" << endl;
            abort();
        }

        for(int n=0;n<stack_orig.getNz();n++)
        {
            ctf_para[n].setN(n);
            ctf_para[n].setAllImagePara(pix,volt,Cs);
            float defocus_tmp[7];
            for(int i=0;i<7;i++)    // CTFFIND4 style
            {
                fscanf(fdefocus,"%f",&defocus_tmp[i]);
            }
            if(unrotated_stack)
            {
                ctf_para[n].setAllCTFPara(defocus_tmp[1],defocus_tmp[2],defocus_tmp[3],defocus_tmp[4],w_cos);
            }
            else
            {
                ctf_para[n].setAllCTFPara(defocus_tmp[1],defocus_tmp[2],defocus_tmp[3]-psi_deg,defocus_tmp[4],w_cos);   // 特别注意：目前的CTF估计结果取自CTFFIND4，是用原图（即未经旋转的图）估计的，因此对于重构旋转后的图，像散角（astig）也要旋转对应的角度！！！
            }
        }
        fflush(fdefocus);
        fclose(fdefocus);
    }

    it=inputPara.find("it_max");
    int it_max;
    if(it!=inputPara.end())
    {
        it_max=atoi(it->second.c_str());
        cout << "Maximum iteration: " << it_max << endl;
    }
    else
    {
        cout << "No maximum iteration, set default: 1" << endl;
        it_max=1;
    }
    
    it=inputPara.find("lambda");
    float lambda;
    if(it!=inputPara.end())
    {
        lambda=atof(it->second.c_str());
        cout << "lambda: " << lambda << endl;
    }
    else
    {
        cout << "No lambda, set default: 0.001" << endl;
        lambda=1e-3;
    }

    it=inputPara.find("initial_model");
    string initial_model;
    bool has_initial_model=0;   // initial model为之前迭代几轮的重构结果
    if(it!=inputPara.end())
    {
        has_initial_model=true;
        initial_model=path+"/"+it->second;
        cout << "Initial model: " << initial_model << endl;
    }
    else
    {
        cout << "No initial model, set default: all zero" << endl;
    }



    // Reconstruction
    cout << endl << "Reconstruction with SIRT in RAM:" << endl << endl;
    
    if(!unrotated_stack)    // rotated stack
    {
        cout << "Using rotated stack" << endl;

        float *stack_recon[stack_orig.getNy()]; // (x,z,y)
        float *stack_med[stack_orig.getNy()];   // (x,z,y)
        for(int j=0;j<stack_orig.getNy();j++)
        {
            stack_recon[j]=new float[stack_orig.getNx()*h_tilt_max];
            stack_med[j]=new float[stack_orig.getNx()*h_tilt_max];
            for(int i=0;i<stack_orig.getNx()*h_tilt_max;i++)
            {
                stack_recon[j][i]=0.0;
                stack_med[j][i]=0.0;
            }
        }
        if(has_initial_model)
        {
            MRC stack_initial(initial_model.c_str(),"rb");
            float *xz_initial=new float[stack_orig.getNx()*h];
            int z_offset=int(float(h_tilt_max-h)/2.0);
            for(int j=0;j<stack_orig.getNy();j++)
            {
                stack_initial.read2DIm_32bit(xz_initial,j);
                memcpy(stack_recon[j]+z_offset*stack_orig.getNx(),xz_initial,sizeof(float)*stack_orig.getNx()*h);
                memcpy(stack_med[j]+z_offset*stack_orig.getNx(),xz_initial,sizeof(float)*stack_orig.getNx()*h);
            }
            delete [] xz_initial;
            stack_initial.close();
        }

        cout << "Start reconstruction:" << endl;
        // loop: it_max (number of iterations)
        for(int it=0;it<it_max;it++)    // round for iteration
        {
            cout << "it " << it << ":" << endl;
            // loop: Nz (number of images)
            for(int n=0;n<stack_orig.getNz();n++)
            {
                cout << "\tImage " << n << ": " << endl;
                // Ax
                // rotation
                cout << "\t\tRotation...";
                float *xz_rot[threads];
                for(int th=0;th<threads;th++)
                {
                    xz_rot[th]=new float[stack_orig.getNx()*h_tilt_max];
                }
                #pragma omp parallel for num_threads(threads)
                // loop: Ny (number of xz-slices)
                for(int j=0;j<stack_orig.getNy();j++)
                { 
                    rotate_image(stack_recon[j],xz_rot[omp_get_thread_num()],stack_orig.getNx(),h_tilt_max,-float(stack_orig.getNx())/2.0,-float(h_tilt_max)/2.0,theta[n]*M_PI/180.0);
                    memcpy(stack_med[j],xz_rot[omp_get_thread_num()],sizeof(float)*stack_orig.getNx()*h_tilt_max);
                }
                for(int th=0;th<threads;th++)
                {
                    delete [] xz_rot[th];
                }
                cout << "Done" << endl;
                // CTF
                if(!skip_ctfcorrection)
                {
                    cout << "\t\tCTF modulation...";
                    float *bufc[threads];
                    fftwf_plan plan_fft[threads],plan_ifft[threads];
                    for(int th=0;th<threads;th++)
                    {
                        bufc[th]=new float[(stack_orig.getNx()+2-stack_orig.getNx()%2)*stack_orig.getNy()];
                        plan_fft[th]=fftwf_plan_dft_r2c_2d(stack_orig.getNy(),stack_orig.getNx(),(float*)bufc[th],reinterpret_cast<fftwf_complex*>(bufc[th]),FFTW_ESTIMATE);
                        plan_ifft[th]=fftwf_plan_dft_c2r_2d(stack_orig.getNy(),stack_orig.getNx(),reinterpret_cast<fftwf_complex*>(bufc[th]),(float*)bufc[th],FFTW_ESTIMATE);
                    }

                    float *xy_now[threads];
                    for(int th=0;th<threads;th++)
                    {
                        xy_now[th]=new float[stack_orig.getNx()*stack_orig.getNy()];
                    }
                    #pragma omp parallel for num_threads(threads)
                    // loop: h_tilt_max (maximum height for tilted volume)
                    for(int k=0;k<h_tilt_max;k++)   // loop over xy-plane
                    {   
                        // loop: Ny (number of xz-slices)
                        for(int j=0;j<stack_orig.getNy();j++)
                        {
                            memcpy(xy_now[omp_get_thread_num()]+j*stack_orig.getNx(),stack_med[j]+k*stack_orig.getNx(),sizeof(float)*stack_orig.getNx());
                        }
                        if(!skip_3dctf) // perform 3D-ART
                        {
                            ctf_modulation_omp(xy_now[omp_get_thread_num()],stack_orig.getNx(),stack_orig.getNy(),ctf_para[n],float(k)-float(h_tilt_max/2),plan_fft[omp_get_thread_num()],plan_ifft[omp_get_thread_num()],bufc[omp_get_thread_num()]);
                        }
                        else
                        {
                            ctf_modulation_omp(xy_now[omp_get_thread_num()],stack_orig.getNx(),stack_orig.getNy(),ctf_para[n],0.0,plan_fft[omp_get_thread_num()],plan_ifft[omp_get_thread_num()],bufc[omp_get_thread_num()]);
                        }
                        // loop: Ny (number of xz-slices)
                        for(int j=0;j<stack_orig.getNy();j++)
                        {
                            memcpy(stack_med[j]+k*stack_orig.getNx(),xy_now[omp_get_thread_num()]+j*stack_orig.getNx(),sizeof(float)*stack_orig.getNx());
                        }
                    }
                    for(int th=0;th<threads;th++)
                    {
                        delete [] xy_now[th];
                    }

                    for(int th=0;th<threads;th++)
                    {
                        fftwf_destroy_plan(plan_fft[th]);
	                    fftwf_destroy_plan(plan_ifft[th]);
                        delete [] bufc[th];
                    }
                    cout << "Done" << endl;
                }
                // projection
                cout << "\t\tProjection...";
                float *proj_now=new float[stack_orig.getNx()*stack_orig.getNy()];   // (x,y)
                // loop: Nx*Ny (whole xy-slice)
                for(int i=0;i<stack_orig.getNx()*stack_orig.getNy();i++)
                {
                    proj_now[i]=0.0;
                }
                #pragma omp parallel for num_threads(threads)
                // loop: Nx*Ny*h_tilt_max (whole tilted volume)
                for(int j=0;j<stack_orig.getNy();j++)   // loop over xz-plane
                {
                    for(int k=0;k<h_tilt_max;k++)
                    {
                        for(int i=0;i<stack_orig.getNx();i++)
                        {
                            proj_now[j*stack_orig.getNx()+i]+=stack_med[j][k*stack_orig.getNx()+i];
                        }
                    }
                }
                cout << "Done" << endl;
                
                // b-Ax
                cout << "\t\tCalculate residual...";
                float *image_now=new float[stack_orig.getNx()*stack_orig.getNy()];
                stack_orig.read2DIm_32bit(image_now,n);
                // loop: Nx*Ny (whole xy-slice)
                for(int i=0;i<stack_orig.getNx()*stack_orig.getNy();i++)
                {
                    image_now[i]-=proj_now[i];
                }
                cout << "Done" << endl;
                
                // (A^T)(b-Ax)
                // back-projection
                cout << "\t\tBack-projection...";
                #pragma omp parallel for num_threads(threads)
                // loop: Ny (number of xz-slices)
                for(int j=0;j<stack_orig.getNy();j++)   // loop over xz-plane
                {
                    // loop: h_tilt_max (maximum height for tilted volume)
                    for(int k=0;k<h_tilt_max;k++)
                    {
                        memcpy(stack_med[j]+k*stack_orig.getNx(),image_now+j*stack_orig.getNx(),sizeof(float)*stack_orig.getNx());
                    }
                }
                cout << "Done" << endl;
                // CTF
                if(!skip_ctfcorrection)
                {
                    cout << "\t\tCTF modulation...";
                    float *bufc[threads];
                    fftwf_plan plan_fft[threads],plan_ifft[threads];
                    for(int th=0;th<threads;th++)
                    {
                        bufc[th]=new float[(stack_orig.getNx()+2-stack_orig.getNx()%2)*stack_orig.getNy()];
                        plan_fft[th]=fftwf_plan_dft_r2c_2d(stack_orig.getNy(),stack_orig.getNx(),(float*)bufc[th],reinterpret_cast<fftwf_complex*>(bufc[th]),FFTW_ESTIMATE);
                        plan_ifft[th]=fftwf_plan_dft_c2r_2d(stack_orig.getNy(),stack_orig.getNx(),reinterpret_cast<fftwf_complex*>(bufc[th]),(float*)bufc[th],FFTW_ESTIMATE);
                    }

                    float *xy_now[threads];
                    for(int th=0;th<threads;th++)
                    {
                        xy_now[th]=new float[stack_orig.getNx()*stack_orig.getNy()];
                    }
                    #pragma omp parallel for num_threads(threads)
                    // loop: h_tilt_max (maximum height for tilted volume)
                    for(int k=0;k<h_tilt_max;k++)   // loop over xy-plane
                    {
                        // loop: Ny (number of xz-slices)
                        for(int j=0;j<stack_orig.getNy();j++)
                        {
                            memcpy(xy_now[omp_get_thread_num()]+j*stack_orig.getNx(),stack_med[j]+k*stack_orig.getNx(),sizeof(float)*stack_orig.getNx());
                        }
                        if(!skip_3dctf) // perform 3D-ART
                        {
                            ctf_modulation_omp(xy_now[omp_get_thread_num()],stack_orig.getNx(),stack_orig.getNy(),ctf_para[n],float(k)-float(h_tilt_max/2),plan_fft[omp_get_thread_num()],plan_ifft[omp_get_thread_num()],bufc[omp_get_thread_num()]);
                        }
                        else
                        {
                            ctf_modulation_omp(xy_now[omp_get_thread_num()],stack_orig.getNx(),stack_orig.getNy(),ctf_para[n],0.0,plan_fft[omp_get_thread_num()],plan_ifft[omp_get_thread_num()],bufc[omp_get_thread_num()]);
                        }
                        // loop: Ny (number of xz-slices)
                        for(int j=0;j<stack_orig.getNy();j++)
                        {
                            memcpy(stack_med[j]+k*stack_orig.getNx(),xy_now[omp_get_thread_num()]+j*stack_orig.getNx(),sizeof(float)*stack_orig.getNx());
                        }
                    }
                    for(int th=0;th<threads;th++)
                    {
                        delete [] xy_now[th];
                    }

                    for(int th=0;th<threads;th++)
                    {
                        fftwf_destroy_plan(plan_fft[th]);
	                    fftwf_destroy_plan(plan_ifft[th]);
                        delete [] bufc[th];
                    }
                    cout << "Done" << endl;
                }
                // rotation
                cout << "\t\tRotation...";
                for(int th=0;th<threads;th++)
                {
                    xz_rot[th]=new float[stack_orig.getNx()*h_tilt_max];
                }
                #pragma omp parallel for num_threads(threads)
                // loop: Ny (number of xz-slices)
                for(int j=0;j<stack_orig.getNy();j++)
                {
                    rotate_image(stack_med[j],xz_rot[omp_get_thread_num()],stack_orig.getNx(),h_tilt_max,-float(stack_orig.getNx())/2.0,-float(h_tilt_max)/2.0,-theta[n]*M_PI/180.0);
                    memcpy(stack_med[j],xz_rot[omp_get_thread_num()],sizeof(float)*stack_orig.getNx()*h_tilt_max);
                }
                for(int th=0;th<threads;th++)
                {
                    delete [] xz_rot[th];
                }
                cout << "Done" << endl;

                // x+a*(A^T)(b-Ax)
                cout << "\t\tCalculate new model...";
                #pragma omp parallel for num_threads(threads)
                // loop: Nx*Ny*h_tilt_max (whole tilted volume)
                for(int j=0;j<stack_orig.getNy();j++)   // loop over xz-plane
                {
                    for(int i=0;i<stack_orig.getNx()*h_tilt_max;i++)
                    {
                        stack_recon[j][i]+=(lambda*stack_med[j][i]);
                    }
                }
                cout << "Done" << endl;

                delete [] proj_now;
                delete [] image_now;
            }
        }

        // clip into final reconstruction volume
        cout << "Write out final reconstruction result:" << endl;
        MRC stack_final(output_mrc.c_str(),"wb");
        stack_final.createMRC_empty(stack_orig.getNx(),h,stack_orig.getNy(),2);
        int z_offset=int(float(h_tilt_max-h)/2.0);
        // loop: Ny (number of xz-slices)
        for(int j=0;j<stack_orig.getNy();j++)
        {
            float *xz_final=new float[stack_orig.getNx()*h];
            memcpy(xz_final,stack_recon[j]+z_offset*stack_orig.getNx(),sizeof(float)*stack_orig.getNx()*h);
            stack_final.write2DIm(xz_final,j);
            delete [] xz_final;
        }

        // update MRC header
        float min_thread[threads];
        float max_thread[threads];
        double mean_thread[threads];
        for(int th=0;th<threads;th++)
        {
            min_thread[th]=stack_recon[0][z_offset*stack_orig.getNx()];
            max_thread[th]=stack_recon[0][z_offset*stack_orig.getNx()];
            mean_thread[th]=0.0;
        }
        #pragma omp parallel for num_threads(threads)
        for(int j=0;j<stack_orig.getNy();j++)
        {
            double mean_now=0.0;
            for(int i=0;i<stack_orig.getNx()*h;i++)
            {
                mean_now+=stack_recon[j][i+z_offset*stack_orig.getNx()];
                if(min_thread[omp_get_thread_num()]>stack_recon[j][i+z_offset*stack_orig.getNx()])
                {
                    min_thread[omp_get_thread_num()]=stack_recon[j][i+z_offset*stack_orig.getNx()];
                }
                if(max_thread[omp_get_thread_num()]<stack_recon[j][i+z_offset*stack_orig.getNx()])
                {
                    max_thread[omp_get_thread_num()]=stack_recon[j][i+z_offset*stack_orig.getNx()];
                }
            }
            mean_thread[omp_get_thread_num()]+=(mean_now/(stack_orig.getNx()*h));
        }
        float min_all=min_thread[0];
        float max_all=max_thread[0];
        double mean_all=mean_thread[0];
        for(int th=0;th<threads;th++)
        {
            mean_all+=mean_thread[th];
            if(min_all>min_thread[th])
            {
                min_all=min_thread[th];
            }
            if(max_all<max_thread[th])
            {
                max_all=max_thread[th];
            }
        }
        mean_all/=stack_orig.getNy();
        stack_final.computeHeader(pix,false,min_all,max_all,float(mean_all));

        stack_final.close();
        cout << "Done!" << endl;

        for(int j=0;j<stack_orig.getNy();j++)
        {
            delete [] stack_recon[j];
            delete [] stack_med[j];
        }

    }
    else    // Unrotated stack
    {
        cout << "Using unrotated stack" << endl;

        float *stack_recon[h_tilt_max]; // (x,y,z)
        float *stack_med[h_tilt_max];   // (x,y,z)
        for(int k=0;k<h_tilt_max;k++)
        {
            stack_recon[k]=new float[stack_orig.getNx()*stack_orig.getNy()];
            stack_med[k]=new float[stack_orig.getNx()*stack_orig.getNy()];
            for(int i=0;i<stack_orig.getNx()*stack_orig.getNy();i++)
            {
                stack_recon[k][i]=0.0;
                stack_med[k][i]=0.0;
            }
        }
        if(has_initial_model)
        {
            MRC stack_initial(initial_model.c_str(),"rb");
            float *xy_initial=new float[stack_orig.getNx()*stack_orig.getNy()];
            int z_offset=int(float(h_tilt_max-h)/2.0);
            for(int k=0;k<h;k++)
            {
                stack_initial.read2DIm_32bit(xy_initial,k);
                memcpy(stack_recon[k+z_offset],xy_initial,sizeof(float)*stack_orig.getNx()*stack_orig.getNy());
                memcpy(stack_med[k+z_offset],xy_initial,sizeof(float)*stack_orig.getNx()*stack_orig.getNy());
            }
            delete [] xy_initial;
            stack_initial.close();
        }

        cout << "Start reconstruction:" << endl;
        // loop: it_max (number of iterations)
        for(int it=0;it<it_max;it++)    // round for iteration
        {
            cout << "it " << it << ":" << endl;
            // loop: Nz (number of images)
            for(int n=0;n<stack_orig.getNz();n++)
            {
                cout << "\tImage " << n << ": " << endl;
                // Ax
                // rotation
                cout << "\t\tRotation...";
                float theta_rad=theta[n]/180*M_PI;
                float x_offset=float(stack_orig.getNx())/2.0;
                float y_offset=float(stack_orig.getNy())/2.0;
                float z_offset=float(h_tilt_max)/2.0;
                #pragma omp parallel for num_threads(threads)
                // loop: Nx*Ny*h_tilt_max (whole volume with maximum height for tilted volume)
                for(int k=0;k<h_tilt_max;k++)
                {
                    for(int j=0;j<stack_orig.getNy();j++)
                    {
                        for(int i=0;i<stack_orig.getNx();i++)
                        {
                            float x_now=i-x_offset;
                            float y_now=j-y_offset;
                            float z_now=k-z_offset;   // move origin to the center

                            float x_psi=x_now*cos(-psi_rad)-y_now*sin(-psi_rad);
                            float y_psi=x_now*sin(-psi_rad)+y_now*cos(-psi_rad);
                            float z_psi=z_now;  // rotate tilt axis to y-axis

                            float x_tlt=x_psi*cos(-theta_rad)-z_psi*sin(-theta_rad);
                            float y_tlt=y_psi;
                            float z_tlt=x_psi*sin(-theta_rad)+z_psi*cos(-theta_rad);  // tilt

                            float x_final=x_tlt*cos(psi_rad)-y_tlt*sin(psi_rad)+x_offset;
                            float y_final=x_tlt*sin(psi_rad)+y_tlt*cos(psi_rad)+y_offset;
                            float z_final=z_tlt+z_offset;    // rotate back

                            float coeff_x=x_final-floor(x_final);
                            float coeff_y=y_final-floor(y_final);
                            float coeff_z=z_final-floor(z_final);

                            float tmp_y_1_1,tmp_y_1_2,tmp_y_2_1,tmp_y_2_2;
                            float tmp_z_1,tmp_z_2;

                            if(floor(x_final)>=0 && ceil(x_final)<stack_orig.getNx() && floor(y_final)>=0 && ceil(y_final)<stack_orig.getNy() && floor(z_final)>=0 && ceil(z_final)<h_tilt_max)
                            {
                                //stack_med[k][j*stack_orig.getNx()+i]=(1-coeff_x)*(1-coeff_y)*(1-coeff_z)*stack_recon[int(floor(z_final))][int(floor(y_final))*stack_orig.getNx()+int(floor(x_final))]+(1-coeff_x)*(1-coeff_y)*(coeff_z)*stack_recon[int(ceil(z_final))][int(floor(y_final))*stack_orig.getNx()+int(floor(x_final))]+(1-coeff_x)*(coeff_y)*(1-coeff_z)*stack_recon[int(floor(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(floor(x_final))]+(1-coeff_x)*(coeff_y)*(coeff_z)*stack_recon[int(ceil(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(floor(x_final))]+(coeff_x)*(1-coeff_y)*(1-coeff_z)*stack_recon[int(floor(z_final))][int(floor(y_final))*stack_orig.getNx()+int(ceil(x_final))]+(coeff_x)*(1-coeff_y)*(coeff_z)*stack_recon[int(ceil(z_final))][int(floor(y_final))*stack_orig.getNx()+int(ceil(x_final))]+(coeff_x)*(coeff_y)*(1-coeff_z)*stack_recon[int(floor(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(ceil(x_final))]+(coeff_x)*(coeff_y)*(coeff_z)*stack_recon[int(ceil(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(ceil(x_final))];
                                tmp_y_1_1=(1-coeff_x)*stack_recon[int(floor(z_final))][int(floor(y_final))*stack_orig.getNx()+int(floor(x_final))]+(coeff_x)*stack_recon[int(floor(z_final))][int(floor(y_final))*stack_orig.getNx()+int(ceil(x_final))];
                                tmp_y_1_2=(1-coeff_x)*stack_recon[int(floor(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(floor(x_final))]+(coeff_x)*stack_recon[int(floor(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(ceil(x_final))];
                                tmp_y_2_1=(1-coeff_x)*stack_recon[int(ceil(z_final))][int(floor(y_final))*stack_orig.getNx()+int(floor(x_final))]+(coeff_x)*stack_recon[int(ceil(z_final))][int(floor(y_final))*stack_orig.getNx()+int(ceil(x_final))];
                                tmp_y_2_2=(1-coeff_x)*stack_recon[int(ceil(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(floor(x_final))]+(coeff_x)*stack_recon[int(ceil(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(ceil(x_final))];
                                tmp_z_1=(1-coeff_y)*tmp_y_1_1+(coeff_y)*tmp_y_1_2;
                                tmp_z_2=(1-coeff_y)*tmp_y_2_1+(coeff_y)*tmp_y_2_2;
                                stack_med[k][j*stack_orig.getNx()+i]=(1-coeff_z)*tmp_z_1+(coeff_z)*tmp_z_2;
                            }
                            else
                            {
                                stack_med[k][j*stack_orig.getNx()+i]=0.0;
                            }
                        }
                    }
                }
                cout << "Done" << endl;
                // CTF
                if(!skip_ctfcorrection)
                {
                    cout << "\t\tCTF modulation...";
                    float *bufc[threads];
                    fftwf_plan plan_fft[threads],plan_ifft[threads];
                    for(int th=0;th<threads;th++)
                    {
                        bufc[th]=new float[(stack_orig.getNx()+2-stack_orig.getNx()%2)*stack_orig.getNy()];
                        plan_fft[th]=fftwf_plan_dft_r2c_2d(stack_orig.getNy(),stack_orig.getNx(),(float*)bufc[th],reinterpret_cast<fftwf_complex*>(bufc[th]),FFTW_ESTIMATE);
                        plan_ifft[th]=fftwf_plan_dft_c2r_2d(stack_orig.getNy(),stack_orig.getNx(),reinterpret_cast<fftwf_complex*>(bufc[th]),(float*)bufc[th],FFTW_ESTIMATE);
                    }

                    #pragma omp parallel for num_threads(threads)
                    // loop: h_tilt_max (maximum height for tilted volume)
                    for(int k=0;k<h_tilt_max;k++)   // loop over xy-plane
                    {
                        if(!skip_3dctf) // perform 3D-ART
                        {
                            ctf_modulation_omp(stack_med[k],stack_orig.getNx(),stack_orig.getNy(),ctf_para[n],float(k)-float(h_tilt_max)/2.0,plan_fft[omp_get_thread_num()],plan_ifft[omp_get_thread_num()],bufc[omp_get_thread_num()]);
                        }
                        else
                        {
                            ctf_modulation_omp(stack_med[k],stack_orig.getNx(),stack_orig.getNy(),ctf_para[n],0.0,plan_fft[omp_get_thread_num()],plan_ifft[omp_get_thread_num()],bufc[omp_get_thread_num()]);
                        }
                    }

                    for(int th=0;th<threads;th++)
                    {
                        fftwf_destroy_plan(plan_fft[th]);
	                    fftwf_destroy_plan(plan_ifft[th]);
                        delete [] bufc[th];
                    }
                    cout << "Done" << endl;
                }
                // projection
                cout << "\t\tProjection...";
                float *proj_now=new float[stack_orig.getNx()*stack_orig.getNy()];   // (x,y)
                // loop: Nx*Ny (whole xy-slice)
                for(int i=0;i<stack_orig.getNx()*stack_orig.getNy();i++)
                {
                    proj_now[i]=0.0;
                }
                #pragma omp parallel for num_threads(threads)
                // loop: Nx*Ny*h_tilt_max (whole volume with maximum height for tilted volume)
                for(int j=0;j<stack_orig.getNy();j++)   // loop over xz-plane
                {
                    for(int k=0;k<h_tilt_max;k++)
                    {
                        for(int i=0;i<stack_orig.getNx();i++)
                        {
                            proj_now[j*stack_orig.getNx()+i]+=stack_med[k][j*stack_orig.getNx()+i];
                        }
                    }
                }
                cout << "Done" << endl;
                
                // b-Ax
                cout << "\t\tCalculate residual...";
                float *image_now=new float[stack_orig.getNx()*stack_orig.getNy()];
                stack_orig.read2DIm_32bit(image_now,n);
                // loop: Nx*Ny (whole xy-slice)
                for(int i=0;i<stack_orig.getNx()*stack_orig.getNy();i++)
                {
                    image_now[i]-=proj_now[i];
                }
                cout << "Done" << endl;
                
                // (A^T)(b-Ax)
                // back-projection
                cout << "\t\tBack-projection...";
                #pragma omp parallel for num_threads(threads)
                // loop: Ny (number of xz-slices)
                for(int j=0;j<stack_orig.getNy();j++)   // loop over xz-plane
                {
                    // loop: h_tilt_max (maximum height for tilted volume)
                    for(int k=0;k<h_tilt_max;k++)
                    {
                        memcpy(stack_med[k]+j*stack_orig.getNx(),image_now+j*stack_orig.getNx(),sizeof(float)*stack_orig.getNx());
                    }
                }
                cout << "Done" << endl;
                // CTF
                if(!skip_ctfcorrection)
                {
                    cout << "\t\tCTF modulation...";
                    float *bufc[threads];
                    fftwf_plan plan_fft[threads],plan_ifft[threads];
                    for(int th=0;th<threads;th++)
                    {
                        bufc[th]=new float[(stack_orig.getNx()+2-stack_orig.getNx()%2)*stack_orig.getNy()];
                        plan_fft[th]=fftwf_plan_dft_r2c_2d(stack_orig.getNy(),stack_orig.getNx(),(float*)bufc[th],reinterpret_cast<fftwf_complex*>(bufc[th]),FFTW_ESTIMATE);
                        plan_ifft[th]=fftwf_plan_dft_c2r_2d(stack_orig.getNy(),stack_orig.getNx(),reinterpret_cast<fftwf_complex*>(bufc[th]),(float*)bufc[th],FFTW_ESTIMATE);
                    }

                    #pragma omp parallel for num_threads(threads)
                    // loop: h_tilt_max (maximum height for tilted volume)
                    for(int k=0;k<h_tilt_max;k++)   // loop over xy-plane
                    {
                        if(!skip_3dctf) // perform 3D-ART
                        {
                            ctf_modulation_omp(stack_med[k],stack_orig.getNx(),stack_orig.getNy(),ctf_para[n],float(k)-float(h_tilt_max)/2.0,plan_fft[omp_get_thread_num()],plan_ifft[omp_get_thread_num()],bufc[omp_get_thread_num()]);
                        }
                        else
                        {
                            ctf_modulation_omp(stack_med[k],stack_orig.getNx(),stack_orig.getNy(),ctf_para[n],0.0,plan_fft[omp_get_thread_num()],plan_ifft[omp_get_thread_num()],bufc[omp_get_thread_num()]);
                        }
                    }

                    for(int th=0;th<threads;th++)
                    {
                        fftwf_destroy_plan(plan_fft[th]);
	                    fftwf_destroy_plan(plan_ifft[th]);
                        delete [] bufc[th];
                    }
                    cout << "Done" << endl;
                }
                // rotation & refinement (x+a*(A^T)(b-Ax))
                cout << "\t\tRotation & Calculate new model...";
                #pragma omp parallel for num_threads(threads)
                // loop: Nx*Ny*h_tilt_max (whole volume with maximum height for tilted volume)
                for(int k=0;k<h_tilt_max;k++)
                {
                    for(int j=0;j<stack_orig.getNy();j++)
                    {
                        for(int i=0;i<stack_orig.getNx();i++)
                        {
                            float tmp;
                            float x_now=i-x_offset;
                            float y_now=j-y_offset;
                            float z_now=k-z_offset;   // move origin to the center

                            float x_psi=x_now*cos(-psi_rad)-y_now*sin(-psi_rad);
                            float y_psi=x_now*sin(-psi_rad)+y_now*cos(-psi_rad);
                            float z_psi=z_now;  // rotate tilt axis to y-axis

                            float x_tlt=x_psi*cos(theta_rad)-z_psi*sin(theta_rad);
                            float y_tlt=y_psi;
                            float z_tlt=x_psi*sin(theta_rad)+z_psi*cos(theta_rad);  // tilt

                            float x_final=x_tlt*cos(psi_rad)-y_tlt*sin(psi_rad)+x_offset;
                            float y_final=x_tlt*sin(psi_rad)+y_tlt*cos(psi_rad)+y_offset;
                            float z_final=z_tlt+z_offset;    // rotate back

                            float coeff_x=x_final-floor(x_final);
                            float coeff_y=y_final-floor(y_final);
                            float coeff_z=z_final-floor(z_final);

                            float tmp_y_1_1,tmp_y_1_2,tmp_y_2_1,tmp_y_2_2;
                            float tmp_z_1,tmp_z_2;

                            if(floor(x_final)>=0 && ceil(x_final)<stack_orig.getNx() && floor(y_final)>=0 && ceil(y_final)<stack_orig.getNy() && floor(z_final)>=0 && ceil(z_final)<h_tilt_max)
                            {
                                //tmp=(1-coeff_x)*(1-coeff_y)*(1-coeff_z)*stack_med[int(floor(z_final))][int(floor(y_final))*stack_orig.getNx()+int(floor(x_final))]+(1-coeff_x)*(1-coeff_y)*(coeff_z)*stack_med[int(ceil(z_final))][int(floor(y_final))*stack_orig.getNx()+int(floor(x_final))]+(1-coeff_x)*(coeff_y)*(1-coeff_z)*stack_med[int(floor(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(floor(x_final))]+(1-coeff_x)*(coeff_y)*(coeff_z)*stack_med[int(ceil(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(floor(x_final))]+(coeff_x)*(1-coeff_y)*(1-coeff_z)*stack_med[int(floor(z_final))][int(floor(y_final))*stack_orig.getNx()+int(ceil(x_final))]+(coeff_x)*(1-coeff_y)*(coeff_z)*stack_med[int(ceil(z_final))][int(floor(y_final))*stack_orig.getNx()+int(ceil(x_final))]+(coeff_x)*(coeff_y)*(1-coeff_z)*stack_med[int(floor(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(ceil(x_final))]+(coeff_x)*(coeff_y)*(coeff_z)*stack_med[int(ceil(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(ceil(x_final))];
                                tmp_y_1_1=(1-coeff_x)*stack_med[int(floor(z_final))][int(floor(y_final))*stack_orig.getNx()+int(floor(x_final))]+(coeff_x)*stack_med[int(floor(z_final))][int(floor(y_final))*stack_orig.getNx()+int(ceil(x_final))];
                                tmp_y_1_2=(1-coeff_x)*stack_med[int(floor(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(floor(x_final))]+(coeff_x)*stack_med[int(floor(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(ceil(x_final))];
                                tmp_y_2_1=(1-coeff_x)*stack_med[int(ceil(z_final))][int(floor(y_final))*stack_orig.getNx()+int(floor(x_final))]+(coeff_x)*stack_med[int(ceil(z_final))][int(floor(y_final))*stack_orig.getNx()+int(ceil(x_final))];
                                tmp_y_2_2=(1-coeff_x)*stack_med[int(ceil(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(floor(x_final))]+(coeff_x)*stack_med[int(ceil(z_final))][int(ceil(y_final))*stack_orig.getNx()+int(ceil(x_final))];
                                tmp_z_1=(1-coeff_y)*tmp_y_1_1+(coeff_y)*tmp_y_1_2;
                                tmp_z_2=(1-coeff_y)*tmp_y_2_1+(coeff_y)*tmp_y_2_2;
                                tmp=(1-coeff_z)*tmp_z_1+(coeff_z)*tmp_z_2;
                            }
                            else
                            {
                                tmp=0.0;
                            }
                            stack_recon[k][j*stack_orig.getNx()+i]+=(lambda*tmp);
                        }
                    }
                }
                cout << "Done" << endl;
                
                delete [] proj_now;
                delete [] image_now;
            }
        }

        // clip into final reconstruction volume
        cout << "Write out final reconstruction result:" << endl;
        MRC stack_final(output_mrc.c_str(),"wb");
        stack_final.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),h,2);
        int z_offset=int(float(h_tilt_max-h)/2.0);
        for(int k=0;k<h;k++)
        {
            stack_final.write2DIm(stack_recon[k+z_offset],k);
        }

        // update MRC header
        float min_thread[threads];
        float max_thread[threads];
        double mean_thread[threads];
        for(int th=0;th<threads;th++)
        {
            min_thread[th]=stack_recon[z_offset][0];
            max_thread[th]=stack_recon[z_offset][0];
            mean_thread[th]=0.0;
        }
        #pragma omp parallel for num_threads(threads)
        for(int k=0;k<h;k++)
        {
            double mean_now=0.0;
            for(int i=0;i<stack_orig.getNx()*stack_orig.getNy();i++)
            {
                mean_now+=stack_recon[k+z_offset][i];
                if(min_thread[omp_get_thread_num()]>stack_recon[k+z_offset][i])
                {
                    min_thread[omp_get_thread_num()]=stack_recon[k+z_offset][i];
                }
                if(max_thread[omp_get_thread_num()]<stack_recon[k+z_offset][i])
                {
                    max_thread[omp_get_thread_num()]=stack_recon[k+z_offset][i];
                }
            }
            mean_thread[omp_get_thread_num()]+=(mean_now/(stack_orig.getNx()*stack_orig.getNy()));
        }
        float min_all=min_thread[0];
        float max_all=max_thread[0];
        double mean_all=mean_thread[0];
        for(int th=0;th<threads;th++)
        {
            mean_all+=mean_thread[th];
            if(min_all>min_thread[th])
            {
                min_all=min_thread[th];
            }
            if(max_all<max_thread[th])
            {
                max_all=max_thread[th];
            }
        }
        mean_all/=h;
        stack_final.computeHeader(pix,false,min_all,max_all,float(mean_all));

        stack_final.close();
        cout << "Done!" << endl;

        for(int k=0;k<h_tilt_max;k++)
        {
            delete [] stack_recon[k];
            delete [] stack_med[k];
        }

    }

    cout << endl << "Finish reconstruction successfully!" << endl;
    cout << "All results save in: " << path << endl << endl;

}
