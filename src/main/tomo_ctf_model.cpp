#include <stdio.h>
#include "util.h"
#include "mrc.h"
#include "time.h"
#include "math.h"
#include "fftw3.h"
#include "CTF.h"
#include "omp.h"

void getTime(time_t start_time)
{
    time_t end_time;
    time(&end_time);

    double seconds_total=difftime(end_time,start_time);
    int hours=((int)seconds_total)/3600;
    int minutes=(((int)seconds_total)%3600)/60;
    int seconds=(((int)seconds_total)%3600)%60;

    cout << "Time elapsed: ";
    if(hours>0)
    {
        cout << hours << "h ";
    }
    if(minutes > 0 || hours > 0)
    {
        cout << minutes << "m ";
    }
    cout << seconds << "s" << endl << endl;
}



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

static void ctf_correction(float *image,int Nx,int Ny,CTF ctf,bool flip_contrast,float z_offset)   // z_offset in pixels
{
    fftwf_plan plan_fft,plan_ifft;
    float *bufc=new float[(Nx+2-Nx%2)*Ny];
    plan_fft=fftwf_plan_dft_r2c_2d(Ny,Nx,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);
    plan_ifft=fftwf_plan_dft_c2r_2d(Ny,Nx,reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);
    buf2fft(image,bufc,Nx,Ny);
    fftwf_execute(plan_fft);

    for(int i=0;i<(Nx+2-Nx%2);i+=2)
    {
        for(int j=0;j<Ny;j++)
        {
            float ctf_now=ctf.computeCTF2D(i/2,j,Nx,Ny,true,flip_contrast,z_offset);
            bufc[i+j*(Nx+2-Nx%2)]*=ctf_now;
            bufc[(i+1)+j*(Nx+2-Nx%2)]*=ctf_now;
        }
    }

    fftwf_execute(plan_ifft);
    fft2buf(image,bufc,Nx,Ny);
    for(int i=0;i<Nx*Ny;i++)   // normalization
    {
        image[i]=image[i]/(Nx*Ny);
    }
    fftwf_destroy_plan(plan_fft);
	fftwf_destroy_plan(plan_ifft);
    delete [] bufc;
}

static void ctf_correction_omp(float *image,int Nx,int Ny,CTF ctf,bool flip_contrast,float z_offset,fftwf_plan plan_fft,fftwf_plan plan_ifft,float *bufc)   // z_offset in pixels
{
    buf2fft(image,bufc,Nx,Ny);
    fftwf_execute(plan_fft);

    for(int i=0;i<(Nx+2-Nx%2);i+=2)
    {
        for(int j=0;j<Ny;j++)
        {
            float ctf_now=ctf.computeCTF2D(i/2,j,Nx,Ny,true,flip_contrast,z_offset);
            bufc[i+j*(Nx+2-Nx%2)]*=ctf_now;
            bufc[(i+1)+j*(Nx+2-Nx%2)]*=ctf_now;
        }
    }

    fftwf_execute(plan_ifft);
    fft2buf(image,bufc,Nx,Ny);
    for(int i=0;i<Nx*Ny;i++)   // normalization
    {
        image[i]=image[i]/(Nx*Ny);
    }
}

static void fft_shift(float *image,int Nx,int Ny)   // move origin to center
{
    // move x-axis
    float *image_shift_x=new float[Nx*Ny];
    for(int i=0;i<Nx*Ny;i++)
    {
        image_shift_x[i]=0.0;
    }
    if(Nx%2==0)
    {
        int x_offset=Nx/2;
        for(int j=0;j<Ny;j++)
        {
            for(int i=0;i<x_offset;i++)
            {
                image_shift_x[j*Nx+i+x_offset]=image[j*Nx+i];
            }
            for(int i=0;i<x_offset;i++)
            {
                image_shift_x[j*Nx+i]=image[j*Nx+i+x_offset];
            }
        }
    }
    else
    {
        int x_offset=floor(Nx/2);
        for(int j=0;j<Ny;j++)
        {
            for(int i=0;i<=x_offset;i++)
            {
                image_shift_x[j*Nx+i+x_offset]=image[j*Nx+i];
            }
            for(int i=0;i<x_offset;i++)
            {
                image_shift_x[j*Nx+i]=image[j*Nx+i+x_offset+1];
            }
        }
    }
    
    // move y-axis
    float *image_shift_y=new float[Nx*Ny];
    for(int i=0;i<Nx*Ny;i++)
    {
        image_shift_y[i]=0.0;
    }
    if(Ny%2==0)
    {
        int y_offset=Ny/2;
        for(int i=0;i<Nx;i++)
        {
            for(int j=0;j<y_offset;j++)
            {
                image_shift_y[(j+y_offset)*Nx+i]=image_shift_x[j*Nx+i];
            }
            for(int j=0;j<y_offset;j++)
            {
                image_shift_y[j*Nx+i]=image_shift_x[(j+y_offset)*Nx+i];
            }
        }
    }
    else
    {
        int y_offset=floor(Ny/2);
        for(int i=0;i<Nx;i++)
        {
            for(int j=0;j<=y_offset;j++)
            {
                image_shift_y[(j+y_offset)*Nx+i]=image_shift_x[j*Nx+i];
            }
            for(int j=0;j<y_offset;j++)
            {
                image_shift_y[j*Nx+i]=image_shift_x[(j+y_offset+1)*Nx+i];
            }
        }
    }

    memcpy(image,image_shift_y,sizeof(float)*Nx*Ny);
    delete [] image_shift_x;
    delete [] image_shift_y;
}



int main(int argc, char **argv)
{
    time_t start_time;
    time(&start_time);

    map<string, string> inputPara;
    map<string, string> outputPara;
    const char *paraFileName = "../conf/para_ctf_model.conf";
    readParaFile(inputPara, paraFileName);
    getAllParas(inputPara);

    // read input parameters
    map<string,string>::iterator it=inputPara.find("output_mrc");
    string output_mrc;
    if(it!=inputPara.end())
    {
        output_mrc=it->second;
        cout << "Output file name: " << output_mrc << endl;
    }
    else
    {
        cerr << "No output file name, set default: tomo_ctf_model.mrc" << endl;
        output_mrc="tomo_ctf_model.mrc";
    }

    int N,nx,ny,h;
    it=inputPara.find("N");
    if(it!=inputPara.end())
    {
        N=atoi(it->second.c_str());
        cout << "Number of micrographs: " << N << endl;
    }
    else
    {
        cerr << "No number of micrographs!" << endl;
        abort();
    }
    it=inputPara.find("length");
    if(it!=inputPara.end())
    {
        nx=atoi(it->second.c_str());
        cout << "Reconstruction length: " << nx << endl;
    }
    else
    {
        cerr << "No length for reconstruction!" << endl;
        abort();
    }
    it=inputPara.find("width");
    if(it!=inputPara.end())
    {
        ny=atoi(it->second.c_str());
        cout << "Reconstruction width: " << ny << endl;
    }
    else
    {
        cerr << "No width for reconstruction!" << endl;
        abort();
    }
    it=inputPara.find("height");
    if(it!=inputPara.end())
    {
        h=atoi(it->second.c_str());
        cout << "Reconstruction height: " << h << endl;
    }
    else
    {
        h=int(nx/4);
        cout << "No height for reconstruction, set default (Nx/4): " << h << endl;
    }

    it=inputPara.find("input_tlt");
    string input_tlt;
    if(it!=inputPara.end())
    {
        input_tlt=it->second;
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
    float theta[N];
    float theta_max=0.0;
    for(int n=0;n<N;n++)
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

    FILE *fpsi=fopen("psi.txt","r");
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

    CTF ctf_para[N];
    float Cs,pix,volt,w_cos;
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
        defocus_file=it->second;
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

    for(int n=0;n<N;n++)
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

    bool skip_3dctf;
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

    it=inputPara.find("flip_contrast");
    bool flip_contrast;
    if(it!=inputPara.end())
    {
        flip_contrast=atoi(it->second.c_str());
        cout << "Flip contrast: " << flip_contrast << endl;
    }
    else
    {
        cout << "No flip contrast, set default: 0" << endl;
        flip_contrast=false;
    }

    int defocus_step=1;
    if(!skip_3dctf)
    {
        it=inputPara.find("defocus_step");
        if(it!=inputPara.end())
        {
            defocus_step=atoi(it->second.c_str());
            cout << "Defocus step (pixels): " << defocus_step << endl;
        }
        else
        {
            cout << "No defocus step, set default: 1" << endl;
            defocus_step=1;
        }
    }

    int h_tilt_max=int(ceil(float(nx)*sin(theta_max/180*M_PI)+float(h)*cos(theta_max/180*M_PI)))+1;    // the maximum height after tilt



    // Reconstruction
    cout << endl << "Reconstruction with (W)BP in RAM:" << endl << endl;
    if(!unrotated_stack)    // input rotated stack (y-axis as tilt axis)
    {
        cout << "Using rotated stack" << endl;

        float *stack_recon[ny]; // (x,z,y)
        for(int j=0;j<ny;j++)
        {
            stack_recon[j]=new float[nx*h];
            for(int i=0;i<nx*h;i++)
            {
                stack_recon[j][i]=0.0;
            }
        }

        cout << "Start reconstruction:" << endl;
        float x_orig_offset=float(nx)/2.0,z_orig_offset=float(h)/2.0;
        for(int n=0;n<N;n++)   // loop for every micrograph
        {
            cout << "Image " << n << ":" << endl;
            float theta_rad=theta[n]/180*M_PI;
            float *image_now=new float[nx*ny];
            for(int i=0;i<nx*ny;i++)
            {
                image_now[i]=0.0;
            }
            image_now[0]=1.0;
            float *image_now_backup=new float[nx*ny];
            memcpy(image_now_backup,image_now,sizeof(float)*nx*ny);

            if(skip_3dctf)  // perform simple CTF correction (no consideration of height)
            {
                cout << "\tPerform uniform CTF correction!" << endl;

                // correction
                cout << "\tStart correction: " << endl;
                ctf_correction(image_now,nx,ny,ctf_para[n],flip_contrast,0.0);
                cout << "\tDone!" << endl;

                cout << "\tStart shifting: " << endl;
                fft_shift(image_now,nx,ny); // move origin to center
                cout << "\tDone!" << endl;

                // recontruction
                cout << "\tStart reconstuction, loop over xz-plane:" << endl;

                #pragma omp parallel for num_threads(threads)
                for(int j=0;j<ny;j++)
                {
//                        cout << "\t\t" << j << ": ";
                    float *strip_now=new float[nx];
                    memcpy(strip_now,image_now+j*nx,sizeof(float)*nx);

                    // BP
//                        cout << "Back-projecting...";
                    float *recon_now=new float[nx*h];   // 第一维x，第二维z
//                        memset(recon_now,0.0,sizeof(recon_now));
                    for(int i=0;i<nx*h;i++)
                    {
                        recon_now[i]=0.0;
                    }
                    for(int i=0;i<nx;i++)   // loop for the xz-plane to perform BP
                    {
                        for(int k=0;k<h;k++)
                        {
                            float x_orig=(float(i)-x_orig_offset)*cos(theta_rad)-(float(k)-z_orig_offset)*sin(theta_rad)+x_orig_offset,z_orig=(float(i)-x_orig_offset)*sin(theta_rad)+(float(k)-z_orig_offset)*cos(theta_rad)+z_orig_offset;
                            float coeff=x_orig-floor(x_orig);
                            if(floor(x_orig)>=0 && ceil(x_orig)<nx)
                            {
                                recon_now[i+k*nx]=(1-coeff)*strip_now[int(floor(x_orig))]+(coeff)*strip_now[int(ceil(x_orig))];
                            }
                            else
                            {
                                recon_now[i+k*nx]=0.0;
                            }
                        }
                    }

                    for(int i=0;i<nx*h;i++)
                    {
                        stack_recon[j][i]+=recon_now[i];
                    }
//                        cout << "Done" << endl;

                    delete [] recon_now;
                    delete [] strip_now;
                }
                cout << "\tDone" << endl;

            }
            else    // perform 3D-CTF correction
            {
                cout << "\tPerform 3D-CTF correction!" << endl;

                // write all corrected and weighted images in one mrc stack
                cout << "\tPerform 3D correction & save corrected stack:" << endl;
                float *stack_corrected[int(h_tilt_max/defocus_step)+1]; // 第一维遍历不同高度，第二维x，第三维y
                int n_zz=0;

                // 3D-CTF correction
                cout << "\tStart 3D-CTF correction..." << endl;
                // setup fftw3 plan for omp
                fftwf_plan plan_fft[threads],plan_ifft[threads];
                float *bufc[threads];
                for(int th=0;th<threads;th++)
                {
                    bufc[th]=new float[(nx+2-nx%2)*ny];
                    plan_fft[th]=fftwf_plan_dft_r2c_2d(ny,nx,(float*)bufc[th],reinterpret_cast<fftwf_complex*>(bufc[th]),FFTW_ESTIMATE);
                    plan_ifft[th]=fftwf_plan_dft_c2r_2d(ny,nx,reinterpret_cast<fftwf_complex*>(bufc[th]),(float*)bufc[th],FFTW_ESTIMATE);
                }
                
                #pragma omp parallel for num_threads(threads) reduction(+:n_zz)
                for(int zz=-int(h_tilt_max/2);zz<int(h_tilt_max/2);zz+=defocus_step)    // loop over every height (correct with different defocus)
                {
                    float *image_now_th=new float[nx*ny];
                    memcpy(image_now_th,image_now,sizeof(float)*nx*ny); // get the raw image!!!

                    // correction
                    int n_z=(zz+int(h_tilt_max/2))/defocus_step;
                    ctf_correction_omp(image_now_th,nx,ny,ctf_para[n],flip_contrast,float(zz)+float(defocus_step-1)/2,plan_fft[omp_get_thread_num()],plan_ifft[omp_get_thread_num()],bufc[omp_get_thread_num()]);

                    // save
                    stack_corrected[n_z]=new float[nx*ny];
                    memcpy(stack_corrected[n_z],image_now_th,sizeof(float)*nx*ny);
                    n_zz++;

                    delete [] image_now_th;
                }
                cout << "\tDone!" << endl;

                for(int th=0;th<threads;th++)
                {
                    fftwf_destroy_plan(plan_fft[th]);
                    fftwf_destroy_plan(plan_ifft[th]);
                    delete [] bufc[th];
                }

                // recontruction

                cout << "\tPerform reconstruction:" << endl;
                #pragma omp parallel for num_threads(threads)
                for(int j=0;j<ny;j++)
                {
//                        cout << "\t\tSlice " << j << ": ";

                    // BP
                    float *recon_now=new float[nx*h];   // 第一维x，第二维z
                    for(int i=0;i<nx*h;i++)
                    {
                        recon_now[i]=0.0;
                    }
                    for(int i=0;i<nx;i++)   // loop for the xz-plane to perform BP
                    {
                        for(int k=0;k<h;k++)
                        {
                            float x_orig=(float(i)-x_orig_offset)*cos(theta_rad)-(float(k)-z_orig_offset)*sin(theta_rad)+x_orig_offset,z_orig=(float(i)-x_orig_offset)*sin(theta_rad)+(float(k)-z_orig_offset)*cos(theta_rad)+z_orig_offset;
                            float coeff=x_orig-floor(x_orig);
                            int n_z=floor(((z_orig-z_orig_offset)+int(h_tilt_max/2))/defocus_step);    // the num in the corrected stack for the current height
                            if(n_z>=0 && n_z<n_zz)
                            {
                                if(floor(x_orig)>=0 && ceil(x_orig)<nx)
                                {
//                                        corrected_mrc.readLine(strip_now,n_z,j);
//                                        recon_now[i+k*nx]=(1-coeff)*strip_now[int(floor(x_orig))]+(coeff)*strip_now[int(ceil(x_orig))];
                                    recon_now[i+k*nx]=(1-coeff)*stack_corrected[n_z][j*nx+int(floor(x_orig))]+(coeff)*stack_corrected[n_z][j*nx+int(ceil(x_orig))];
                                }
                                else
                                {
                                    recon_now[i+k*nx]=0.0;
                                }
                            }
                            else
                            {
                                recon_now[i+k*nx]=0.0;
                            }
                        }
                    }

                    for(int i=0;i<nx*h;i++)
                    {
                        stack_recon[j][i]+=recon_now[i];
                    }
//                        cout << "Done" << endl;
                    delete [] recon_now;
                }
                cout << "\tDone" << endl;

                for(int n_z=0;n_z<n_zz;n_z++)
                {
                    delete [] stack_corrected[n_z];
                }
            }
            delete [] image_now;
            delete [] image_now_backup;
        }

        // write out final result
        cout << "Wrtie out final reconstruction result:" << endl;
        MRC stack_final(output_mrc.c_str(),"wb");
        stack_final.createMRC_empty(nx,h,ny,2); // (x,z,y)
        for(int j=0;j<ny;j++)
        {
            stack_final.write2DIm(stack_recon[j],j);
//            delete [] stack_recon[j];
        }

        // update MRC header
        float min_thread[threads],max_thread[threads];
        double mean_thread[threads];
        for(int th=0;th<threads;th++)
        {
            min_thread[th]=stack_recon[0][0];
            max_thread[th]=stack_recon[0][0];
            mean_thread[th]=0.0;
        }
        #pragma omp parallel for num_threads(threads)
        for(int j=0;j<ny;j++)
        {
            double mean_now=0.0;
            for(int i=0;i<nx*h;i++)
            {
                mean_now+=stack_recon[j][i];
                if(min_thread[omp_get_thread_num()]>stack_recon[j][i])
                {
                    min_thread[omp_get_thread_num()]=stack_recon[j][i];
                }
                if(max_thread[omp_get_thread_num()]<stack_recon[j][i])
                {
                    max_thread[omp_get_thread_num()]=stack_recon[j][i];
                }
            }
            mean_thread[omp_get_thread_num()]+=(mean_now/(nx*h));
        }
        float min_all=min_thread[0],max_all=max_thread[0];
        double mean_all=mean_thread[0];
        for(int th=1;th<threads;th++)
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
        mean_all/=ny;
        stack_final.computeHeader(pix,false,min_all,max_all,float(mean_all));

        for(int j=0;j<ny;j++)
        {
            delete [] stack_recon[j];
        }

        stack_final.close();
        cout << "Done" << endl;

    }
    else    // input unrotated stack
    {
        cout << "Using unrotated stack" << endl;

        float *stack_recon[h];  // (x,y,z)
        for(int k=0;k<h;k++)
        {
            stack_recon[k]=new float[nx*ny];
            for(int i=0;i<nx*ny;i++)
            {
                stack_recon[k][i]=0.0;
            }
        }

        cout << "Start reconstruction:" << endl;
        float x_orig_offset=float(nx)/2.0,y_orig_offset=float(ny)/2.0,z_orig_offset=float(h)/2.0;
        for(int n=0;n<N;n++)   // loop for every micrograph
        {
            cout << "Image " << n << ":" << endl;
            float theta_rad=theta[n]/180*M_PI;
            float *image_now=new float[nx*ny];
            for(int i=0;i<nx*ny;i++)
            {
                image_now[i]=0.0;
            }
            image_now[0]=1.0;
            float *image_now_backup=new float[nx*ny];
            memcpy(image_now_backup,image_now,sizeof(float)*nx*ny);
          
            if(skip_3dctf)  // perform simple CTF correction (no consideration of height)
            {
                cout << "\tPerform uniform CTF correction!" << endl;

                // correction
                cout << "\tStart correction: " << endl;
                ctf_correction(image_now,nx,ny,ctf_para[n],flip_contrast,0.0);
                cout << "\tDone!" << endl;

                // recontruction
                cout << "\tStart reconstuction, loop over xy-plane:" << endl;

                #pragma omp parallel for num_threads(threads)
                for(int k=0;k<h;k++)    // loop for every xy-plane
                {
//                        cout << "\tSlice " << k << ": ";
                    float *recon_now=new float[nx*ny];  // (x,y)
                    for(int i=0;i<nx;i++)
                    {
                        for(int j=0;j<ny;j++)
                        {
                            float x=i-x_orig_offset,y=j-y_orig_offset,z=k-z_orig_offset;
                            float x_rot=x*cos(-psi_rad)-y*sin(-psi_rad),y_rot=x*sin(-psi_rad)+y*cos(-psi_rad),z_rot=z;  // rotate tilt-axis to y-axis
                            float x_tlt=x_rot*cos(theta_rad)-z_rot*sin(theta_rad),y_tlt=y_rot,z_tlt=x_rot*sin(theta_rad)+z_rot*cos(theta_rad);  // tilt the specimen around y-axis
                            float x_final=x_tlt*cos(psi_rad)-y_tlt*sin(psi_rad)+x_orig_offset,y_final=x_tlt*sin(psi_rad)+y_tlt*cos(psi_rad)+y_orig_offset;  // rotate back to the origal image
                            if(floor(x_final)>=0 && ceil(x_final)<nx && floor(y_final)>=0 && ceil(y_final)<ny)
                            {
                                float coeff_x=x_final-floor(x_final),coeff_y=y_final-floor(y_final);
                                recon_now[i+j*nx]=(1-coeff_x)*(1-coeff_y)*image_now[int(floor(x_final)+floor(y_final)*nx)]+(coeff_x)*(1-coeff_y)*image_now[int(ceil(x_final)+floor(y_final)*nx)]+(1-coeff_x)*(coeff_y)*image_now[int(floor(x_final)+ceil(y_final)*nx)]+(coeff_x)*(coeff_y)*image_now[int(ceil(x_final)+ceil(y_final)*nx)];
                            }
                            else
                            {
                                recon_now[i+j*nx]=0.0;
                            }
                        }
                    }
                    
                    for(int i=0;i<nx*ny;i++)
                    {
                        stack_recon[k][i]+=recon_now[i];
                    }
//                        cout << "Done" << endl;
                        
                    delete [] recon_now;
                }
                cout << "\tDone" << endl;
                
            }
            else    // perform 3D-CTF correction
            {
                cout << "\tPerform 3D-CTF correction!" << endl;

                // write all corrected and weighted images in one mrc stack
                cout << "\tPerform 3D correction & save corrected stack:" << endl;
                
                float *stack_corrected[int(h_tilt_max/defocus_step)+1];   // 第一维遍历不同高度，第二维x，第三维y
                int n_zz=0;

                // 3D-CTF correction
                cout << "\tStart 3D-CTF correction..." << endl;
                // setup fftw3 plan for omp
                fftwf_plan plan_fft[threads],plan_ifft[threads];
                float *bufc[threads];
                for(int th=0;th<threads;th++)
                {
                    bufc[th]=new float[(nx+2-nx%2)*ny];
                    plan_fft[th]=fftwf_plan_dft_r2c_2d(ny,nx,(float*)bufc[th],reinterpret_cast<fftwf_complex*>(bufc[th]),FFTW_ESTIMATE);
                    plan_ifft[th]=fftwf_plan_dft_c2r_2d(ny,nx,reinterpret_cast<fftwf_complex*>(bufc[th]),(float*)bufc[th],FFTW_ESTIMATE);
                }
                
                #pragma omp parallel for num_threads(threads) reduction(+:n_zz)
                for(int zz=-int(h_tilt_max/2);zz<int(h_tilt_max/2);zz+=defocus_step)    // loop over every height (correct with different defocus)
                {
                    float *image_now_th=new float[nx*ny];
                    memcpy(image_now_th,image_now,sizeof(float)*nx*ny); // get the raw image!!!

                    // correction
                    int n_z=(zz+int(h_tilt_max/2))/defocus_step;
                    ctf_correction_omp(image_now_th,nx,ny,ctf_para[n],flip_contrast,float(zz)+float(defocus_step-1)/2,plan_fft[omp_get_thread_num()],plan_ifft[omp_get_thread_num()],bufc[omp_get_thread_num()]);
/*                        #pragma omp critical
                    {
                        ctf_correction(image_now_th,nx,ny,ctf_para[n],flip_contrast,float(zz)+float(defocus_step-1)/2);
                    }*/

                    // save
//                        corrected_mrc.write2DIm(image_now,n_zz);
                    stack_corrected[n_z]=new float[nx*ny];
                    memcpy(stack_corrected[n_z],image_now_th,sizeof(float)*nx*ny);
                    n_zz++;

                    delete [] image_now_th;
                }
                cout << "\tDone!" << endl;

                for(int th=0;th<threads;th++)
                {
                    fftwf_destroy_plan(plan_fft[th]);
                    fftwf_destroy_plan(plan_ifft[th]);
                    delete [] bufc[th];
                }

                // recontruction
                cout << "\tPerform reconstruction: " << endl;
                #pragma omp parallel for num_threads(threads)
                for(int k=0;k<h;k++)    // loop for every xy-plane
                {
//                        cout << "\t\tSlice " << k << ": ";
                    float *recon_now=new float[nx*ny];  // (x,y)
                    for(int i=0;i<nx;i++)
                    {
                        for(int j=0;j<ny;j++)
                        {
                            float x=i-x_orig_offset,y=j-y_orig_offset,z=k-z_orig_offset;
                            float x_rot=x*cos(-psi_rad)-y*sin(-psi_rad),y_rot=x*sin(-psi_rad)+y*cos(-psi_rad),z_rot=z;  // rotate tilt-axis to y-axis
                            float x_tlt=x_rot*cos(theta_rad)-z_rot*sin(theta_rad),y_tlt=y_rot,z_tlt=x_rot*sin(theta_rad)+z_rot*cos(theta_rad);  // tilt the specimen around y-axis
                            float x_final=x_tlt*cos(psi_rad)-y_tlt*sin(psi_rad)+x_orig_offset,y_final=x_tlt*sin(psi_rad)+y_tlt*cos(psi_rad)+y_orig_offset;  // rotate back to the origal image
                            int n_z=floor((z_tlt+int(h_tilt_max/2))/defocus_step);    // the num in the corrected stack for the current height
                            if(n_z>=0 && n_z<n_zz)
                            {
                                if(floor(x_final)>=0 && ceil(x_final)<nx && floor(y_final)>=0 && ceil(y_final)<ny)
                                {
                                    float coeff_x=x_final-floor(x_final),coeff_y=y_final-floor(y_final);
                                    recon_now[i+j*nx]=(1-coeff_x)*(1-coeff_y)*stack_corrected[n_z][int(floor(x_final)+floor(y_final)*nx)]+(coeff_x)*(1-coeff_y)*stack_corrected[n_z][int(ceil(x_final)+floor(y_final)*nx)]+(1-coeff_x)*(coeff_y)*stack_corrected[n_z][int(floor(x_final)+ceil(y_final)*nx)]+(coeff_x)*(coeff_y)*stack_corrected[n_z][int(ceil(x_final)+ceil(y_final)*nx)];
                                }
                                else
                                {
                                    recon_now[i+j*nx]=0.0;
                                }
                            }
                            else
                            {
                                recon_now[i+j*nx]=0.0;
                            }   
                        }
                    }
                    
                    for(int i=0;i<nx*ny;i++)
                    {
                        stack_recon[k][i]+=recon_now[i];
                    }
//                        cout << "Done" << endl;
                    
                    delete [] recon_now;
                }
                cout << "\tDone" << endl;

                for(int n_z=0;n_z<n_zz;n_z++)
                {
                    delete [] stack_corrected[n_z];
                }

            }
            delete [] image_now;
            delete [] image_now_backup;
        }

        // write out final result
        cout << "Write out final reconstruction result:" << endl;
        MRC stack_final(output_mrc.c_str(),"wb");
        stack_final.createMRC_empty(nx,ny,h,2); // (x,y,z)
        for(int k=0;k<h;k++)
        {
            stack_final.write2DIm(stack_recon[k],k);
//            delete [] stack_recon[k];
        }

        // update MRC header
        float min_thread[threads],max_thread[threads];
        double mean_thread[threads];
        for(int th=0;th<threads;th++)
        {
            min_thread[th]=stack_recon[0][0];
            max_thread[th]=stack_recon[0][0];
            mean_thread[th]=0.0;
        }
        #pragma omp parallel for num_threads(threads)
        for(int k=0;k<h;k++)
        {
            double mean_now=0.0;
            for(int i=0;i<nx*ny;i++)
            {
                mean_now+=stack_recon[k][i];
                if(min_thread[omp_get_thread_num()]>stack_recon[k][i])
                {
                    min_thread[omp_get_thread_num()]=stack_recon[k][i];
                }
                if(max_thread[omp_get_thread_num()]<stack_recon[k][i])
                {
                    max_thread[omp_get_thread_num()]=stack_recon[k][i];
                }
            }
            mean_thread[omp_get_thread_num()]+=(mean_now/(nx*ny));
        }
        float min_all=min_thread[0],max_all=max_thread[0];
        double mean_all=mean_thread[0];
        for(int th=1;th<threads;th++)
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

        for(int k=0;k<h;k++)
        {
            delete [] stack_recon[k];
        }

        stack_final.close();
        cout << "Done" << endl;

    }

    cout << endl << "Finish generating CTF model successfully!" << endl << endl;

    getTime(start_time);

    return 0;
}
