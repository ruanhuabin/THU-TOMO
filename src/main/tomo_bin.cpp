#include <stdio.h>
#include "util.h"
#include "mrc.h"
#include "time.h"
#include "math.h"
#include "fftw3.h"

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

void bin_image(float *image_orig,float *image_bin,int Nx,int Ny,int bin)
{
    int i,j,ii,jj;
    float tmp;
    int minus;
    for(i=0;i<Nx;i+=bin)
    {
        for(j=0;j<Ny;j+=bin)
        {
            tmp=0.0;
            minus=0;
            for(ii=0;ii<bin;ii++)
            {
                for(jj=0;jj<bin;jj++)
                {
                    if(i+ii>=0 && i+ii<Nx && j+jj>=0 && j+jj<Ny)
                    {
                        tmp+=image_orig[(i+ii)+(j+jj)*Nx];
                    }
                    else
                    {
                        minus++;
                    }
                }
            }
            tmp=tmp/(bin*bin-minus);
            image_bin[i/bin+j/bin*int(ceil(float(Nx)/float(bin)))]=tmp;
        }
    }
}

void bin_image_fft(float *image_orig,float *image_bin,int Nx,int Ny,int bin)
{
    int Nx_bin=int(ceil(float(Nx)/float(bin))),Ny_bin=int(ceil(float(Ny)/float(bin)));
    fftwf_plan plan_fft,plan_ifft;
    float *bufc=new float[(Nx+2-Nx%2)*Ny];
    float *bufc_bin=new float[(Nx_bin+2-Nx_bin%2)*Ny_bin];
    plan_fft=fftwf_plan_dft_r2c_2d(Ny,Nx,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);    // 读进来的image行是y列是x！！！（第一维表示y，第二维表示x）
    plan_ifft=fftwf_plan_dft_c2r_2d(Ny_bin,Nx_bin,reinterpret_cast<fftwf_complex*>(bufc_bin),(float*)bufc_bin,FFTW_ESTIMATE);
    buf2fft(image_orig,bufc,Nx,Ny);
    fftwf_execute(plan_fft);
    
    for(int j=0;j<=int(floor((Ny_bin-1)/2));j++)
    {
        for(int i=0;i<(Nx_bin+2-Nx_bin%2);i+=2)
        {
            bufc_bin[j*(Nx_bin+2-Nx_bin%2)+i]=bufc[j*(Nx+2-Nx%2)+i];
            bufc_bin[j*(Nx_bin+2-Nx_bin%2)+i+1]=bufc[j*(Nx+2-Nx%2)+i+1];
        }
    }
    if(Ny_bin%2==0)
    {
        int j=Ny_bin/2;
        for(int i=0;i<(Nx_bin+2-Nx_bin%2);i+=2)
        {
            bufc_bin[j*(Nx_bin+2-Nx_bin%2)+i]=bufc[j*(Nx+2-Nx%2)+i];
            bufc_bin[j*(Nx_bin+2-Nx_bin%2)+i+1]=bufc[j*(Nx+2-Nx%2)+i+1];
        }
    }
    for(int j=Ny-1;j>=Ny-int(floor((Ny_bin-1)/2));j--)
    {
        for(int i=0;i<(Nx_bin+2-Nx_bin%2);i+=2)
        {
            bufc_bin[(Ny_bin-1-(Ny-1-j))*(Nx_bin+2-Nx_bin%2)+i]=bufc[j*(Nx+2-Nx%2)+i];
            bufc_bin[(Ny_bin-1-(Ny-1-j))*(Nx_bin+2-Nx_bin%2)+i+1]=bufc[j*(Nx+2-Nx%2)+i+1];
        }
    }

    fftwf_execute(plan_ifft);
    fft2buf(image_bin,bufc_bin,Nx_bin,Ny_bin);
    for(int i=0;i<Nx_bin*Ny_bin;i++)    // normalization
    {
        image_bin[i]/=(Nx_bin*Ny_bin);
    }

    fftwf_destroy_plan(plan_fft);
	fftwf_destroy_plan(plan_ifft);
    delete [] bufc;
    delete [] bufc_bin;
}

int main(int argc, char **argv)
{
    time_t start_time;
    time(&start_time);

    map<string, string> inputPara;
    map<string, string> outputPara;
    const char *paraFileName = "../conf/para_bin.conf";
    readParaFile(inputPara, paraFileName);
    getAllParas(inputPara);

    // read input parameters
    map<string,string>::iterator it=inputPara.find("input_mrc");
    string input_mrc;
    if(it!=inputPara.end())
    {
        input_mrc=it->second;
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
        output_mrc=it->second;
        cout << "Output file name: " << output_mrc << endl;
    }
    else
    {
        cerr << "No output file name, set default: tomo_bin.st" << endl;
        output_mrc="tomo_bin.st";
    }

    int bin;
    it=inputPara.find("bin");
    if(it!=inputPara.end())
    {
        bin=atoi(it->second.c_str());
        cout << "Bin: " << bin << endl;
    }
    else
    {
        cerr << "No bin parameter!" << endl;
        abort();
    }

    // bin image
    cout << endl << "Performing binning:" << endl;
    if(bin==1)
    {
        cout << "bin 1, skip binning!" << endl;
    }
    else
    {
        cout << "bin " << bin << ":" << endl;
        MRC stack_bin(output_mrc.c_str(),"wb");
        float *image_now=new float[stack_orig.getNx()*stack_orig.getNy()];
        float *image_bin=new float[int(ceil(float(stack_orig.getNx())/float(bin)))*int(ceil(float(stack_orig.getNy())/float(bin)))];
        stack_bin.createMRC_empty(int(ceil(float(stack_orig.getNx())/float(bin))),int(ceil(float(stack_orig.getNy())/float(bin))),stack_orig.getNz(),2);
        for(int n=0;n<stack_orig.getNz();n++)
        {
            cout << n << ": ";
            stack_orig.read2DIm_32bit(image_now,n);
            bin_image_fft(image_now,image_bin,stack_orig.getNx(),stack_orig.getNy(),bin);
            stack_bin.write2DIm(image_bin,n);
            cout << "Done" << endl;
        }
        delete [] image_now;
        delete [] image_bin;
        stack_bin.close();
    }
    cout << "Finish binning!" << endl;

    stack_orig.close();

    getTime(start_time);

    return 0;
}
