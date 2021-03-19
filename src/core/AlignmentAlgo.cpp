/*******************************************************************
 *       Filename:  AlignmentAlgo.cpp                                     
 *                                                                 
 *    Description:                                        
 *                                                                 
 *        Version:  1.0                                            
 *        Created:  07/07/2020 04:33:40 PM                                 
 *       Revision:  none                                           
 *       Compiler:  gcc                                           
 *                                                                 
 *         Author:  Ruan Huabin                                      
 *          Email:  ruanhuabin@tsinghua.edu.cn                                        
 *        Company:  Dep. of CS, Tsinghua Unversity                                      
 *                                                                 
 *******************************************************************/
#include "AlignmentAlgo.h"
#include "mrc.h"
#include "stdio.h"
#include "math.h"
#include "fftw3.h"
#include "Dense"
#include "nr.h"
#include "omp.h"
#include "alignment.h"
#include "low_pass.h"
#include "bin.h"


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

static void bin_image(float *image_orig,float *image_bin,int Nx,int Ny,int bin)
{
    int i,j,ii,jj;
    float tmp;
    int minus;
    // loop: Nx*Ny (whole image)
    for(j=0;j<Ny;j+=bin)
    {
        for(i=0;i<Nx;i+=bin)
        {
            tmp=0.0;
            minus=0;
            // loop: bin*bin (bin*bin patch)
            for(jj=0;jj<bin;jj++)
            {
                for(ii=0;ii<bin;ii++)
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

static void bin_image_fft(float *image_orig,float *image_bin,int Nx,int Ny,int bin)
{
    int Nx_bin=int(ceil(float(Nx)/float(bin)));
    int Ny_bin=int(ceil(float(Ny)/float(bin)));
    fftwf_plan plan_fft,plan_ifft;
    float *bufc=new float[(Nx+2-Nx%2)*Ny];
    float *bufc_bin=new float[(Nx_bin+2-Nx_bin%2)*Ny_bin];
    plan_fft=fftwf_plan_dft_r2c_2d(Ny,Nx,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);    // 读进来的image行是y列是x！！！（第一维表示y，第二维表示x）
    plan_ifft=fftwf_plan_dft_c2r_2d(Ny_bin,Nx_bin,reinterpret_cast<fftwf_complex*>(bufc_bin),(float*)bufc_bin,FFTW_ESTIMATE);
    buf2fft(image_orig,bufc,Nx,Ny);
    fftwf_execute(plan_fft);
    
    // loop: (Nx_bin+2-Nx_bin%2)*floor((Ny_bin-1)/2) (left half of low freq component)
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
    // loop: (Nx_bin+2-Nx_bin%2)*floor((Ny_bin-1)/2) (right half of low freq component)
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

static void rotate_image(float *image_orig,float *image_rot,int Nx,int Ny,float Nx_orig_offset,float Ny_orig_offset,float psi_rad)   // counter-clockwise
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
            x=float(i)+Nx_orig_offset;
            y=float(j)+Ny_orig_offset;
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

static void move_image(float *image_orig,float *image_move,int Nx,int Ny,int dx,int dy,bool padding_avg)
{
    float image_avg=0;
    int i,j;

    // loop: Nx*Ny (whole image)
    for(j=0;j<Ny;j++)
    {
        for(i=0;i<Nx;i++)   // calculate average
        {
            image_avg+=image_orig[i+j*Nx];
        }
    }
    image_avg=image_avg/(Nx*Ny);
    
    // loop: Nx*Ny (whole image)
    for(j=0;j<Ny;j++)
    {
        for(i=0;i<Nx;i++)
        {
            if(i-dx>=0 && i-dx<Nx && j-dy>=0 && j-dy<Ny)
            {
                image_move[i+j*Nx]=image_orig[(i-dx)+(j-dy)*Nx];
            }
            else
            {
                if(padding_avg)
                {
                    image_move[i+j*Nx]=image_avg;
                }
                else
                {
                    image_move[i+j*Nx]=0;
                }
            }
        }
    }
}

static void move_image_fft(float *image_orig,float *image_move,int Nx,int Ny,float dx,float dy,bool clip)   // 右移dx
{
    int n,i,j;
    fftwf_plan plan_fft,plan_ifft;
    float *bufc=new float[(Nx+2-Nx%2)*Ny];
    plan_fft=fftwf_plan_dft_r2c_2d(Ny,Nx,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);    // 读进来的image行是y列是x！！！（第一维表示y，第二维表示x）
    plan_ifft=fftwf_plan_dft_c2r_2d(Ny,Nx,reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);
    buf2fft(image_orig,bufc,Nx,Ny);
//    float *bufc_backup=new float[(Nx+2-Nx%2)*Ny];
//    memcpy(bufc_backup,bufc,sizeof(float)*(Nx+2-Nx%2)*Ny);
    fftwf_execute(plan_fft);

    // loop: Nx*Ny (whole image)
    for(j=0;j<Ny;j++)
    {
        for(i=0;i<Nx+2-Nx%2;i+=2)
        {
            float r1=bufc[i+j*(Nx+2-Nx%2)],i1=bufc[i+1+j*(Nx+2-Nx%2)];
            float r2=cos(2*M_PI*dx*float(i/2)/float(Nx)),i2=-sin(2*M_PI*dx*float(i/2)/float(Nx));
            float r3=cos(2*M_PI*dy*float(j)/float(Ny)),i3=-sin(2*M_PI*dy*float(j)/float(Ny));
            if(j>=ceil((Ny+1)/2))    // move to [-pi,pi]
            {
                r3=cos(2*M_PI*dy*float(j-Ny)/float(Ny));
                i3=-sin(2*M_PI*dy*float(j-Ny)/float(Ny));
            }
            float r4=r1*r2-i1*i2,i4=r1*i2+r2*i1;
            float r5=r3*r4-i3*i4,i5=r3*i4+r4*i3;
            bufc[i+j*(Nx+2-Nx%2)]=r5;
            bufc[i+1+j*(Nx+2-Nx%2)]=i5;
        }
    }
    fftwf_execute(plan_ifft);
    fft2buf(image_move,bufc,Nx,Ny);
    for(j=0;j<Ny;j++)
    {
        for(i=0;i<Nx;i++)   // normalization
        {
            image_move[i+j*Nx]=image_move[i+j*Nx]/(Nx*Ny);
        }
    }
    if(clip)
    {
        for(j=0;j<Ny;j++)
        {
            for(i=0;i<Nx;i++)
            {
                if(!(i-dx>=0 && i-dx<Nx && j-dy>=0 && j-dy<Ny))
                {
                    image_move[i+j*Nx]=0.0;
                }
            }
        }
    }
    fftwf_destroy_plan(plan_fft);
	fftwf_destroy_plan(plan_ifft);
    delete [] bufc;
}

static void move_image_fft_omp(float *image_orig,float *image_move,int Nx,int Ny,float dx,float dy,bool clip,fftwf_plan plan_fft,fftwf_plan plan_ifft,float *bufc)   // 右移dx
{
    int n,i,j;
    buf2fft(image_orig,bufc,Nx,Ny);
//    float *bufc_backup=new float[(Nx+2-Nx%2)*Ny];
//    memcpy(bufc_backup,bufc,sizeof(float)*(Nx+2-Nx%2)*Ny);
    fftwf_execute(plan_fft);

    // loop: Nx*Ny (whole image)
    for(j=0;j<Ny;j++)
    {
        for(i=0;i<Nx+2-Nx%2;i+=2)
        {
            float r1=bufc[i+j*(Nx+2-Nx%2)],i1=bufc[i+1+j*(Nx+2-Nx%2)];
            float r2=cos(2*M_PI*dx*float(i/2)/float(Nx)),i2=-sin(2*M_PI*dx*float(i/2)/float(Nx));
            float r3=cos(2*M_PI*dy*float(j)/float(Ny)),i3=-sin(2*M_PI*dy*float(j)/float(Ny));
            if(j>=ceil((Ny+1)/2))    // move to [-pi,pi]
            {
                r3=cos(2*M_PI*dy*float(j-Ny)/float(Ny));
                i3=-sin(2*M_PI*dy*float(j-Ny)/float(Ny));
            }
            float r4=r1*r2-i1*i2,i4=r1*i2+r2*i1;
            float r5=r3*r4-i3*i4,i5=r3*i4+r4*i3;
            bufc[i+j*(Nx+2-Nx%2)]=r5;
            bufc[i+1+j*(Nx+2-Nx%2)]=i5;
        }
    }
    fftwf_execute(plan_ifft);
    fft2buf(image_move,bufc,Nx,Ny);
    for(j=0;j<Ny;j++)
    {
        for(i=0;i<Nx;i++)   // normalization
        {
            image_move[i+j*Nx]=image_move[i+j*Nx]/(Nx*Ny);
        }
    }
    if(clip)
    {
        for(j=0;j<Ny;j++)
        {
            for(i=0;i<Nx;i++)
            {
                if(!(i-dx>=0 && i-dx<Nx && j-dy>=0 && j-dy<Ny))
                {
                    image_move[i+j*Nx]=0.0;
                }
            }
        }
    }
}

static void transform_image(float *image_orig,float *image_final,int Nx,int Ny,bool is_rot,float Nx_orig_offset,float Ny_orig_offset,float psi_rad,bool is_move,float dx,float dy)
{
    float *image_move=new float[Nx*Ny];
    move_image_fft(image_orig,image_move,Nx,Ny,-dx,-dy,true);

    // rotate tilt axis to y-axis
    if(is_rot)
    {
        rotate_image(image_move,image_final,Nx,Ny,Nx_orig_offset,Ny_orig_offset,psi_rad);
    }
    else
    {
        memcpy(image_final,image_move,sizeof(float)*Nx*Ny);
    }

    delete [] image_move;
}

static void transform_image_omp(float *image_orig,float *image_final,int Nx,int Ny,bool is_rot,float Nx_orig_offset,float Ny_orig_offset,float psi_rad,bool is_move,float dx,float dy,fftwf_plan plan_fft,fftwf_plan plan_ifft,float *bufc)
{
    float *image_move=new float[Nx*Ny];
    move_image_fft_omp(image_orig,image_move,Nx,Ny,-dx,-dy,true,plan_fft,plan_ifft,bufc);

    // rotate tilt axis to y-axis
    if(is_rot)
    {
        rotate_image(image_move,image_final,Nx,Ny,Nx_orig_offset,Ny_orig_offset,psi_rad);
    }
    else
    {
        memcpy(image_final,image_move,sizeof(float)*Nx*Ny);
    }

    delete [] image_move;
}

static void transform_image_BI(float *image_orig,float *image_final,int Nx,int Ny,float Nx_orig_offset,float Ny_orig_offset,float psi_rad,float dx,float dy,float *c_x,float *c_y,bool is_rot,int bin)
{
    float *image_BI=new float[Nx*Ny];
    for(int i=0;i<Nx*Ny;i++)
    {
        image_BI[i]=0.0;
    }

    // move for beam-induced motion
    // loop: Nx*Ny (whole image)
    for(int j=0;j<Ny;j++)
    {
        for(int i=0;i<Nx;i++)
        {
//            float BI_x_rot,BI_y_rot;
            float BI_x,BI_y;
            float x_now=(float(i)+Nx_orig_offset)/float(bin);
            float y_now=(float(j)+Ny_orig_offset)/float(bin);
            BI_x=c_x[0]+c_x[1]*x_now+c_x[2]*x_now*x_now+c_x[3]*y_now+c_x[4]*y_now*y_now+c_x[5]*x_now*y_now+dx;
            BI_y=c_y[0]+c_y[1]*x_now+c_y[2]*x_now*x_now+c_y[3]*y_now+c_y[4]*y_now*y_now+c_y[5]*x_now*y_now+dy;
            BI_x=BI_x*float(bin);
            BI_y=BI_y*float(bin);
            if(floor(i-BI_x)>=0 && ceil(i-BI_x)<Nx && floor(j-BI_y)>=0 && ceil(j-BI_y)<Ny)
            {
                float coeff_x=(i-BI_x)-floor(i-BI_x);
                float coeff_y=(j-BI_y)-floor(j-BI_y);
                image_BI[int(floor(i-BI_x))+int(floor(j-BI_y))*Nx]+=(1-coeff_x)*(1-coeff_y)*image_orig[i+j*Nx];
                image_BI[int(floor(i-BI_x))+int(ceil(j-BI_y))*Nx]+=(1-coeff_x)*(coeff_y)*image_orig[i+j*Nx];
                image_BI[int(ceil(i-BI_x))+int(floor(j-BI_y))*Nx]+=(coeff_x)*(1-coeff_y)*image_orig[i+j*Nx];
                image_BI[int(ceil(i-BI_x))+int(ceil(j-BI_y))*Nx]+=(coeff_x)*(coeff_y)*image_orig[i+j*Nx];
            }
        }
    }

    // rotate tilt axis to y-axis
    if(is_rot)
    {
        rotate_image(image_BI,image_final,Nx,Ny,Nx_orig_offset,Ny_orig_offset,psi_rad);
    }
    else
    {
        memcpy(image_final,image_BI,sizeof(float)*Nx*Ny);
    }
    delete [] image_BI;
}

static void normalize_image(float *image,int Nx,int Ny) // normailze to 0~1
{
    int i;
    float image_min,image_max;
    image_min=image[0];
    // loop: Nx*Ny (whole image)
    for(i=1;i<Nx*Ny;i++)
    {
        if(image[i]<image_min)
        {
            image_min=image[i];
        }
    }
    // loop: Nx*Ny (whole image)
    for(i=0;i<Nx*Ny;i++)
    {
        image[i]-=image_min;
    }
    image_max=image[0];
    // loop: Nx*Ny (whole image)
    for(i=1;i<Nx*Ny;i++)
    {
        if(image[i]>image_max)
        {
            image_max=image[i];
        }
    }
    // loop: Nx*Ny (whole image)
    for(i=0;i<Nx*Ny;i++)
    {
        image[i]/=image_max;
    }
}

static void standardize_image(float *image,int Nx,int Ny) // standardize to 0-mean & 1-std
{
    double sum=0.0;
    double sum2=0.0;
    // loop: Nx*Ny (whole image)
    for(int i=0;i<Nx*Ny;i++)
    {
        sum+=double(image[i]);
        sum2+=(double(image[i])*double(image[i]));
    }
    double mean=sum/(Nx*Ny);
    double mean2=sum2/(Nx*Ny);
    double std=sqrt(mean2-mean*mean);
    // loop: Nx*Ny (whole image)
    for(int i=0;i<Nx*Ny;i++)
    {
        image[i]=float((double(image[i])-mean)/std);
    }
}

static int tracks_size(vector<Track> &tracks)
{
    int size=0;
    // loop: num of tracks
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        if(it->getStatus()==true)
        {
            size++;
        }
    }
    return size;
}

DP loss(Vec_I_DP &x,vector<Track> &tracks,float *d_x,float *d_y,float psi)
{
    DP loss_sum=0.0;
    Eigen::Matrix3f R_psi,R_theta;  // theta: tilt angle; psi: axis rotation
    Eigen::Matrix<float,2,3> P; // P: projection matrix {{1,0,0};{0,1,0}}
    Eigen::Matrix<float,2,3> A;
    Eigen::Vector3f r;
    Eigen::Vector2f p,d,l;
    R_psi.setZero();
    R_theta.setZero();
    P.setZero();
    P(0,0)=1;
    P(1,1)=1;
    // loop: num of tracks
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        if(it->getStatus()==true)
        {
            float r_x=it->getMarker3D_x(),r_y=it->getMarker3D_y(),r_z=it->getMarker3D_z();
            // loop: length of track
            for(int t=0;t<it->getLength();t++)
            {
                float p_x=it->getMarker2D_x(t);
                float p_y=it->getMarker2D_y(t);
                int n=it->getMarker2D_n(t);
                R_psi.setZero();
                R_theta.setZero();
                R_psi << cos(-psi),-sin(-psi),0,sin(-psi),cos(-psi),0,0,0,1;
                R_theta << cos(x[n]),0,-sin(x[n]),0,1,0,sin(x[n]),0,cos(x[n]);
                A=P*R_theta*R_psi;
                p << p_x,p_y;
                r << r_x,r_y,r_z;
                d << d_x[n],d_y[n];
                l=p-(A*r+d);
                loss_sum=loss_sum+l(0)*l(0)+l(1)*l(1);
            }
        }
    }
    return loss_sum;
}

DP loss_single(DP x,vector<Track> &tracks,float *d_x,float *d_y,float psi,int ref,float *c_x,float *c_y)
{
    DP loss_sum=0.0;
    Eigen::Matrix3f R_psi,R_theta;  // theta: tilt angle; psi: axis rotation
    Eigen::Matrix<float,2,3> P; // P: projection matrix {{1,0,0};{0,1,0}}
    Eigen::Matrix<float,2,3> A;
    Eigen::Vector3f r;
    Eigen::Vector2f p,d,l,s;
    R_psi.setZero();
    R_theta.setZero();
    P.setZero();
    P(0,0)=1;
    P(1,1)=1;
    // loop: num of tracks
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        if(it->getStatus()==true)
        {
            float r_x=it->getMarker3D_x();
            float r_y=it->getMarker3D_y();
            float r_z=it->getMarker3D_z();
            // loop: length of track
            for(int t=0;t<it->getLength();t++)
            {
                float p_x=it->getMarker2D_x(t);
                float p_y=it->getMarker2D_y(t);
                int n=it->getMarker2D_n(t);
                if(n==ref)
                {
                    R_psi.setZero();
                    R_theta.setZero();
                    R_psi << cos(-psi),-sin(-psi),0,sin(-psi),cos(-psi),0,0,0,1;
                    R_theta << cos(x),0,-sin(x),0,1,0,sin(x),0,cos(x);
                    float s_x=0.0,s_y=0.0;
                    s_x=c_x[0]+c_x[1]*p_x+c_x[2]*p_x*p_x+c_x[3]*p_y+c_x[4]*p_y*p_y+c_x[5]*p_x*p_y;
                    s_y=c_y[0]+c_y[1]*p_x+c_y[2]*p_x*p_x+c_y[3]*p_y+c_y[4]*p_y*p_y+c_y[5]*p_x*p_y;
                    A=P*R_psi.transpose()*R_theta*R_psi;
                    p << p_x,p_y;
                    r << r_x,r_y,r_z;
                    d << d_x[n],d_y[n];
                    s << s_x,s_y;
                    l=p-(A*r+d+s);
                    loss_sum=loss_sum+l(0)*l(0)+l(1)*l(1);
                }
            }
        }
    }
    return loss_sum;
}



AlignmentAlgo::~AlignmentAlgo()
{
}

void AlignmentAlgo::doAlignment(map<string, string> &inputPara, map<string, string> &outputPara)
{
    cout<<"Run doAlignment() in AlignmentAlgo"<<endl;

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

    it=inputPara.find("original_mrc");
    string original_mrc;
    if(it!=inputPara.end())
    {
        original_mrc=path+"/"+it->second;
        cout << "Original file name: " << original_mrc << endl;
    }
    else
    {
        cout << "No original file name, set default (input file name): " << input_mrc << endl;
        original_mrc=input_mrc;
    }

    bool skip_lowpass,skip_bin,skip_coarse,skip_patch,skip_fine,unbinned_stack;
    it=inputPara.find("skip_lowpass");
    if(it!=inputPara.end())
    {
        skip_lowpass=atoi(it->second.c_str());
        cout << "Skip lowpass: " << skip_lowpass << endl;
    }
    else
    {
        cout << "No skip_lowpass, set default: 0" << endl;
        skip_lowpass=0;
    }
    it=inputPara.find("skip_bin");
    if(it!=inputPara.end())
    {
        skip_bin=atoi(it->second.c_str());
        cout << "Skip bin: " << skip_bin << endl;
    }
    else
    {
        cout << "No skip_bin, set default: 0" << endl;
        skip_bin=0;
    }
    it=inputPara.find("skip_coarse");
    if(it!=inputPara.end())
    {
        skip_coarse=atoi(it->second.c_str());
        cout << "Skip coarse alignment: " << skip_coarse << endl;
    }
    else
    {
        cout << "No skip_coarse, set default: 0" << endl;
        skip_coarse=0;
    }
    it=inputPara.find("skip_patch");
    if(it!=inputPara.end())
    {
        skip_patch=atoi(it->second.c_str());
        cout << "Skip patch tracking: " << skip_patch << endl;
    }
    else
    {
        cout << "No skip_patch, set default: 0" << endl;
        skip_patch=0;
    }
    it=inputPara.find("skip_fine");
    if(it!=inputPara.end())
    {
        skip_fine=atoi(it->second.c_str());
        cout << "Skip fine alignment: " << skip_fine << endl;
    }
    else
    {
        cout << "No skip_fine, set default: 0" << endl;
        skip_fine=0;
    }
    it=inputPara.find("unbinned_stack");
    if(it!=inputPara.end())
    {
        unbinned_stack=atoi(it->second.c_str());
        cout << "Write out unbinned stack: " << unbinned_stack << endl;
    }
    else
    {
        cout << "No unbinned_stack, set default: 1" << endl;
        unbinned_stack=1;
    }

    float pix;
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

    // Preprocessing (bin + low pass filter)
    int n,k,i,j;
    float *image_now=new float[stack_orig.getNx()*stack_orig.getNy()];

    #ifdef GPU_VERSION
    cout << "Start GPU version!" << endl;
    #else
    cout << "Start CPU version!" << endl;
    #endif
    // low pass filter
    if(!skip_lowpass)
    {
        cout << endl << "Performing lowpass filtering:" << endl;
        fftwf_plan plan_fft,plan_ifft;
        string filtered_mrc=prfx+".flt";
        MRC stack_lp(filtered_mrc.c_str(),"wb+");
        stack_lp.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);
        it=inputPara.find("lp");
        float lp;
        if(it!=inputPara.end())
        {
            lp=atof(it->second.c_str());
            cout << "Low pass: " << lp << endl;
        }
        else
        {
            lp=0.1;
            cout << "No low pass parameter, set default: 0.1" << endl;
        }
        #ifdef GPU_VERSION
        low_pass(&stack_orig, &stack_lp, lp);
        #else
        float *bufc=new float[(stack_orig.getNx()+2)*stack_orig.getNy()];
        plan_fft=fftwf_plan_dft_r2c_2d(stack_orig.getNy(),stack_orig.getNx(),(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);    // 读进来的image行是y列是x！！！（第一维表示y，第二维表示x）
        plan_ifft=fftwf_plan_dft_c2r_2d(stack_orig.getNy(),stack_orig.getNx(),reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);
        cout << "lowpass filter:" << endl;
        // loop: Nz (number of images)
        for(n=0;n<stack_orig.getNz();n++)
        {
            cout << n << ": ";
            stack_orig.read2DIm_32bit(image_now,n);
            buf2fft(image_now,bufc,stack_orig.getNx(),stack_orig.getNy());
            fftwf_execute(plan_fft);
            float fft_x[stack_orig.getNx()+2],fft_y[stack_orig.getNy()];
            for(i=0;i<stack_orig.getNx()+2;i++)
            {
                fft_x[i]=2*M_PI/(stack_orig.getNx())*i;
                if(fft_x[i]>M_PI)
                {
                    fft_x[i]=fft_x[i]-2*M_PI;
                }
            }
            for(j=0;j<stack_orig.getNy();j++)
            {
                fft_y[j]=2*M_PI/stack_orig.getNy()*j;
                if(fft_y[j]>M_PI)
                {
                    fft_y[j]=fft_y[j]-2*M_PI;
                }
            }
            // loop: Nx*Ny (whole image)
            for(j=0;j<stack_orig.getNy();j++)
            {
                for(i=0;i<stack_orig.getNx()+2;i+=2)
                {
                    if((fft_x[i/2]*fft_x[i/2]+fft_y[j]*fft_y[j])>(lp*M_PI)*(lp*M_PI))
                    {
                        if((fft_x[i/2]*fft_x[i/2]+fft_y[j]*fft_y[j])>(1.2*lp*M_PI)*(1.2*lp*M_PI))
                        {
                            bufc[i+j*(stack_orig.getNx()+2)]=0;
                            bufc[i+1+j*(stack_orig.getNx()+2)]=0;
                        }
                        else    // smooth with cosine
                        {
                            bufc[i+j*(stack_orig.getNx()+2)]*=((1+cos((sqrt(fft_x[i/2]*fft_x[i/2]+fft_y[j]*fft_y[j])-sqrt((lp*M_PI)*(lp*M_PI)))/sqrt((0.2*lp)*(0.2*lp))))/2.0);
                            bufc[i+1+j*(stack_orig.getNx()+2)]*=((1+cos((sqrt(fft_x[i/2]*fft_x[i/2]+fft_y[j]*fft_y[j])-sqrt((lp*M_PI)*(lp*M_PI)))/sqrt((0.2*lp)*(0.2*lp))))/2.0);
                        }
                    }
                }
            }
            fftwf_execute(plan_ifft);
            fft2buf(image_now,bufc,stack_orig.getNx(),stack_orig.getNy());
            // loop: Nx*Ny (whole image)
            for(i=0;i<stack_orig.getNx()*stack_orig.getNy();i++)
            {
                image_now[i]=image_now[i]/(stack_orig.getNx()*stack_orig.getNy());  // normalization
            }
            stack_lp.write2DIm(image_now,n);
            cout << "Done" << endl;
        }
        fftwf_destroy_plan(plan_fft);
        fftwf_destroy_plan(plan_ifft);
        delete [] bufc;
        #endif
        stack_lp.computeHeader_omp(pix,true,threads);
        stack_lp.close();
        stack_orig.close();
        stack_orig.open(filtered_mrc.c_str(),"rb");
        cout << "Finish lowpass filtering!" << endl;
    }
    else
    {
        cout << "Skip lowpass filtering!" << endl;
    }

    // bin
    int bin;
    it=inputPara.find("bin");
    if(it!=inputPara.end())
    {
        bin=atoi(it->second.c_str());
        cout << "Bin: " << bin << endl;
    }
    else
    {
        bin=1;
        cout << "No bin, set default: 1" << endl;
    }
    if(!skip_bin)
    {
        cout << endl << "Performing binning:" << endl;
        if(bin!=1)
        {
            string bin_str=to_string(bin);
            string bin_mrc=prfx+".bin"+bin_str;
            MRC stack_bin(bin_mrc.c_str(),"wb+");
            stack_bin.createMRC_empty(int(ceil(float(stack_orig.getNx())/float(bin))),int(ceil(float(stack_orig.getNy())/float(bin))),stack_orig.getNz(),2);
            #ifdef GPU_VERSION
            bin_image(&stack_orig, &stack_bin, bin);
            #else
            float *image_bin=new float[int(ceil(float(stack_orig.getNx())/float(bin)))*int(ceil(float(stack_orig.getNy())/float(bin)))];
            cout << "bin: " << endl;
            // loop: Nz (number of images)
            for(n=0;n<stack_orig.getNz();n++)
            {
                cout << n << ": ";
                stack_orig.read2DIm_32bit(image_now,n);
                bin_image_fft(image_now,image_bin,stack_orig.getNx(),stack_orig.getNy(),bin);
                stack_bin.write2DIm(image_bin,n);
                cout << "Done" << endl;
            }
            delete [] image_bin;
            #endif
            stack_bin.computeHeader_omp(pix*float(bin),true,threads);
            stack_bin.close();
            stack_orig.close();
            stack_orig.open(bin_mrc.c_str(),"rb");
        }
        cout << "Finish binning!" << endl;
    }
    else
    {
        cout << "Skip binning!" << endl;
    }



    // Coarse align

    // Read tilt angles
    it=inputPara.find("input_rawtlt");
    string input_rawtlt;
    if(it!=inputPara.end())
    {
        input_rawtlt=path+"/"+it->second;
        cout << "Input rawtlt file name: " << input_rawtlt << endl;
    }
    else
    {
        cerr << "No input rawtlt file name!" << endl;
        abort();
    }
    FILE *rawtlt=fopen(input_rawtlt.c_str(),"r");
    if(rawtlt==NULL)
    {
        cerr << "Cannot open rawtlt file!" << endl;
        abort();
    }
    float theta[stack_orig.getNz()];
    float min_tlt=90.0;
    int min_tlt_n=0;
    for(n=0;n<stack_orig.getNz();n++)
    {
        fscanf(rawtlt,"%f",&theta[n]);
//        cout << theta[n] << endl;
        if(fabs(theta[n])<min_tlt)
        {
            min_tlt=fabs(theta[n]);
            min_tlt_n=n;
        }
    }
//    cout << min_tlt_n << " " << min_tlt << endl;

    // Read patch parameters
    int patch_size,patch_size_half,patch_Nx,patch_Ny,patch_trans;   // for patch tracking
    it=inputPara.find("patch_size_tracking");
    if(it!=inputPara.end())
    {
        patch_size=atoi(it->second.c_str());
        if(patch_size%2==0)
        {
            patch_size--;
        }
        patch_size_half=(patch_size-1)/2;
        cout << "Patch size for patch tracking: " << patch_size << endl;
    }
    else
    {
        cout << "No patch size for patch tracking, set default: 255" << endl;
        patch_size=255;
        patch_size_half=(patch_size-1)/2;
    }
    it=inputPara.find("patch_Nx_tracking");
    if(it!=inputPara.end())
    {
        patch_Nx=atoi(it->second.c_str());
        cout << "Patch Nx for patch tracking: " << patch_Nx << endl;
    }
    else
    {
        cout << "No patch Nx for patch tracking, set default: 6" << endl;
        patch_Nx=6;
    }
    it=inputPara.find("patch_Ny_tracking");
    if(it!=inputPara.end())
    {
        patch_Ny=atoi(it->second.c_str());
        cout << "Patch Ny for patch tracking: " << patch_Ny << endl;
    }
    else
    {
        cout << "No patch Ny for patch tracking, set default: 6" << endl;
        patch_Ny=6;
    }
    it=inputPara.find("patch_search_range_tracking");
    if(it!=inputPara.end())
    {
        patch_trans=atoi(it->second.c_str());
        cout << "Patch search range for patch tracking: " << patch_trans << endl;
    }
    else
    {
        cout << "No patch search range for patch tracking, set default: 16" << endl;
        patch_trans=16;
    }
    
    int patch_size_coarse,patch_size_coarse_half,patch_Nx_coarse,patch_Ny_coarse,patch_trans_coarse;   // for coarse alignment
    it=inputPara.find("patch_size_coarse");
    if(it!=inputPara.end())
    {
        patch_size_coarse=atoi(it->second.c_str());
        if(patch_size_coarse%2==0)
        {
            patch_size_coarse--;
        }
        patch_size_coarse_half=(patch_size_coarse-1)/2;
        cout << "Patch size for coarse alignment: " << patch_size_coarse << endl;
    }
    else
    {
        cout << "No patch size for coarse alignment, set default: 255" << endl;
        patch_size_coarse=255;
        patch_size_coarse_half=(patch_size_coarse-1)/2;
    }
    it=inputPara.find("patch_Nx_coarse");
    if(it!=inputPara.end())
    {
        patch_Nx_coarse=atoi(it->second.c_str());
        cout << "Patch Nx for coarse alignment: " << patch_Nx_coarse << endl;
    }
    else
    {
        cout << "No patch Nx for coarse alignment, set default: 6" << endl;
        patch_Nx_coarse=6;
    }
    it=inputPara.find("patch_Ny_coarse");
    if(it!=inputPara.end())
    {
        patch_Ny_coarse=atoi(it->second.c_str());
        cout << "Patch Ny for coarse alignment: " << patch_Ny_coarse << endl;
    }
    else
    {
        cout << "No patch Ny for coarse alignment, set default: 6" << endl;
        patch_Ny_coarse=6;
    }
    it=inputPara.find("patch_search_range_coarse");
    if(it!=inputPara.end())
    {
        patch_trans_coarse=atoi(it->second.c_str());
        cout << "Patch search range for coarse alignment: " << patch_trans_coarse << endl;
    }
    else
    {
        cout << "No patch search range for coarse alignment, set default: 32" << endl;
        patch_trans_coarse=32;
    }

    // Initial model (divide the reference image(0 degree) equally)
    int ref_n=min_tlt_n;
    float x,y;
    int t=0;
    // for coarse alignment
    #ifdef GPU_VERSION
    #else
    int patch_x_ref_coarse[patch_Nx_coarse*patch_Ny_coarse];
    int patch_y_ref_coarse[patch_Nx_coarse*patch_Ny_coarse];  // 存放特征点坐标
    int patch_deltaX_coarse[patch_Nx_coarse*patch_Ny_coarse];
    int patch_deltaY_coarse[patch_Nx_coarse*patch_Ny_coarse];   // 对应特征点坐标的变化量（平移量）
    float patch_dx_coarse=float(stack_orig.getNx()-1)/float(patch_Nx_coarse+1);
    float patch_dy_coarse=float(stack_orig.getNy()-1)/float(patch_Ny_coarse+1);    // 每个patch中心点之间的距离
    float patch_dx_orig_coarse=patch_dx_coarse;
    float patch_dy_orig_coarse=patch_dy_coarse;
    float patch_dx_offset_coarse=0;
    float patch_dy_offset_coarse=0;


    if(patch_dx_coarse<patch_size_coarse_half)    // patch比较大，大于原图边界，则将patch整体往里缩，以保证patch不越界
    {
        patch_dx_offset_coarse=patch_size_coarse_half-floor(patch_dx_coarse);
        patch_dx_coarse=float(stack_orig.getNx()-1-patch_size_coarse)/float(patch_Nx_coarse+1-2);
    }
    if(patch_dy_coarse<patch_size_coarse_half)
    {
        patch_dy_offset_coarse=patch_size_coarse_half-floor(patch_dy_coarse);
        patch_dy_coarse=float(stack_orig.getNy()-1-patch_size_coarse)/float(patch_Ny_coarse+1-2);
    }
    // loop: patch_Nx_coarse*patch_Ny_coarse (number of patches)
    for(x=patch_dx_orig_coarse+patch_dx_offset_coarse;x<stack_orig.getNx()-patch_dx_orig_coarse-patch_dx_offset_coarse;x+=patch_dx_coarse)
    {
        for(y=patch_dy_orig_coarse+patch_dy_offset_coarse;y<stack_orig.getNy()-patch_dy_orig_coarse-patch_dy_offset_coarse;y+=patch_dy_coarse)
        {
            patch_x_ref_coarse[t]=floor(x);
            patch_y_ref_coarse[t]=floor(y);
            t++;
        }
    }
    #endif 

    // Coarse align with cross-correlation
    float *image_next=new float[stack_orig.getNx()*stack_orig.getNy()];
    float *image_coarse=new float[stack_orig.getNx()*stack_orig.getNy()];
    int patch_dx_sum=0;
    int patch_dy_sum=0;
    int patch_dx_sum_all[stack_orig.getNz()];
    int patch_dy_sum_all[stack_orig.getNz()];
    memset(patch_dx_sum_all,0,sizeof(patch_dx_sum_all));
    memset(patch_dy_sum_all,0,sizeof(patch_dy_sum_all));

    if(!skip_coarse)
    {
        cout << endl << "Performing coarse alignment:" << endl;
        string coarse_mrc=prfx+".coarse";
        MRC stack_coarse(coarse_mrc.c_str(),"wb+");
        stack_coarse.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);

        #ifdef GPU_VERSION
        string path_coarse=path+"/"+"coarse.txt";
        coarse_alignment(ref_n, patch_size_coarse, patch_trans_coarse, patch_Nx_coarse, patch_Ny_coarse, patch_dx_sum_all, patch_dy_sum_all, &stack_orig, &stack_coarse, path_coarse);
        #else
        // loop: Nz (number of images)
        for(n=ref_n;n<stack_orig.getNz()-1;n++) // 从0度图开始向两边找
        {
            int patch_availN=patch_Nx_coarse*patch_Ny_coarse;
            stack_orig.read2DIm_32bit(image_now,n); // 读进来的image行是y列是x！！！（第一维表示y，第二维表示x）
            stack_orig.read2DIm_32bit(image_next,n+1);
            
            // normalization for cross-correlation
            //normalize_image(image_now,stack_orig.getNx(),stack_orig.getNy());
            //normalize_image(image_next,stack_orig.getNx(),stack_orig.getNy());
            standardize_image(image_now,stack_orig.getNx(),stack_orig.getNy());
            standardize_image(image_next,stack_orig.getNx(),stack_orig.getNy());

            float *patch_now[threads],*patch_next[threads];
            for(int th=0;th<threads;th++)   // pre-allocation of memory for omp
            {
                patch_now[th]=new float[patch_size_coarse*patch_size_coarse];
                patch_next[th]=new float[patch_size_coarse*patch_size_coarse];
            }

            #pragma omp parallel for num_threads(threads) reduction(-:patch_availN)
            // loop: patch_Nx_coarse*patch_Ny_coarse (number of patches)
            for(t=0;t<patch_Nx_coarse*patch_Ny_coarse;t++)
            {
                // loop: patch_size_coarse*patch_size_coarse (size of patch)
                for(int j=0;j<patch_size_coarse;j++)    // extrack original patch
                {
                    for(int i=0;i<patch_size_coarse;i++)   // can be be optimized?
                    {
                        patch_now[omp_get_thread_num()][j*patch_size_coarse+i]=image_now[(patch_x_ref_coarse[t]+i-patch_size_coarse_half)+(patch_y_ref_coarse[t]+j-patch_size_coarse_half)*stack_orig.getNx()];
                        patch_next[omp_get_thread_num()][j*patch_size_coarse+i]=0.0;
                    }
                }
                float cc_mat[2*patch_trans_coarse+1][2*patch_trans_coarse+1];
                float cc_max=0;
                int cc_max_x=0;
                int cc_max_y=0;
                // loop: (2*patch_trans_coarse)^2 (translation of each patch)
                for(int jj=-patch_trans_coarse;jj<=patch_trans_coarse;jj++)
                {
                    for(int ii=-patch_trans_coarse;ii<=patch_trans_coarse;ii++)
                    {
                        // loop: patch_size_coarse*patch_size_coarse (size of patch)
                        for(int j=0;j<patch_size_coarse;j++)
                        {
                            for(int i=0;i<patch_size_coarse;i++)   // extract current patch
                            {
                                if(patch_x_ref_coarse[t]+i-patch_size_coarse_half+ii>=0 && patch_x_ref_coarse[t]+i-patch_size_coarse_half+ii<stack_orig.getNx() && patch_y_ref_coarse[t]+j-patch_size_coarse_half+jj>=0 && patch_y_ref_coarse[t]+j-patch_size_coarse_half+jj<stack_orig.getNy())    // check border with 0-padding
                                {
                                    patch_next[omp_get_thread_num()][j*patch_size_coarse+i]=image_next[(patch_x_ref_coarse[t]+i-patch_size_coarse_half+ii)+(patch_y_ref_coarse[t]+j-patch_size_coarse_half+jj)*stack_orig.getNx()];
                                }
                            }
                        }
                        float cc=0,cc_de_1=0,cc_de_2=0;
                        // loop: patch_size_coarse*patch_size_coarse (size of patch)
                        for(int i=0;i<patch_size_coarse*patch_size_coarse;i++)    // compute cross-correlation
                        {
                            cc=cc+patch_now[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                            cc_de_1=cc_de_1+patch_now[omp_get_thread_num()][i]*patch_now[omp_get_thread_num()][i];
                            cc_de_2=cc_de_2+patch_next[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                        }
                        cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse]=cc/sqrt(cc_de_1*cc_de_2);
                        if(cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse]>cc_max)
                        {
                            cc_max=cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse];
                            cc_max_x=ii;
                            cc_max_y=jj;
                        }
                    }
                }
                cout << n << " " << t << " " << cc_max << " " << cc_max_x << " " << cc_max_y << endl;
                patch_deltaX_coarse[t]=cc_max_x;
                patch_deltaY_coarse[t]=cc_max_y;

                // reverse checking
                // loop: patch_size_coarse*patch_size_coarse (size of patch)
                for(int i=0;i<patch_size_coarse*patch_size_coarse;i++)
                {
                    patch_next[omp_get_thread_num()][i]=0.0;
                    patch_now[omp_get_thread_num()][i]=0.0;
                }
                // loop: patch_size_coarse*patch_size_coarse (size of patch)
                for(int j=0;j<patch_size_coarse;j++)
                {
                    for(int i=0;i<patch_size_coarse;i++)    // extract the current patch
                    {
                        if(patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half>=0 && patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half<stack_orig.getNx() && patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half>=0 && patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half<stack_orig.getNy())    // check border with 0-padding
                        {
                            patch_next[omp_get_thread_num()][j*patch_size_coarse+i]=image_next[(patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half)+(patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half)*stack_orig.getNx()];
                        }
                    }
                }
                cc_max=0;
                int cc_max_x_rev=0,cc_max_y_rev=0;
                // loop: (2*patch_trans_coarse)^2 (translation of each patch)
                for(int jj=-patch_trans_coarse;jj<=patch_trans_coarse;jj++)
                {
                    for(int ii=-patch_trans_coarse;ii<=patch_trans_coarse;ii++)
                    {
                        // loop: patch_size_coarse*patch_size_coarse (size of patch)
                        for(int j=0;j<patch_size_coarse;j++)
                        {
                            for(int i=0;i<patch_size_coarse;i++)   // extract current patch
                            {
                                if(patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half+ii>=0 && patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half+ii<stack_orig.getNx() && patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half+jj>=0 && patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half+jj<stack_orig.getNy())    // check border with 0-padding
                                {
                                    patch_now[omp_get_thread_num()][j*patch_size_coarse+i]=image_now[(patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half+ii)+(patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half+jj)*stack_orig.getNx()];
                                }
                            }
                        }
                        float cc=0,cc_de_1=0,cc_de_2=0;
                        // loop: patch_size_coarse*patch_size_coarse (size of patch)
                        for(int i=0;i<patch_size_coarse*patch_size_coarse;i++)    // compute cross-correlation
                        {
                            cc=cc+patch_now[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                            cc_de_1=cc_de_1+patch_now[omp_get_thread_num()][i]*patch_now[omp_get_thread_num()][i];
                            cc_de_2=cc_de_2+patch_next[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                        }
                        cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse]=cc/sqrt(cc_de_1*cc_de_2);
                        if(cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse]>cc_max)
                        {
                            cc_max=cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse];
                            cc_max_x_rev=ii;
                            cc_max_y_rev=jj;
                        }
                    }
                }
                cout << n << " " << t << " " << cc_max << " " << cc_max_x_rev << " " << cc_max_y_rev << endl;
                if(abs(cc_max_x+cc_max_x_rev)<3 && abs(cc_max_y+cc_max_y_rev)<3)
                {
                    cout << "accepted patch" << endl;
                }
                else
                {
                    patch_availN--;
                    patch_deltaX_coarse[t]=0;
                    patch_deltaY_coarse[t]=0;
                    cout << "rejected patch" << endl;
                }
                
            }

            for(int th=0;th<threads;th++)
            {
                delete [] patch_now[th];
                delete [] patch_next[th];
            }

            // average all patch translations as the result of coarse alignment
            int patch_dx_avg=0,patch_dy_avg=0;
            // loop: patch_Nx_coarse*patch_Ny_coarse (number of patches)
            for(t=0;t<patch_Nx_coarse*patch_Ny_coarse;t++)
            {
                patch_dx_avg+=patch_deltaX_coarse[t];
                patch_dy_avg+=patch_deltaY_coarse[t];
            }
            patch_dx_avg/=patch_availN;
            patch_dy_avg/=patch_availN;
            patch_dx_sum+=patch_dx_avg; // 找到与0度图的平移量（一连串平移累加）
            patch_dy_sum+=patch_dy_avg;
            patch_dx_sum_all[n+1]=patch_dx_sum;
            patch_dy_sum_all[n+1]=patch_dy_sum;

            cout << n << ":" << endl;
            cout << "adjacent: (" << patch_dx_avg << "," << patch_dy_avg << ")" << endl;
            cout << "all: (" << patch_dx_sum << "," << patch_dy_sum << ")" << endl;

            // move the micrograph (padding with avg) & write out the coarse aligned stack
            float image_avg=0;
            // loop: Nx*Ny (whole image)
            for(j=0;j<stack_orig.getNy();j++)
            {
                for(i=0;i<stack_orig.getNx();i++)   // calculate average
                {
                    image_avg+=image_next[i+j*stack_orig.getNx()];
                }
            }
            image_avg=image_avg/(stack_orig.getNx()*stack_orig.getNy());
            // loop: Nx*Ny (whole image)
            for(j=0;j<stack_orig.getNy();j++)
            {
                for(i=0;i<stack_orig.getNx();i++)
                {
                    if(i+patch_dx_sum>=0 && i+patch_dx_sum<stack_orig.getNx() && j+patch_dy_sum>=0 && j+patch_dy_sum<stack_orig.getNy())
                    {
                        image_coarse[i+j*stack_orig.getNx()]=image_next[(i+patch_dx_sum)+(j+patch_dy_sum)*stack_orig.getNx()];
                    }
                    else
                    {
                        image_coarse[i+j*stack_orig.getNx()]=image_avg;
                    }
                }
            }
            if(n==ref_n)
            {
                stack_coarse.write2DIm(image_now,n);
            }
            stack_coarse.write2DIm(image_coarse,n+1);
        }
        patch_dx_sum=0;
        patch_dy_sum=0;

        // loop: Nz (number of images)
        for(n=ref_n;n>=1;n--) // 反向找
        {
            int patch_availN=patch_Nx_coarse*patch_Ny_coarse;
            stack_orig.read2DIm_32bit(image_now,n); // 读进来的image行是y列是x！！！（第一维表示y，第二维表示x）
            stack_orig.read2DIm_32bit(image_next,n-1);

            // normalization for cross-correlation
            //normalize_image(image_now,stack_orig.getNx(),stack_orig.getNy());
            //normalize_image(image_next,stack_orig.getNx(),stack_orig.getNy());
            standardize_image(image_now,stack_orig.getNx(),stack_orig.getNy());
            standardize_image(image_next,stack_orig.getNx(),stack_orig.getNy());

            float *patch_now[threads],*patch_next[threads];
            for(int th=0;th<threads;th++)   // pre-allocation of memory for omp
            {
                patch_now[th]=new float[patch_size_coarse*patch_size_coarse];
                patch_next[th]=new float[patch_size_coarse*patch_size_coarse];
            }

            #pragma omp parallel for num_threads(threads) reduction(-:patch_availN)
            // loop: patch_Nx_coarse*patch_Ny_coarse (number of patches)
            for(t=0;t<patch_Nx_coarse*patch_Ny_coarse;t++)
            {
                // loop: patch_size_coarse*patch_size_coarse (size of patch)
                for(int j=0;j<patch_size_coarse;j++)
                {
                    for(int i=0;i<patch_size_coarse;i++)   // can be be optimized?
                    {
                        patch_now[omp_get_thread_num()][j*patch_size_coarse+i]=image_now[(patch_x_ref_coarse[t]+i-patch_size_coarse_half)+(patch_y_ref_coarse[t]+j-patch_size_coarse_half)*stack_orig.getNx()];
                        patch_next[omp_get_thread_num()][j*patch_size_coarse+i]=0.0;
                    }
                }
                float cc_mat[2*patch_trans_coarse+1][2*patch_trans_coarse+1];
                float cc_max=0;
                int cc_max_x=0,cc_max_y=0;
                // loop: (2*patch_trans_coarse)^2 (translation of each patch)
                for(int jj=-patch_trans_coarse;jj<=patch_trans_coarse;jj++)
                {
                    for(int ii=-patch_trans_coarse;ii<=patch_trans_coarse;ii++)
                    {
                        // loop: patch_size_coarse*patch_size_coarse (size of patch)
                        for(int j=0;j<patch_size_coarse;j++)
                        {
                            for(int i=0;i<patch_size_coarse;i++)   // extract current patch
                            {
                                if(patch_x_ref_coarse[t]+i-patch_size_coarse_half+ii>=0 && patch_x_ref_coarse[t]+i-patch_size_coarse_half+ii<stack_orig.getNx() && patch_y_ref_coarse[t]+j-patch_size_coarse_half+jj>=0 && patch_y_ref_coarse[t]+j-patch_size_coarse_half+jj<stack_orig.getNy())    // check border with 0-padding
                                {
                                    patch_next[omp_get_thread_num()][j*patch_size_coarse+i]=image_next[(patch_x_ref_coarse[t]+i-patch_size_coarse_half+ii)+(patch_y_ref_coarse[t]+j-patch_size_coarse_half+jj)*stack_orig.getNx()];
                                }
                            }
                        }
                        float cc=0,cc_de_1=0,cc_de_2=0;
                        // loop: patch_size_coarse*patch_size_coarse (size of patch)
                        for(int i=0;i<patch_size_coarse*patch_size_coarse;i++)    // compute cross-correlation
                        {
                            cc=cc+patch_now[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                            cc_de_1=cc_de_1+patch_now[omp_get_thread_num()][i]*patch_now[omp_get_thread_num()][i];
                            cc_de_2=cc_de_2+patch_next[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                        }
                        cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse]=cc/sqrt(cc_de_1*cc_de_2);
                        if(cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse]>cc_max)
                        {
                            cc_max=cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse];
                            cc_max_x=ii;
                            cc_max_y=jj;
                        }
                    }
                }
                cout << n << " " << t << " " << cc_max << " " << cc_max_x << " " << cc_max_y << endl;
                patch_deltaX_coarse[t]=cc_max_x;
                patch_deltaY_coarse[t]=cc_max_y;

                // reverse checking
                // loop: patch_size_coarse*patch_size_coarse (size of patch)
                for(int i=0;i<patch_size_coarse*patch_size_coarse;i++)
                {
                    patch_next[omp_get_thread_num()][i]=0.0;
                    patch_now[omp_get_thread_num()][i]=0.0;
                }
                // loop: patch_size_coarse*patch_size_coarse (size of patch)
                for(int j=0;j<patch_size_coarse;j++)
                {
                    for(int i=0;i<patch_size_coarse;i++)
                    {
                        if(patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half>=0 && patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half<stack_orig.getNx() && patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half>=0 && patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half<stack_orig.getNy())    // check border with 0-padding
                        {
                            patch_next[omp_get_thread_num()][j*patch_size_coarse+i]=image_next[(patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half)+(patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half)*stack_orig.getNx()];
                        }
                    }
                }
                cc_max=0;
                int cc_max_x_rev=0,cc_max_y_rev=0;
                // loop: (2*patch_trans_coarse)^2 (translation of each patch)
                for(int jj=-patch_trans_coarse;jj<=patch_trans_coarse;jj++)
                {
                    for(int ii=-patch_trans_coarse;ii<=patch_trans_coarse;ii++)
                    {
                        // loop: patch_size_coarse*patch_size_coarse (size of patch)
                        for(int j=0;j<patch_size_coarse;j++)
                        {
                            for(int i=0;i<patch_size_coarse;i++)   // extract current patch
                            {
                                if(patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half+ii>=0 && patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half+ii<stack_orig.getNx() && patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half+jj>=0 && patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half+jj<stack_orig.getNy())    // check border with 0-padding
                                {
                                    patch_now[omp_get_thread_num()][j*patch_size_coarse+i]=image_now[(patch_x_ref_coarse[t]+patch_deltaX_coarse[t]+i-patch_size_coarse_half+ii)+(patch_y_ref_coarse[t]+patch_deltaY_coarse[t]+j-patch_size_coarse_half+jj)*stack_orig.getNx()];
                                }
                            }
                        }
                        float cc=0,cc_de_1=0,cc_de_2=0;
                        // loop: patch_size_coarse*patch_size_coarse (size of patch)
                        for(int i=0;i<patch_size_coarse*patch_size_coarse;i++)    // compute cross-correlation
                        {
                            cc=cc+patch_now[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                            cc_de_1=cc_de_1+patch_now[omp_get_thread_num()][i]*patch_now[omp_get_thread_num()][i];
                            cc_de_2=cc_de_2+patch_next[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                        }
                        cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse]=cc/sqrt(cc_de_1*cc_de_2);
                        if(cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse]>cc_max)
                        {
                            cc_max=cc_mat[ii+patch_trans_coarse][jj+patch_trans_coarse];
                            cc_max_x_rev=ii;
                            cc_max_y_rev=jj;
                        }
                    }
                }
                cout << n << " " << t << " " << cc_max << " " << cc_max_x_rev << " " << cc_max_y_rev << endl;
                if(abs(cc_max_x+cc_max_x_rev)<3 && abs(cc_max_y+cc_max_y_rev)<3)
                {
                    cout << "accepted patch" << endl;
                }
                else
                {
                    patch_availN--;
                    patch_deltaX_coarse[t]=0;
                    patch_deltaY_coarse[t]=0;
                    cout << "rejected patch" << endl;
                }
            }

            for(int th=0;th<threads;th++)
            {
                delete [] patch_now[th];
                delete [] patch_next[th];
            }

            // average all patch translations as the result of coarse alignment
            int patch_dx_avg=0,patch_dy_avg=0;
            // loop: patch_Nx_coarse*patch_Ny_coarse (number of patches)
            for(t=0;t<patch_Nx_coarse*patch_Ny_coarse;t++)
            {
                patch_dx_avg+=patch_deltaX_coarse[t];
                patch_dy_avg+=patch_deltaY_coarse[t];
            }
            patch_dx_avg/=patch_availN;
            patch_dy_avg/=patch_availN;
            patch_dx_sum+=patch_dx_avg; // 找到与0度图的平移量（一连串平移累加）
            patch_dy_sum+=patch_dy_avg;
            patch_dx_sum_all[n-1]=patch_dx_sum;
            patch_dy_sum_all[n-1]=patch_dy_sum;

            cout << n << ":" << endl;
            cout << "adjacent: (" << patch_dx_avg << "," << patch_dy_avg << ")" << endl;
            cout << "all: (" << patch_dx_sum << "," << patch_dy_sum << ")" << endl;

            // move the micrograph (padding with avg) & write out the coarse aligned stack
            float image_avg=0;
            // loop: Nx*Ny (whole image)
            for(j=0;j<stack_orig.getNy();j++)
            {
                for(i=0;i<stack_orig.getNx();i++)   // calculate average
                {
                    image_avg+=image_next[i+j*stack_orig.getNx()];
                }
            }
            image_avg=image_avg/(stack_orig.getNx()*stack_orig.getNy());
            // loop: Nx*Ny (whole image)
            for(j=0;j<stack_orig.getNy();j++)
            {
                for(i=0;i<stack_orig.getNx();i++)
                {
                    if(i+patch_dx_sum>=0 && i+patch_dx_sum<stack_orig.getNx() && j+patch_dy_sum>=0 && j+patch_dy_sum<stack_orig.getNy())
                    {
                        image_coarse[i+j*stack_orig.getNx()]=image_next[(i+patch_dx_sum)+(j+patch_dy_sum)*stack_orig.getNx()];
                    }
                    else
                    {
                        image_coarse[i+j*stack_orig.getNx()]=image_avg;
                    }
                }
            }
            stack_coarse.write2DIm(image_coarse,n-1);
        }

        // write out coarse translation
        string path_coarse=path+"/"+"coarse.txt";
        FILE *fcoarse=fopen(path_coarse.c_str(),"w");  // 该文件中的平移量均为相对0度图的平移
        for(n=0;n<stack_orig.getNz();n++)
        {
            fprintf(fcoarse,"%d %d\n",patch_dx_sum_all[n],patch_dy_sum_all[n]);
        }

        fflush(fcoarse);
        fclose(fcoarse);
        delete [] image_coarse;
        #endif

        stack_coarse.computeHeader_omp(pix*float(bin),true,threads);
        stack_coarse.close();
        stack_orig.close();
        stack_orig.open(coarse_mrc.c_str(),"rb");
        cout << "Finish coarse alignment!" << endl;
    
    }
    else
    {
        cout << "Skip coarse alignment, read coarse align file instead!" << endl;
        // read in coarse alignment results
        string path_coarse=path+"/"+"coarse.txt";
        FILE *fcoarse=fopen(path_coarse.c_str(),"r");
        if(fcoarse==NULL)
        {
            cout << "No coarse align file, set default: 0" << endl;
        }
        else
        {
            for(n=0;n<stack_orig.getNz();n++)
            {
                fscanf(fcoarse,"%d %d",&patch_dx_sum_all[n],&patch_dy_sum_all[n]);
            }
            cout << "Finish reading coarse align file" << endl;
        }
    }



    // Patch tracking

    // initial model for patch tracking
    #ifdef GPU_VERSION
    #else
    t=0;
    int patch_x_ref[patch_Nx*patch_Ny],patch_y_ref[patch_Nx*patch_Ny];  // 存放特征点坐标
    int patch_deltaX[patch_Nx*patch_Ny],patch_deltaY[patch_Nx*patch_Ny];   // 对应特征点坐标的变化量（平移量）
    float patch_dx=float(stack_orig.getNx()-1)/float(patch_Nx+1);
    float patch_dy=float(stack_orig.getNy()-1)/float(patch_Ny+1);    // 每个patch中心点之间的距离
    float patch_dx_orig=patch_dx;
    float patch_dy_orig=patch_dy;
    float patch_dx_offset=0;
    float patch_dy_offset=0;
    if(patch_dx<patch_size_half)    // patch比较大，大于原图边界，则将patch整体往里缩，以保证patch不越界
    {
        patch_dx_offset=patch_size_half-floor(patch_dx);
        patch_dx=float(stack_orig.getNx()-1-patch_size)/float(patch_Nx+1-2);
    }
    if(patch_dy<patch_size_half)
    {
        patch_dy_offset=patch_size_half-floor(patch_dy);
        patch_dy=float(stack_orig.getNy()-1-patch_size)/float(patch_Ny+1-2);
    }
    // loop: patch_Nx*patch_Ny (number of patches)
    for(x=patch_dx_orig+patch_dx_offset;x<stack_orig.getNx()-patch_dx_orig-patch_dx_offset;x+=patch_dx)
    {
        for(y=patch_dy_orig+patch_dy_offset;y<stack_orig.getNy()-patch_dy_orig-patch_dy_offset;y+=patch_dy)
        {
            patch_x_ref[t]=floor(x);
            patch_y_ref[t]=floor(y);
            t++;
        }
    }
    #endif

    // Initial model (divide the reference image(0 degree) equally)
    int patch_x[stack_orig.getNz()][patch_Nx*patch_Ny];
    int patch_y[stack_orig.getNz()][patch_Nx*patch_Ny];  // 存放特征点坐标，第一维遍历图片，第二维遍历特征点
    bool patch_avail[stack_orig.getNz()][patch_Nx*patch_Ny];    // patch_avail[n][t]=true表示第t个marker在第n张图和第n+1张图中的对应patch可以接受
//    float patch_dx=float(stack_orig.getNx()-1)/float(patch_Nx+1),patch_dy=float(stack_orig.getNy()-1)/float(patch_Ny+1);    // 每个patch中心点之间的距离
    if(!skip_patch)
    {
        cout << endl << "Performing patch tracking:" << endl;
        #ifdef GPU_VERSION
        string path_string=path+"/"+"patch.txt";
        patch_tracking(ref_n, patch_size, patch_trans, patch_Nx, patch_Ny, &stack_orig, path_string);
        #else
        t=0;
        memset(patch_avail,0,sizeof(bool)*stack_orig.getNz()*patch_Nx*patch_Ny);
        // loop: patch_Nx*patch_Ny (number of patches)
        for(x=patch_dx_orig+patch_dx_offset;x<stack_orig.getNx()-patch_dx_orig-patch_dx_offset;x+=patch_dx)
        {
            for(y=patch_dy_orig+patch_dy_offset;y<stack_orig.getNy()-patch_dy_orig-patch_dy_offset;y+=patch_dy)
            {
                patch_x[ref_n][t]=floor(x);
                patch_y[ref_n][t]=floor(y);
                t++;
            }
        }

        // Tracking & reverse checking
        // loop: Nz (number of images)
        for(n=ref_n;n<stack_orig.getNz()-1;n++) // 从0度图开始向两边找
        {
            stack_orig.read2DIm_32bit(image_now,n); // 读进来的image行是y列是x！！！（第一维表示y，第二维表示x）
            stack_orig.read2DIm_32bit(image_next,n+1);

            float *patch_now[threads],*patch_next[threads];
            for(int th=0;th<threads;th++)   // pre-allocation of memory for omp
            {
                patch_now[th]=new float[patch_size*patch_size];
                patch_next[th]=new float[patch_size*patch_size];
            }

            #pragma omp parallel for num_threads(threads)
            // loop: patch_Nx*patch_Ny (number of patches)
            for(t=0;t<patch_Nx*patch_Ny;t++)
            {
                // loop: patch_size^2 (size of patch)
                for(int j=0;j<patch_size;j++)
                {
                    for(int i=0;i<patch_size;i++)   // can be be optimized?
                    {
                        patch_now[omp_get_thread_num()][j*patch_size+i]=image_now[(patch_x[n][t]+i-patch_size_half)+(patch_y[n][t]+j-patch_size_half)*stack_orig.getNx()];
                        patch_next[omp_get_thread_num()][j*patch_size+i]=0.0;
                    }
                }

                float cc_mat[2*patch_trans+1][2*patch_trans+1];
                float cc_max=0;
                int cc_max_x=0,cc_max_y=0;
                // loop: (2*patch_trans+1)^2 (translation of patch)
                
                for(int jj=-patch_trans;jj<=patch_trans;jj++)
                {
                    for(int ii=-patch_trans;ii<=patch_trans;ii++)
                    {
                        // loop: patch_size^2 (size of patch)
                        for(int j=0;j<patch_size;j++)
                        {
                            for(int i=0;i<patch_size;i++)   // extract current patch
                            {
                                if(patch_x[n][t]+i-patch_size_half+ii>=0 && patch_x[n][t]+i-patch_size_half+ii<stack_orig.getNx() && patch_y[n][t]+j-patch_size_half+jj>=0 && patch_y[n][t]+j-patch_size_half+jj<stack_orig.getNy())    // check border with 0-padding
                                {
                                    patch_next[omp_get_thread_num()][j*patch_size+i]=image_next[(patch_x[n][t]+i-patch_size_half+ii)+(patch_y[n][t]+j-patch_size_half+jj)*stack_orig.getNx()];
                                }
                            }
                        }
                        float cc=0,cc_de_1=0,cc_de_2=0;
                        // loop: patch_size^2 (size of patch)
                        for(int i=0;i<patch_size*patch_size;i++)    // compute cross-correlation
                        {
                            cc=cc+patch_now[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                            cc_de_1=cc_de_1+patch_now[omp_get_thread_num()][i]*patch_now[omp_get_thread_num()][i];
                            cc_de_2=cc_de_2+patch_next[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                        }
                        cc_mat[ii+patch_trans][jj+patch_trans]=cc/sqrt(cc_de_1*cc_de_2);
                        if(cc_mat[ii+patch_trans][jj+patch_trans]>cc_max)
                        {
                            cc_max=cc_mat[ii+patch_trans][jj+patch_trans];
                            cc_max_x=ii;
                            cc_max_y=jj;
                        }
                    }
                }
                cout << n << " " << t << " " << cc_max << " " << cc_max_x << " " << cc_max_y << endl;
                patch_x[n+1][t]=patch_x[n][t]+cc_max_x;
                patch_y[n+1][t]=patch_y[n][t]+cc_max_y;

                // reverse checking
                // loop: patch_size^2 (size of patch)
                for(int i=0;i<patch_size*patch_size;i++)
                {
                    patch_next[omp_get_thread_num()][i]=0.0;
                    patch_now[omp_get_thread_num()][i]=0.0;
                }
                // loop: patch_size^2 (size of patch)
                for(int j=0;j<patch_size;j++)
                {
                    for(int i=0;i<patch_size;i++)
                    {
                        if(patch_x[n+1][t]+i-patch_size_half>=0 && patch_x[n+1][t]+i-patch_size_half<stack_orig.getNx() && patch_y[n+1][t]+j-patch_size_half>=0 && patch_y[n+1][t]+j-patch_size_half<stack_orig.getNy())    // check border with 0-padding
                        {
                            patch_next[omp_get_thread_num()][j*patch_size+i]=image_next[(patch_x[n+1][t]+i-patch_size_half)+(patch_y[n+1][t]+j-patch_size_half)*stack_orig.getNx()];
                        }
                    }
                }
                cc_max=0;
                int cc_max_x_rev=0,cc_max_y_rev=0;
                // loop: (2*patch_trans+1)^2 (translation of patch)
                for(int jj=-patch_trans;jj<=patch_trans;jj++)
                {
                    for(int ii=-patch_trans;ii<=patch_trans;ii++)
                    {
                        // loop: patch_size^2 (size of patch)
                        for(int j=0;j<patch_size;j++)
                        {
                            for(int i=0;i<patch_size;i++)   // extract current patch
                            {
                                if(patch_x[n+1][t]+i-patch_size_half+ii>=0 && patch_x[n+1][t]+i-patch_size_half+ii<stack_orig.getNx() && patch_y[n+1][t]+j-patch_size_half+jj>=0 && patch_y[n+1][t]+j-patch_size_half+jj<stack_orig.getNy())    // check border with 0-padding
                                {
                                    patch_now[omp_get_thread_num()][j*patch_size+i]=image_now[(patch_x[n+1][t]+i-patch_size_half+ii)+(patch_y[n+1][t]+j-patch_size_half+jj)*stack_orig.getNx()];
                                }
                            }
                        }
                        float cc=0,cc_de_1=0,cc_de_2=0;
                        // loop: patch_size^2 (size of patch)
                        for(int i=0;i<patch_size*patch_size;i++)    // compute cross-correlation
                        {
                            cc=cc+patch_now[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                            cc_de_1=cc_de_1+patch_now[omp_get_thread_num()][i]*patch_now[omp_get_thread_num()][i];
                            cc_de_2=cc_de_2+patch_next[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                        }
                        cc_mat[ii+patch_trans][jj+patch_trans]=cc/sqrt(cc_de_1*cc_de_2);
                        if(cc_mat[ii+patch_trans][jj+patch_trans]>cc_max)
                        {
                            cc_max=cc_mat[ii+patch_trans][jj+patch_trans];
                            cc_max_x_rev=ii;
                            cc_max_y_rev=jj;
                        }
                    }
                }
                cout << n << " " << t << " " << cc_max << " " << cc_max_x_rev << " " << cc_max_y_rev << endl;
                if(abs(cc_max_x+cc_max_x_rev)<3 && abs(cc_max_y+cc_max_y_rev)<3)
                {
                    patch_avail[n][t]=true;
                    cout << "accepted patch" << endl;
                }
                else
                {
                    cout << "rejected patch" << endl;
                }
            }

            for(int th=0;th<threads;th++)
            {
                delete [] patch_now[th];
                delete [] patch_next[th];
            }
        }

        // loop: Nz (number of images)
        for(n=ref_n;n>=1;n--) // 从0度图开始向两边找
        {
            stack_orig.read2DIm_32bit(image_now,n); // 读进来的image行是y列是x！！！
            stack_orig.read2DIm_32bit(image_next,n-1);

            float *patch_now[threads],*patch_next[threads];
            for(int th=0;th<threads;th++)   // pre-allocation of memory for omp
            {
                patch_now[th]=new float[patch_size*patch_size];
                patch_next[th]=new float[patch_size*patch_size];
            }

            #pragma omp parallel for num_threads(threads)
            // loop: patch_Nx*patch_Ny (number of patches)
            for(t=0;t<patch_Nx*patch_Ny;t++)
            {
                // loop: patch_size^2 (size of patch)
                for(int j=0;j<patch_size;j++)
                {
                    for(int i=0;i<patch_size;i++)   // can be optimized??
                    {
                        patch_now[omp_get_thread_num()][j*patch_size+i]=image_now[(patch_x[n][t]+i-patch_size_half)+(patch_y[n][t]+j-patch_size_half)*stack_orig.getNx()];
                        patch_next[omp_get_thread_num()][j*patch_size+i]=0.0;
                    }
                }
                float cc_mat[2*patch_trans+1][2*patch_trans+1];
                float cc_max=0;
                int cc_max_x=0,cc_max_y=0;
                // loop: (2*patch_trans+1)^2 (translation of patch)
                for(int jj=-patch_trans;jj<=patch_trans;jj++)
                {
                    for(int ii=-patch_trans;ii<=patch_trans;ii++)
                    {
                        // loop: patch_size^2 (size of patch)
                        for(int j=0;j<patch_size;j++)
                        {
                            for(int i=0;i<patch_size;i++)   // extract current patch
                            {
                                if(patch_x[n][t]+i-patch_size_half+ii>=0 && patch_x[n][t]+i-patch_size_half+ii<stack_orig.getNx() && patch_y[n][t]+j-patch_size_half+jj>=0 && patch_y[n][t]+j-patch_size_half+jj<stack_orig.getNy())    // check border with 0-padding
                                {
                                    patch_next[omp_get_thread_num()][j*patch_size+i]=image_next[(patch_x[n][t]+i-patch_size_half+ii)+(patch_y[n][t]+j-patch_size_half+jj)*stack_orig.getNx()];
                                }
                            }
                        }
                        float cc=0,cc_de_1=0,cc_de_2=0;
                        // loop: patch_size^2 (size of patch)
                        for(int i=0;i<patch_size*patch_size;i++)    // compute cross-correlation
                        {
                            cc=cc+patch_now[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                            cc_de_1=cc_de_1+patch_now[omp_get_thread_num()][i]*patch_now[omp_get_thread_num()][i];
                            cc_de_2=cc_de_2+patch_next[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                        }
                        cc_mat[ii+patch_trans][jj+patch_trans]=cc/sqrt(cc_de_1*cc_de_2);
                        if(cc_mat[ii+patch_trans][jj+patch_trans]>cc_max)
                        {
                            cc_max=cc_mat[ii+patch_trans][jj+patch_trans];
                            cc_max_x=ii;
                            cc_max_y=jj;
                        }
                    }
                }
                cout << n << " " << t << " " << cc_max << " " << cc_max_x << " " << cc_max_y << endl;
                patch_x[n-1][t]=patch_x[n][t]+cc_max_x;
                patch_y[n-1][t]=patch_y[n][t]+cc_max_y;

                // reverse checking
                // loop: patch_size^2 (size of patch)
                for(int i=0;i<patch_size*patch_size;i++)
                {
                    patch_next[omp_get_thread_num()][i]=0.0;
                    patch_now[omp_get_thread_num()][i]=0.0;
                }
                // loop: patch_size^2 (size of patch)
                for(int j=0;j<patch_size;j++)
                {
                    for(int i=0;i<patch_size;i++)
                    {
                        if(patch_x[n-1][t]+i-patch_size_half>=0 && patch_x[n-1][t]+i-patch_size_half<stack_orig.getNx() && patch_y[n-1][t]+j-patch_size_half>=0 && patch_y[n-1][t]+j-patch_size_half<stack_orig.getNy())    // check border with 0-padding
                        {
                            patch_next[omp_get_thread_num()][j*patch_size+i]=image_next[(patch_x[n-1][t]+i-patch_size_half)+(patch_y[n-1][t]+j-patch_size_half)*stack_orig.getNx()];
                        }
                    }
                }
                cc_max=0;
                int cc_max_x_rev=0,cc_max_y_rev=0;
                // loop: (2*patch_trans+1)^2 (translation of patch)
                for(int jj=-patch_trans;jj<=patch_trans;jj++)
                {
                    for(int ii=-patch_trans;ii<=patch_trans;ii++)
                    {
                        // loop: patch_size^2 (size of patch)
                        for(int j=0;j<patch_size;j++)
                        {
                            for(int i=0;i<patch_size;i++)   // extract current patch
                            {
                                if(patch_x[n-1][t]+i-patch_size_half+ii>=0 && patch_x[n-1][t]+i-patch_size_half+ii<stack_orig.getNx() && patch_y[n-1][t]+j-patch_size_half+jj>=0 && patch_y[n-1][t]+j-patch_size_half+jj<stack_orig.getNy())    // check border with 0-padding
                                {
                                    patch_now[omp_get_thread_num()][j*patch_size+i]=image_now[(patch_x[n-1][t]+i-patch_size_half+ii)+(patch_y[n-1][t]+j-patch_size_half+jj)*stack_orig.getNx()];
                                }
                            }
                        }
                        float cc=0,cc_de_1=0,cc_de_2=0;
                        // loop: patch_size^2 (size of patch)
                        for(int i=0;i<patch_size*patch_size;i++)    // compute cross-correlation
                        {
                            cc=cc+patch_now[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                            cc_de_1=cc_de_1+patch_now[omp_get_thread_num()][i]*patch_now[omp_get_thread_num()][i];
                            cc_de_2=cc_de_2+patch_next[omp_get_thread_num()][i]*patch_next[omp_get_thread_num()][i];
                        }
                        cc_mat[ii+patch_trans][jj+patch_trans]=cc/sqrt(cc_de_1*cc_de_2);
                        if(cc_mat[ii+patch_trans][jj+patch_trans]>cc_max)
                        {
                            cc_max=cc_mat[ii+patch_trans][jj+patch_trans];
                            cc_max_x_rev=ii;
                            cc_max_y_rev=jj;
                        }
                    }
                }
                cout << n << " " << t << " " << cc_max << " " << cc_max_x_rev << " " << cc_max_y_rev << endl;
                if(abs(cc_max_x+cc_max_x_rev)<3 && abs(cc_max_y+cc_max_y_rev)<3)
                {
                    patch_avail[n-1][t]=true;
                    cout << "accepted patch" << endl;
                }
                else
                {
                    cout << "rejected patch" << endl;
                }
            }

            for(int th=0;th<threads;th++)
            {
                delete [] patch_now[th];
                delete [] patch_next[th];
            }
        }
        #endif
        cout << "Finish patch tracking!" << endl;
    }
    else
    {
        cout << endl << "Skip patch tracking, load patch stats instead!" << endl;
    }
    // load patch file
    string path_string=path+"/"+"patch.txt";
    FILE *fpatch=fopen(path_string.c_str(),"r");
    if(fpatch==NULL)
    {
        cerr << "No patch file available!" << endl;
            abort();
    }
        // loop: Nz (number of images)
    for(n=0;n<stack_orig.getNz();n++)   // patch_avail
    {
        // loop: patch_Nx*patch_Ny (number of patches)
        for(t=0;t<patch_Nx*patch_Ny;t++)
        {
            fscanf(fpatch,"%d",&patch_avail[n][t]);
        }
    }
    // loop: Nz (number of images)
    for(n=0;n<stack_orig.getNz();n++)
    {
        // loop: patch_Nx*patch_Ny (number of patches)
        for(t=0;t<patch_Nx*patch_Ny;t++)
        {
            fscanf(fpatch,"%d",&patch_x[n][t]);
        }
    }
    // loop: Nz (number of images)
    for(n=0;n<stack_orig.getNz();n++)
    {
        // loop: patch_Nx*patch_Ny (number of patches)
        for(t=0;t<patch_Nx*patch_Ny;t++)
        {
            fscanf(fpatch,"%d",&patch_y[n][t]);
        }
    }
    fflush(fpatch);
    fclose(fpatch);
    
    // Check and save tracks
    vector<Track> tracks;
    int patch_num_all=0;
    const int track_length_avail_min=5;
    // loop: patch_Nx*patch_Ny (number of patches)
    for(t=0;t<patch_Nx*patch_Ny;t++)
    {
        int track_start=0;
        // loop: Nz (number of images)
        for(n=0;n<stack_orig.getNz();n++)
        {
            if(patch_avail[n][t]==0)    // find the end of the last patch
            {
                int track_length=n-track_start+1;
                if(track_length>=track_length_avail_min) // track is long enough
                {
                    float corr;
                    float x_avg=0,y_avg=0,x_sum2=0,y_sum2=0,xy_sum=0;
                    for(k=track_start;k<=n;k++)
                    {
                        x_avg=x_avg+patch_x[k][t];
                        y_avg=y_avg+patch_y[k][t];
                    }
                    x_avg=x_avg/track_length;
                    y_avg=y_avg/track_length;
                    for(k=track_start;k<=n;k++)
                    {
                        xy_sum=xy_sum+(patch_x[k][t]-x_avg)*(patch_y[k][t]-y_avg);
                        x_sum2=x_sum2+(patch_x[k][t]-x_avg)*(patch_x[k][t]-x_avg);
                        y_sum2=y_sum2+(patch_y[k][t]-y_avg)*(patch_y[k][t]-y_avg);
                    }
                    corr=xy_sum/sqrt(x_sum2*y_sum2);
                    if(fabs(corr)>0.8)    // check linearity
                    {
                        patch_num_all+=track_length;
                        Track track_now(patch_x[ref_n][t],patch_y[ref_n][t],0);
                        for(k=track_start;k<=n;k++)
                        {
                            track_now.addPatch(patch_x[k][t],patch_y[k][t],k);
                        }
                        tracks.push_back(track_now);
                    }
                }
                track_start=n+1;
            }
        }
    }
    n=0;
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        cout << "Track " << n << ": " << endl;
        cout << "Length: " << it->getLength() << endl;
        cout << "Marker3D: (" << it->getMarker3D_x() << "," << it->getMarker3D_y() << "," << it->getMarker3D_z() << ")" << endl;
        cout << "Patch:" << endl;
        for(t=0;t<it->getLength();t++)
        {
            cout << "(" << it->getMarker2D_x(t) << "," << it->getMarker2D_y(t) << "," << it->getMarker2D_n(t) << ")" << endl;
        }
        n++;
    }

    // Least square regression and eliminate wild tracks
    int A_Nx=patch_num_all;
    int A_Ny=tracks.size()+1;
    Eigen::MatrixXf A(A_Nx,A_Ny);
    Eigen::VectorXf b(A_Nx);
    A.setZero();
    i=0;
    j=1;
    // loop: number of tracks
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        // loop: length of track
        for(t=0;t<it->getLength();t++)
        {
            A(i,j)=1;
            A(i,0)=it->getMarker2D_x(t);
            b(i)=it->getMarker2D_y(t);
            i++;
        }
        j++;
    }
    Eigen::MatrixXf ATA(A_Ny,A_Ny);
    Eigen::VectorXf ATb(A_Ny);
    ATA=A.transpose()*A;
    ATb=A.transpose()*b;
    Eigen::VectorXf para(A_Ny);
    para=ATA.inverse()*ATb;
    for(i=0;i<A_Ny;i++)
    {
        cout << para(i) << " ";
    }
    cout << endl;
    // eliminate wild tracks(tracks with big regression loss)
    const float loss_avail_max=1;
    n=1;
    // loop: number of tracks
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        float loss=0;
        // loop: length of track
        for(t=0;t<it->getLength();t++)
        {
            float y_est=para(0)*it->getMarker2D_x(t)+para(n);
            loss=loss+((y_est-it->getMarker2D_y(t))*cos(atan(para(0))))*((y_est-it->getMarker2D_y(t))*cos(atan(para(0))));
        }
        loss=loss/it->getLength();
        if(loss>loss_avail_max)
        {
            it->changeStatus(false);
            patch_num_all-=it->getLength();
        }
        n++;
    }
    // another least square regression without wild tracks
    A_Nx=patch_num_all;
    A_Ny=tracks_size(tracks)+1;
    A.resize(A_Nx,A_Ny);
    b.resize(A_Nx);
    A.setZero();
    i=0;
    j=1;
    // loop: number of tracks
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        if(it->getStatus()==true)
        {
            // loop: length of track
            for(t=0;t<it->getLength();t++)
            {
                A(i,j)=1;
                A(i,0)=it->getMarker2D_x(t);
                b(i)=it->getMarker2D_y(t);
                i++;
            }
            j++;
        }
    }
    ATA.resize(A_Ny,A_Ny);
    ATb.resize(A_Ny);
    ATA=A.transpose()*A;
    ATb=A.transpose()*b;
    para=ATA.inverse()*ATb;
    for(i=0;i<A_Ny;i++)
    {
        cout << para(i) << " ";
    }
    cout << endl;
    cout << "Remained tracks:" << endl;
    n=0;
    // loop: number of tracks
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        if(it->getStatus())
        {
            cout << "Track " << n << ": " << endl;
            cout << "Length: " << it->getLength() << endl;
            cout << "Marker3D: (" << it->getMarker3D_x() << "," << it->getMarker3D_y() << "," << it->getMarker3D_z() << ")" << endl;
            cout << "Patch:" << endl;
            // loop: length of track
            for(t=0;t<it->getLength();t++)
            {
                cout << "(" << it->getMarker2D_x(t) << "," << it->getMarker2D_y(t) << "," << it->getMarker2D_n(t) << ")" << endl;
            }
            n++;
        }
    }

    // remove wild points from regression
/*    n=1;
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        if(it->getStatus())
        {
            for(t=0;t<it->getLength();t++)
            {
                float y_est=para(0)*it->getMarker2D_x(t)+para(n);
                float loss=((y_est-it->getMarker2D_y(t))*cos(atan(para(0))))*((y_est-it->getMarker2D_y(t))*cos(atan(para(0))));
                if(loss>loss_avail_max)
                {
                    it->changeStatus2D(false,t);
                    patch_num_all--;
                }
            }
            n++;
        }
    }
    // another least square regression without wild points
    A_Nx=patch_num_all;
    A_Ny=tracks_size(tracks)+1;
    A.resize(A_Nx,A_Ny);
    b.resize(A_Nx);
    A.setZero();
    i=0;
    j=1;
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        if(it->getStatus()==true)
        {
            for(t=0;t<it->getLength();t++)
            {
                if(it->getMarker2D_avail(t))
                {
                    A(i,j)=1;
                    A(i,0)=it->getMarker2D_x(t);
                    b(i)=it->getMarker2D_y(t);
                    i++;
                }
            }
            j++;
        }
    }
    ATA.resize(A_Ny,A_Ny);
    ATb.resize(A_Ny);
    ATA=A.transpose()*A;
    ATb=A.transpose()*b;
    para=ATA.inverse()*ATb;
    for(i=0;i<A_Ny;i++)
    {
        cout << para(i) << " ";
    }
    cout << endl;
    cout << "Remained tracks:" << endl;
    n=0;
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        if(it->getStatus())
        {
            cout << "Track " << n << ": " << endl;
            cout << "Length: " << it->getLength() << endl;
            cout << "Marker3D: (" << it->getMarker3D_x() << "," << it->getMarker3D_y() << "," << it->getMarker3D_z() << ")" << endl;
            cout << "Patch:" << endl;
            for(t=0;t<it->getLength();t++)
            {
                if(it->getMarker2D_avail(t))
                {
                    cout << "(" << it->getMarker2D_x(t) << "," << it->getMarker2D_y(t) << "," << it->getMarker2D_n(t) << ")" << endl;
                }
            }
            n++;
        }
    }*/



    // Fine align
    cout << endl << "Performing fine alignment:" << endl;

    bool beam_induced_motion=false;
    it=inputPara.find("beam_induced_motion");
    if(it!=inputPara.end())
    {
        beam_induced_motion=atoi(it->second.c_str());
        cout << "Align for beam-induced motion: " << beam_induced_motion << endl;
    }
    else
    {
        cout << "Skip aligning for beam-induced motion!" << endl;
    }

    int it_max=5;
    it=inputPara.find("it_max");
    if(it!=inputPara.end())
    {
        it_max=atoi(it->second.c_str());
        cout << "Maximum iteration: " << it_max << endl;
    }
    else
    {
        cout << "No maximum iteration, set default: 5" << endl;
        it_max=5;
    }

    bool correct_BI[stack_orig.getNz()];    // whether available to correct beam-induced motion
    float BI_x[stack_orig.getNz()];
    float BI_y[stack_orig.getNz()];
    float BI_para_x[stack_orig.getNz()][6];
    float BI_para_y[stack_orig.getNz()][6];
    // loop: Nz (number of images)
    for(n=0;n<stack_orig.getNz();n++)
    {
        correct_BI[n]=false;
        BI_x[n]=0.0;
        BI_y[n]=0.0;
        for(int t=0;t<6;t++)
        {
            BI_para_x[n][t]=0.0;
            BI_para_y[n][t]=0.0;
        }
    }

    float psi=atan(para(0));    // in rad
    float psi_deg=psi/M_PI*180;
    // change the origin to the center & flip y-axis
    // loop: number of tracks
    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
    {
        it->move_origin(-float(stack_orig.getNx())/2,-float(stack_orig.getNy())/2,false,false);
    }
    int iter;
    float d_x[stack_orig.getNz()],d_y[stack_orig.getNz()];
    bool reliable[stack_orig.getNz()];
    for(n=0;n<stack_orig.getNz();n++)
    {
        reliable[n]=true;
        d_x[n]=0.0;
        d_y[n]=0.0;
    }

    if(!skip_fine)
    {
        Eigen::Matrix3f tmp_1;
        Eigen::Vector3f tmp_2;
        cout << "it: 0 (initial 3D-marker)" << endl;
        // initial 3D-marker
        int marker_N=0;
        // loop: number of tracks
        for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
        {
            if(it->getStatus()==true)
            {
                tmp_1.setZero();
                tmp_2.setZero();
                Eigen::Matrix3f R_psi,R_theta;  // theta: tilt angle; psi: axis rotation
                Eigen::Matrix<float,2,3> P; // P: projection matrix {{1,0,0};{0,1,0}}
                R_psi.setZero();
                R_theta.setZero();
                P.setZero();
                P(0,0)=1;
                P(1,1)=1;
                // loop: length of track
                for(t=0;t<it->getLength();t++)
                {
                    if(it->getMarker2D_avail(t))
                    {
                        n=it->getMarker2D_n(t);
                        R_psi.setZero();    // R_psi: [[cosd(-axis),-sind(-axis),0];[sind(-axis),cosd(-axis),0];[0,0,1]]
                        R_theta.setZero();  // R_theta: [[cosd(theta_est(n)),0,-sind(theta_est(n))];[0,1,0];[sind(theta_est(n)),0,cosd(theta_est(n))]]
                        float psi_rad=psi,theta_rad=theta[n]/180*M_PI;
                        R_psi << cos(-psi_rad),-sin(-psi_rad),0,sin(-psi_rad),cos(-psi_rad),0,0,0,1;
                        R_theta << cos(theta_rad),0,-sin(theta_rad),0,1,0,sin(theta_rad),0,cos(theta_rad);
                        A.resize(2,3);
                        b.resize(2);
                        ATA.resize(3,3);
                        ATb.resize(3);
                        A=P*R_psi.transpose()*R_theta*R_psi;
                        ATA=A.transpose()*A;
                        b << (it->getMarker2D_x(t)-d_x[n]),(it->getMarker2D_y(t)-d_y[n]);
                        ATb=A.transpose()*b;
                        tmp_1=tmp_1+ATA;
                        tmp_2=tmp_2+ATb;
                    }
                }
                Eigen::Vector3f r_new;
                r_new=tmp_1.inverse()*tmp_2;
                it->setMarker3D(r_new(0),r_new(1),r_new(2));
                cout << "Marker " << marker_N << ": (" << r_new(0) << "," << r_new(1) << "," << r_new(2) << ")" << endl;
                marker_N++;
            }
        }
        // loop: it_max (number of iterations)
        for(iter=1;iter<=it_max;iter++)
        {
            cout << "it: " << iter << endl;
            
            // optimize for translation (dx,dy)
            float p_sum_x[stack_orig.getNz()],p_sum_y[stack_orig.getNz()];  // 2D-marker
            float r_sum_x[stack_orig.getNz()],r_sum_y[stack_orig.getNz()],r_sum_z[stack_orig.getNz()];  // 3D-marker
            float s_sum_x[stack_orig.getNz()],s_sum_y[stack_orig.getNz()];  // beam-induced motion
            int patch_N[stack_orig.getNz()];
            
            // loop: Nz (number of images)
            for(n=0;n<stack_orig.getNz();n++)
            {
                p_sum_x[n]=0.0;
                p_sum_y[n]=0.0;
                r_sum_x[n]=0.0;
                r_sum_y[n]=0.0;
                r_sum_z[n]=0.0;
                s_sum_x[n]=0.0;
                s_sum_y[n]=0.0;
                patch_N[n]=0;
            }
            // loop: number of tracks
            for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
            {
                if(it->getStatus()==true)
                {
                    // loop: length of track
                    for(t=0;t<it->getLength();t++)
                    {
                        if(it->getMarker2D_avail(t))
                        {
                            n=it->getMarker2D_n(t);
                            patch_N[n]++;
                            p_sum_x[n]+=it->getMarker2D_x(t);
                            p_sum_y[n]+=it->getMarker2D_y(t);
                            r_sum_x[n]+=it->getMarker3D_x();
                            r_sum_y[n]+=it->getMarker3D_y();
                            r_sum_z[n]+=it->getMarker3D_z();
                            /*if(beam_induced_motion)
                            {
                                s_sum_x[n]+=BI_para_x[n][0]+BI_para_x[n][1]*it->getMarker2D_x(t)+BI_para_x[n][2]*it->getMarker2D_x(t)*it->getMarker2D_x(t)+BI_para_x[n][3]*it->getMarker2D_y(t)+BI_para_x[n][4]*it->getMarker2D_y(t)*it->getMarker2D_y(t)+BI_para_x[n][5]*it->getMarker2D_x(t)*it->getMarker2D_y(t);
                                s_sum_y[n]+=BI_para_y[n][0]+BI_para_y[n][1]*it->getMarker2D_x(t)+BI_para_y[n][2]*it->getMarker2D_x(t)*it->getMarker2D_x(t)+BI_para_y[n][3]*it->getMarker2D_y(t)+BI_para_y[n][4]*it->getMarker2D_y(t)*it->getMarker2D_y(t)+BI_para_y[n][5]*it->getMarker2D_x(t)*it->getMarker2D_y(t);
                            }*/
                        }
                    }
                }
            }
            // loop: Nz (number of images)
            for(n=0;n<stack_orig.getNz();n++)
            {
                if(n==ref_n)    // keep the reference image unchanged (0 degree)
                {
    //                continue;
                }
                float psi_rad=psi,theta_rad=theta[n]/180*M_PI;
                Eigen::Matrix3f R_psi,R_theta;  // theta: tilt angle; psi: axis rotation
                Eigen::Matrix<float,2,3> P; // P: projection matrix {{1,0,0};{0,1,0}}
                R_psi.setZero();
                R_theta.setZero();
                P.setZero();
                P(0,0)=1;
                P(1,1)=1;
                R_psi << cos(-psi_rad),-sin(-psi_rad),0,sin(-psi_rad),cos(-psi_rad),0,0,0,1;
                R_theta << cos(theta_rad),0,-sin(theta_rad),0,1,0,sin(theta_rad),0,cos(theta_rad);
                A.resize(2,3);
                A=P*R_psi.transpose()*R_theta*R_psi;
                Eigen::Vector2f p_sum,s_sum;
                Eigen::Vector3f r_sum;
                p_sum << p_sum_x[n],p_sum_y[n];
                r_sum << r_sum_x[n],r_sum_y[n],r_sum_z[n];
                s_sum << s_sum_x[n],s_sum_y[n];
                Eigen::Vector2f d_new;
                if(patch_N[n]==0)
                {
                    reliable[n]=false;
                    d_x[n]=0.0;
                    d_y[n]=0.0;
                }
                else
                {
                    d_new=(p_sum-A*r_sum-s_sum)/patch_N[n];
                    d_x[n]=d_new(0);
                    d_y[n]=d_new(1);
                }
                cout << "Image " << n << ": (" << d_x[n] << "," << d_y[n] << ")" << endl;
            }

            // move to origin
            float d_x_orig=d_x[ref_n];
            float d_y_orig=d_y[ref_n];  // use 0 degree as reference
            cout << "Move to reference image" << endl;
            // loop: Nz (number of images)
            for(n=0;n<stack_orig.getNz();n++)
            {
                if(reliable[n])
                {
                    d_x[n]-=d_x_orig;
                    d_y[n]-=d_y_orig;
                }
                cout << "Image " << n << ": (" << d_x[n] << "," << d_y[n] << ")" << endl;
            }
            // loop: number of tracks
            for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
            {
                if(it->getStatus())
                {
                    // loop: length of track
                    for(t=0;t<it->getLength();t++)
                    {
                        float p_x=it->getMarker2D_x(t);
                        float p_y=it->getMarker2D_y(t);
                        it->setMarker2D(p_x-d_x_orig,p_y-d_y_orig,t);
                    }
                }
            }


            // optimize for 3D-marker
            int marker_N=0;
            // loop: number of tracks
            for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
            {
    //            if(it->getStatus()==true) // optimize for all 3D-markers
                {
                    tmp_1.setZero();
                    tmp_2.setZero();
                    Eigen::Matrix3f R_psi,R_theta;  // theta: tilt angle; psi: axis rotation
                    Eigen::Matrix<float,2,3> P; // P: projection matrix {{1,0,0};{0,1,0}}
                    R_psi.setZero();
                    R_theta.setZero();
                    P.setZero();
                    P(0,0)=1;
                    P(1,1)=1;
                    // loop: length of track
                    for(t=0;t<it->getLength();t++)
                    {
                        if(it->getMarker2D_avail(t))
                        {
                            n=it->getMarker2D_n(t);
                            R_psi.setZero();    // R_psi: [[cosd(-axis),-sind(-axis),0];[sind(-axis),cosd(-axis),0];[0,0,1]]
                            R_theta.setZero();  // R_theta: [[cosd(theta_est(n)),0,-sind(theta_est(n))];[0,1,0];[sind(theta_est(n)),0,cosd(theta_est(n))]]
                            float psi_rad=psi;
                            float theta_rad=theta[n]/180*M_PI;
                            R_psi << cos(-psi_rad),-sin(-psi_rad),0,sin(-psi_rad),cos(-psi_rad),0,0,0,1;
                            R_theta << cos(theta_rad),0,-sin(theta_rad),0,1,0,sin(theta_rad),0,cos(theta_rad);
                            float s_x=0.0;
                            float s_y=0.0;  // beam-induced motion
                            /*if(beam_induced_motion)
                            {
                                s_x=BI_para_x[n][0]+BI_para_x[n][1]*it->getMarker2D_x(t)+BI_para_x[n][2]*it->getMarker2D_x(t)*it->getMarker2D_x(t)+BI_para_x[n][3]*it->getMarker2D_y(t)+BI_para_x[n][4]*it->getMarker2D_y(t)*it->getMarker2D_y(t)+BI_para_x[n][5]*it->getMarker2D_x(t)*it->getMarker2D_y(t);
                                s_y=BI_para_y[n][0]+BI_para_y[n][1]*it->getMarker2D_x(t)+BI_para_y[n][2]*it->getMarker2D_x(t)*it->getMarker2D_x(t)+BI_para_y[n][3]*it->getMarker2D_y(t)+BI_para_y[n][4]*it->getMarker2D_y(t)*it->getMarker2D_y(t)+BI_para_y[n][5]*it->getMarker2D_x(t)*it->getMarker2D_y(t);
                            }*/
                            A.resize(2,3);
                            b.resize(2);
                            ATA.resize(3,3);
                            ATb.resize(3);
                            A=P*R_psi.transpose()*R_theta*R_psi;
                            ATA=A.transpose()*A;
                            b << (it->getMarker2D_x(t)-d_x[n]-s_x),(it->getMarker2D_y(t)-d_y[n]-s_y);
                            ATb=A.transpose()*b;
                            tmp_1=tmp_1+ATA;
                            tmp_2=tmp_2+ATb;
                        }
                    }
                    Eigen::Vector3f r_new;
                    r_new=tmp_1.inverse()*tmp_2;
                    it->setMarker3D(r_new(0),r_new(1),r_new(2));
                    cout << "Marker " << marker_N << ": (" << r_new(0) << "," << r_new(1) << "," << r_new(2) << ")" << endl;
                    marker_N++;
                }
            }

            // optimize for tilt angle (theta)
            const DP TOL=1.0e-4;
            DP ax,bx,cx,fa,fb,fc,xmin;
            
            cout << "theta:" << endl;
            // loop: Nz (number of images)
            for(n=0;n<stack_orig.getNz();n++)
            {
                ax=(theta[n]-0.1)/180*M_PI;
                bx=(theta[n])/180*M_PI;
                NR::mnbrak(ax,bx,cx,fa,fb,fc,loss_single,tracks,d_x,d_y,psi,n,BI_para_x[n],BI_para_y[n]);
                NR::golden(ax,bx,cx,loss_single,TOL,xmin,tracks,d_x,d_y,psi,n,BI_para_x[n],BI_para_y[n]);
                theta[n]=xmin*180/M_PI;
                cout << n << ": " << theta[n] << endl;
            }

            // remove wild points
    /*        cout << "Remove wild points:" << endl;
            int nn=0;
            for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
            {
                if(it->getStatus())
                {
    //                cout << "Track " << nn << ": " << endl;
                    Eigen::Matrix3f R_psi,R_theta;  // theta: tilt angle; psi: axis rotation
                    Eigen::Matrix<float,2,3> P; // P: projection matrix {{1,0,0};{0,1,0}}
                    Eigen::Vector3f r_now;
                    Eigen::Vector2f d_now,p_now;
                    R_psi.setZero();
                    R_theta.setZero();
                    P.setZero();
                    P(0,0)=1;
                    P(1,1)=1;
                    r_now.setZero();
                    r_now << it->getMarker3D_x(),it->getMarker3D_y(),it->getMarker3D_z();
                    for(t=0;t<it->getLength();t++)
                    {
                        if(it->getMarker2D_avail(t))
                        {
                            n=it->getMarker2D_n(t);
                            R_psi.setZero();    // R_psi: [[cosd(-axis),-sind(-axis),0];[sind(-axis),cosd(-axis),0];[0,0,1]]
                            R_theta.setZero();  // R_theta: [[cosd(theta_est(n)),0,-sind(theta_est(n))];[0,1,0];[sind(theta_est(n)),0,cosd(theta_est(n))]]
                            float psi_rad=psi,theta_rad=theta[n]/180*M_PI;
                            R_psi << cos(-psi_rad),-sin(-psi_rad),0,sin(-psi_rad),cos(-psi_rad),0,0,0,1;
                            R_theta << cos(theta_rad),0,-sin(theta_rad),0,1,0,sin(theta_rad),0,cos(theta_rad);
                            A.resize(2,3);
                            A=P*R_theta*R_psi;
                            d_now << d_x[n],d_y[n];
                            p_now=A*r_now+d_now;
                            float loss=(p_now(0)-it->getMarker2D_x(t))*(p_now(0)-it->getMarker2D_x(t))+(p_now(1)-it->getMarker2D_y(t))*(p_now(1)-it->getMarker2D_y(t));
        //                    cout << n << ": " << loss << endl;
    /*                        if(loss>100)
                            {
                                it->changeStatus2D(false,t);
                                patch_num_all--;
                            }
                        }
                    }
                    nn++;
                }
            }*/

            // align for beam-induced motion
            /*if(beam_induced_motion)
            {
                cout << "Performing alignment for beam-induced motion" << endl;
                
                int patch_N[stack_orig.getNz()];
                for(n=0;n<stack_orig.getNz();n++)
                {
                    patch_N[n]=0;
                }
                for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)    // look for the number of patches in each image
                {
        //            if(it->getStatus()==true)   // all patches are available for aligning beam-induced motion (not only linear)
                    {
                        for(t=0;t<it->getLength();t++)
                        {
        //                    if(it->getMarker2D_avail(t))
                            {
                                n=it->getMarker2D_n(t);
                                patch_N[n]++;
                            }
                        }
                    }
                }

                for(int n=0;n<stack_orig.getNz();n++)
                {
                    cout << n << ": ";
                    if(patch_N[n]>=6*1.5)   // 要求至少1.5倍超定
                    {
                        correct_BI[n]=true;

                        A.resize(patch_N[n],6);
                        b.resize(2);
                        Eigen::VectorXf b_x(patch_N[n]),b_y(patch_N[n]);
                        Eigen::Matrix<float,2,3> AA; // projection matrix with rotation
                        Eigen::Matrix3f R_psi,R_theta;  // theta: tilt angle; psi: axis rotation
                        Eigen::Matrix<float,2,3> P; // P: projection matrix {{1,0,0};{0,1,0}}
                        Eigen::Vector3f r_now;
                        Eigen::Vector2f d_now,p_now;
                        Eigen::VectorXf c_x(6),c_y(6);
                        P.setZero();
                        P(0,0)=1;
                        P(1,1)=1;
                        R_psi.setZero();
                        float psi_rad=psi;
                        R_psi << cos(-psi_rad),-sin(-psi_rad),0,sin(-psi_rad),cos(-psi_rad),0,0,0,1;
                        int nn=0;
                        for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
                        {
        //                    if(it->getStatus()==true)
                            {
                                for(t=0;t<it->getLength();t++)
                                {
        //                            if(it->getMarker2D_avail(t))
                                    {
                                        if(n==it->getMarker2D_n(t))
                                        {
                                            A(nn,0)=1;
                                            A(nn,1)=it->getMarker2D_x(t);
                                            A(nn,2)=it->getMarker2D_x(t)*it->getMarker2D_x(t);
                                            A(nn,3)=it->getMarker2D_y(t);
                                            A(nn,4)=it->getMarker2D_y(t)*it->getMarker2D_y(t);
                                            A(nn,5)=it->getMarker2D_x(t)*it->getMarker2D_y(t);

                                            float theta_rad=theta[n]/180*M_PI;
                                            R_theta.setZero();
                                            R_theta << cos(theta_rad),0,-sin(theta_rad),0,1,0,sin(theta_rad),0,cos(theta_rad);
                                            AA=P*R_psi.transpose()*R_theta*R_psi;

                                            r_now << it->getMarker3D_x(),it->getMarker3D_y(),it->getMarker3D_z();
                                            d_now << d_x[n],d_y[n];
                                            p_now << it->getMarker2D_x(t),it->getMarker2D_y(t);

                                            b=p_now-(AA*r_now+d_now);
                                            b_x(nn)=b(0);
                                            b_y(nn)=b(1);

                                            nn++;
                                        }
                                    }
                                }
                            }
                        }

                        ATA.resize(6,6);
                        ATb.resize(6);
                        ATA=A.transpose()*A;
                        ATb=A.transpose()*b_x;
                        c_x=ATA.inverse()*ATb;
                        ATb=A.transpose()*b_y;
                        c_y=ATA.inverse()*ATb;

                        for(int t=0;t<6;t++)
                        {
                            BI_para_x[n][t]=c_x(t);
                            BI_para_y[n][t]=c_y(t);
                        }

                        cout << c_x.transpose() << c_y.transpose() << endl;

                    }
                    else    // 直接输出原图
                    {
                        correct_BI[n]=false;
                        cout << "Skip!" << endl;
                    }
                }
            }*/

        }

        // print tilt axis angle
        cout << "psi: " << psi_deg << endl;
        cout << "Finish fine alignment!" << endl;



        // align for beam-induced motion
        if(beam_induced_motion)
        {
            cout << endl << "Performing alignment for beam-induced motion" << endl;
            
            int patch_N[stack_orig.getNz()];
            // loop: Nz (number of images)
            for(n=0;n<stack_orig.getNz();n++)
            {
                patch_N[n]=0;
            }
            // loop: number of tracks
            for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)    // look for the number of patches in each image
            {
    //            if(it->getStatus()==true)   // all patches are available for aligning beam-induced motion (not only linear)
                {
                    // loop: length of track
                    for(t=0;t<it->getLength();t++)
                    {
    //                    if(it->getMarker2D_avail(t))
                        {
                            n=it->getMarker2D_n(t);
                            patch_N[n]++;
                        }
                    }
                }
            }

            // loop: Nz (number of images)
            for(int n=0;n<stack_orig.getNz();n++)
            {
                cout << n << ": ";
                if(patch_N[n]>=6*1.5)   // 要求至少1.5倍超定
                {
                    correct_BI[n]=true;

                    A.resize(patch_N[n],6);
                    b.resize(2);
                    Eigen::VectorXf b_x(patch_N[n]),b_y(patch_N[n]);
                    Eigen::Matrix<float,2,3> AA; // projection matrix with rotation
                    Eigen::Matrix3f R_psi,R_theta;  // theta: tilt angle; psi: axis rotation
                    Eigen::Matrix<float,2,3> P; // P: projection matrix {{1,0,0};{0,1,0}}
                    Eigen::Vector3f r_now;
                    Eigen::Vector2f d_now,p_now;
                    Eigen::VectorXf c_x(6),c_y(6);
                    P.setZero();
                    P(0,0)=1;
                    P(1,1)=1;
                    R_psi.setZero();
                    float psi_rad=psi;
                    R_psi << cos(-psi_rad),-sin(-psi_rad),0,sin(-psi_rad),cos(-psi_rad),0,0,0,1;
                    int nn=0;
                    // loop: number of tracks
                    for(vector<Track>::iterator it=tracks.begin();it!=tracks.end();it++)
                    {
    //                    if(it->getStatus()==true)
                        {
                            // loop: length of track
                            for(t=0;t<it->getLength();t++)
                            {
    //                            if(it->getMarker2D_avail(t))
                                {
                                    if(n==it->getMarker2D_n(t))
                                    {
                                        A(nn,0)=1;
                                        A(nn,1)=it->getMarker2D_x(t);
                                        A(nn,2)=it->getMarker2D_x(t)*it->getMarker2D_x(t);
                                        A(nn,3)=it->getMarker2D_y(t);
                                        A(nn,4)=it->getMarker2D_y(t)*it->getMarker2D_y(t);
                                        A(nn,5)=it->getMarker2D_x(t)*it->getMarker2D_y(t);

                                        float theta_rad=theta[n]/180*M_PI;
                                        R_theta.setZero();
                                        R_theta << cos(theta_rad),0,-sin(theta_rad),0,1,0,sin(theta_rad),0,cos(theta_rad);
                                        AA=P*R_psi.transpose()*R_theta*R_psi;

                                        r_now << it->getMarker3D_x(),it->getMarker3D_y(),it->getMarker3D_z();
                                        d_now << d_x[n],d_y[n];
                                        p_now << it->getMarker2D_x(t),it->getMarker2D_y(t);

                                        b=p_now-(AA*r_now+d_now);
                                        b_x(nn)=b(0);
                                        b_y(nn)=b(1);

                                        nn++;
                                    }
                                }
                            }
                        }
                    }

                    ATA.resize(6,6);
                    ATb.resize(6);
                    ATA=A.transpose()*A;
                    ATb=A.transpose()*b_x;
                    c_x=ATA.inverse()*ATb;
                    ATb=A.transpose()*b_y;
                    c_y=ATA.inverse()*ATb;

                    for(int t=0;t<6;t++)
                    {
                        BI_para_x[n][t]=c_x(t);
                        BI_para_y[n][t]=c_y(t);
                    }

                    cout << c_x.transpose() << c_y.transpose() << endl;

                }
                else    // 直接输出原图
                {
                    cout << "Skip!" << endl;
                }
            }
        }
    }
    else
    {
        cout << endl << "Skip fine alignment!" << endl;
    }
    



    // write final aligned stack
    string output_mrc=prfx+".ali",output_mrc_BI=prfx+"_BI.ali";
    MRC stack_aligned(output_mrc.c_str(),"wb+");
    MRC stack_aligned_BI;
    stack_aligned.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);
    if(beam_induced_motion)
    {
        stack_aligned_BI.open(output_mrc_BI.c_str(),"wb+");
        stack_aligned_BI.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);
    }
    float *image_now_omp[stack_orig.getNz()];
    float *image_final_omp[stack_orig.getNz()];
    fftwf_plan plan_fft_omp[stack_orig.getNz()];
    fftwf_plan plan_ifft_omp[stack_orig.getNz()];
    float *bufc_omp[n];
    cout << endl << "Write final aligned stack:" << endl;
    
    // loop: Nz (number of images)
    for(n=0;n<stack_orig.getNz();n++)
    {
        image_now_omp[n]=new float [stack_orig.getNx()*stack_orig.getNy()];
        image_final_omp[n]=new float [stack_orig.getNx()*stack_orig.getNy()];
        stack_orig.read2DIm_32bit(image_now_omp[n],n);
        bufc_omp[n]=new float[(stack_orig.getNx()+2-stack_orig.getNx()%2)*stack_orig.getNy()];
        plan_fft_omp[n]=fftwf_plan_dft_r2c_2d(stack_orig.getNy(),stack_orig.getNx(),(float*)bufc_omp[n],reinterpret_cast<fftwf_complex*>(bufc_omp[n]),FFTW_ESTIMATE);
        plan_ifft_omp[n]=fftwf_plan_dft_c2r_2d(stack_orig.getNy(),stack_orig.getNx(),reinterpret_cast<fftwf_complex*>(bufc_omp[n]),(float*)bufc_omp[n],FFTW_ESTIMATE);
    }
    // loop: Nz (number of images)
    #pragma omp parallel for num_threads(threads)
    for(n=0;n<stack_orig.getNz();n++)
    {
        transform_image_omp(image_now_omp[n],image_final_omp[n],stack_orig.getNx(),stack_orig.getNy(),true,-float(stack_orig.getNx())/2.0,-float(stack_orig.getNy())/2.0,-psi,true,d_x[n],d_y[n],plan_fft_omp[n],plan_ifft_omp[n],bufc_omp[n]);
        if(!reliable[n])
        {
            cout << "No patch tracked in Image " << n << ", use coarse align result instead, but it may NOT be reliable!" << endl;
        }

        if(beam_induced_motion) // write out alignment for beam_induced motion
        {
            if(correct_BI[n])
            {
                transform_image_BI(image_now_omp[n],image_final_omp[n],stack_orig.getNx(),stack_orig.getNy(),-float(stack_orig.getNx())/2.0,-float(stack_orig.getNy())/2.0,-psi,d_x[n],d_y[n],BI_para_x[n],BI_para_y[n],true,1);
            }
            else
            {
                cout << "Not enough patches for aligning beam-induced motion in Image " << n << ", use fine align result instead!" << endl;
            }
        }
    }
    // loop: Nz (number of images)
    for(n=0;n<stack_orig.getNz();n++)
    {
        stack_aligned.write2DIm(image_final_omp[n],n);
        if(beam_induced_motion)
        {
            stack_aligned_BI.write2DIm(image_final_omp[n],n);
        }
    }
    cout << "Done!" << endl;
    stack_aligned.computeHeader_omp(pix*float(bin),true,threads);
    stack_aligned.close();
    if(beam_induced_motion)
    {
        stack_aligned_BI.computeHeader_omp(pix*float(bin),true,threads);
        stack_aligned_BI.close();
    }

    // write final tilt angles
    string file_tilt=prfx+".tlt";
    FILE *ftilt=fopen(file_tilt.c_str(),"w");
    for(n=0;n<stack_orig.getNz();n++)
    {
        fprintf(ftilt,"%f\n",theta[n]);
    }
    fflush(ftilt);
    fclose(ftilt);

    string path_psi=path+"/"+"psi.txt";
    FILE *fpsi=fopen(path_psi.c_str(),"w");
    fprintf(fpsi,"%f",psi_deg);
    fflush(fpsi);
    fclose(fpsi);

    // write out unrotated aligned stack
    bool unrotated_stack=false;
    it=inputPara.find("unrotated_stack");
    if(it!=inputPara.end())
    {
        unrotated_stack=atoi(it->second.c_str());
        cout << "Write unrotated aligned stack: " << unrotated_stack << endl;
    }
    else
    {
        cout << "No unrotated aligned stack option, set default: 0" << endl;
        unrotated_stack=false;
    }

    if(unrotated_stack)
    {
        cout << "Start writing unrotated aligned stack:" << endl;
        string unrotated_mrc=prfx+"_unrotated.ali";
        string unrotated_mrc_BI=prfx+"_unrotated_BI.ali";
        MRC stack_unrotated(unrotated_mrc.c_str(),"wb+");
        MRC stack_unrotated_BI;
        stack_unrotated.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);
        if(beam_induced_motion)
        {
            stack_unrotated_BI.open(unrotated_mrc_BI.c_str(),"wb+");
            stack_unrotated_BI.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);
        }
        // loop: Nz (number of images)
        #pragma omp parallel for num_threads(threads)
        for(n=0;n<stack_orig.getNz();n++)
        {
            transform_image_omp(image_now_omp[n],image_final_omp[n],stack_orig.getNx(),stack_orig.getNy(),false,-float(stack_orig.getNx())/2.0,-float(stack_orig.getNy())/2.0,-psi,true,d_x[n],d_y[n],plan_fft_omp[n],plan_ifft_omp[n],bufc_omp[n]);
            if(!reliable[n])
            {
                cout << "No patch tracked in Image " << n << ", use coarse align result instead, but it may NOT be reliable!" << endl;
            }

            if(beam_induced_motion) // write out alignment for beam_induced motion
            {
                if(correct_BI[n])
                {
                    transform_image_BI(image_now_omp[n],image_final_omp[n],stack_orig.getNx(),stack_orig.getNy(),-float(stack_orig.getNx())/2.0,-float(stack_orig.getNy())/2.0,-psi,d_x[n],d_y[n],BI_para_x[n],BI_para_y[n],false,1);
                }
                else
                {
                    cout << "Not enough patches for aligning beam-induced motion in Image " << n << ", use fine align result instead!" << endl;
                }
            }
        }
        // loop: Nz (number of images)
        for(n=0;n<stack_orig.getNz();n++)
        {
            stack_unrotated.write2DIm(image_final_omp[n],n);
            if(beam_induced_motion)
            {
                stack_unrotated_BI.write2DIm(image_final_omp[n],n);
            }
        }
        cout << "Done!" << endl;
        stack_unrotated.computeHeader_omp(pix*float(bin),true,threads);
        stack_unrotated.close();
        if(beam_induced_motion)
        {
            stack_unrotated_BI.computeHeader_omp(pix*float(bin),true,threads);
            stack_unrotated_BI.close();
        }
    }
    else
    {
        cout << "Skip writing unrotated aligned stack!" << endl;
    }

    for(n=0;n<stack_orig.getNz();n++)
    {
        delete [] image_now_omp[n];
        delete [] image_final_omp[n];
        fftwf_destroy_plan(plan_fft_omp[n]);
	    fftwf_destroy_plan(plan_ifft_omp[n]);
        delete [] bufc_omp[n];
    }
    delete [] image_now;
    delete [] image_next;
    stack_orig.close();
    
    // write original aligned stack (unbinned)
    if(bin!=1 && unbinned_stack)
    {
        stack_orig.open(original_mrc.c_str(),"rb");
        if(!stack_orig.hasFile())
        {
            cout << "Cannot open original mrc stack, skip writing unbinned aligned stack!" << endl;
        }
        else
        {
            string output_mrc_unbinned=prfx+"_unbinned.ali";
            string output_mrc_unbinned_BI=prfx+"_unbinned_BI.ali";
            MRC stack_unbinned(output_mrc_unbinned.c_str(),"wb+");
            MRC stack_unbinned_BI;
            stack_unbinned.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);
            if(beam_induced_motion)
            {
                stack_unbinned_BI.open(output_mrc_unbinned_BI.c_str(),"wb+");
                stack_unbinned_BI.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);
            }
            float *image_now_unbinned_omp[stack_orig.getNz()];
            float *image_move_unbinned_omp[stack_orig.getNz()];
            float *image_final_unbinned_omp[stack_orig.getNz()];
            cout << endl << "Write unbinned aligned stack:" << endl;
            for(n=0;n<stack_orig.getNz();n++)
            {
                image_now_unbinned_omp[n]=new float [stack_orig.getNx()*stack_orig.getNy()];
                image_move_unbinned_omp[n]=new float [stack_orig.getNx()*stack_orig.getNy()];
                image_final_unbinned_omp[n]=new float [stack_orig.getNx()*stack_orig.getNy()];
                stack_orig.read2DIm_32bit(image_now_unbinned_omp[n],n);
                bufc_omp[n]=new float[(stack_orig.getNx()+2-stack_orig.getNx()%2)*stack_orig.getNy()];
                plan_fft_omp[n]=fftwf_plan_dft_r2c_2d(stack_orig.getNy(),stack_orig.getNx(),(float*)bufc_omp[n],reinterpret_cast<fftwf_complex*>(bufc_omp[n]),FFTW_ESTIMATE);
                plan_ifft_omp[n]=fftwf_plan_dft_c2r_2d(stack_orig.getNy(),stack_orig.getNx(),reinterpret_cast<fftwf_complex*>(bufc_omp[n]),(float*)bufc_omp[n],FFTW_ESTIMATE);
            }
            #pragma omp parallel for num_threads(threads)
            for(n=0;n<stack_orig.getNz();n++)
            {
                move_image(image_now_unbinned_omp[n],image_move_unbinned_omp[n],stack_orig.getNx(),stack_orig.getNy(),-patch_dx_sum_all[n]*bin,-patch_dy_sum_all[n]*bin,true);
                transform_image_omp(image_move_unbinned_omp[n],image_final_unbinned_omp[n],stack_orig.getNx(),stack_orig.getNy(),true,-float(stack_orig.getNx())/2.0,-float(stack_orig.getNy())/2.0,-psi,true,d_x[n]*float(bin),d_y[n]*float(bin),plan_fft_omp[n],plan_ifft_omp[n],bufc_omp[n]);
                if(!reliable[n])
                {
                    cout << "No patch tracked in Image " << n << ", use coarse align result instead, but it may NOT be reliable!" << endl;
                }

                if(beam_induced_motion) // write out alignment for beam_induced motion
                {
                    if(correct_BI[n])
                    {
                        transform_image_BI(image_move_unbinned_omp[n],image_final_unbinned_omp[n],stack_orig.getNx(),stack_orig.getNy(),-float(stack_orig.getNx())/2.0,-float(stack_orig.getNy())/2.0,-psi,d_x[n],d_y[n],BI_para_x[n],BI_para_y[n],true,bin);
                    }
                    else
                    {
                        cout << "Not enough patches for aligning beam-induced motion in Image " << n << ", use fine align result instead!" << endl;
                    }
                }
            }
            for(n=0;n<stack_orig.getNz();n++)
            {
                stack_unbinned.write2DIm(image_final_unbinned_omp[n],n);
                if(beam_induced_motion)
                {
                    stack_unbinned_BI.write2DIm(image_final_unbinned_omp[n],n);
                }
            }
            cout << "Done!" << endl;
            stack_unbinned.computeHeader_omp(pix,true,threads);
            stack_unbinned.close();
            if(beam_induced_motion)
            {
                stack_unbinned_BI.computeHeader_omp(pix,true,threads);
                stack_unbinned_BI.close();
            }

            if(unrotated_stack)
            {
                cout << "Start writing unbinned unrotated aligned stack:" << endl;
                string unrotated_mrc=prfx+"_unbinned_unrotated.ali";
                string unrotated_mrc_BI=prfx+"_unbinned_unrotated_BI.ali";
                MRC stack_unrotated(unrotated_mrc.c_str(),"wb+");
                MRC stack_unrotated_BI;
                stack_unrotated.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);
                if(beam_induced_motion)
                {
                    stack_unrotated_BI.open(unrotated_mrc_BI.c_str(),"wb+");
                    stack_unrotated_BI.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);
                }
                #pragma omp parallel for num_threads(threads)
                for(n=0;n<stack_orig.getNz();n++)
                {
                    move_image(image_now_unbinned_omp[n],image_move_unbinned_omp[n],stack_orig.getNx(),stack_orig.getNy(),-patch_dx_sum_all[n]*bin,-patch_dy_sum_all[n]*bin,true);
                    transform_image_omp(image_move_unbinned_omp[n],image_final_unbinned_omp[n],stack_orig.getNx(),stack_orig.getNy(),false,-float(stack_orig.getNx())/2.0,-float(stack_orig.getNy())/2.0,-psi,true,d_x[n]*float(bin),d_y[n]*float(bin),plan_fft_omp[n],plan_ifft_omp[n],bufc_omp[n]);
                    if(!reliable[n])
                    {
                        cout << "No patch tracked in Image " << n << ", use coarse align result instead, but it may NOT be reliable!" << endl;
                    }

                    if(beam_induced_motion) // write out alignment for beam_induced motion
                    {
                        if(correct_BI[n])
                        {
                            transform_image_BI(image_move_unbinned_omp[n],image_final_unbinned_omp[n],stack_orig.getNx(),stack_orig.getNy(),-float(stack_orig.getNx())/2.0,-float(stack_orig.getNy())/2.0,-psi,d_x[n],d_y[n],BI_para_x[n],BI_para_y[n],false,bin);
                        }
                        else
                        {
                            cout << "Not enough patches for aligning beam-induced motion in Image " << n << ", use fine align result instead!" << endl;
                        }
                    }
                }
                for(n=0;n<stack_orig.getNz();n++)
                {
                    stack_unrotated.write2DIm(image_final_unbinned_omp[n],n);
                    if(beam_induced_motion)
                    {
                        stack_unrotated_BI.write2DIm(image_final_unbinned_omp[n],n);
                    }
                }
                cout << "Done!" << endl;
                stack_unrotated.computeHeader_omp(pix,true,threads);
                stack_unrotated.close();
                if(beam_induced_motion)
                {
                    stack_unrotated_BI.computeHeader_omp(pix,true,threads);
                    stack_unrotated_BI.close();
                }
            }
            else
            {
                cout << "Skip writing unbinned unrotated aligned stack!" << endl;
            }

            for(n=0;n<stack_orig.getNz();n++)
            {
                delete [] image_now_unbinned_omp[n];
                delete [] image_move_unbinned_omp[n];
                delete [] image_final_unbinned_omp[n];
                fftwf_destroy_plan(plan_fft_omp[n]);
                fftwf_destroy_plan(plan_ifft_omp[n]);
                delete [] bufc_omp[n];
            }
            stack_orig.close();
        }
        
    }
    else
    {
        cout << "Skip writing unbinned aligned stack!" << endl;
    }
    

    cout << endl << "Finish all alignment procedures successfully!" << endl;
    cout << "All results save in: " << path << endl << endl;

}
