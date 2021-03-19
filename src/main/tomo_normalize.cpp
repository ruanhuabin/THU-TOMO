#include <stdio.h>
#include "util.h"
#include "mrc.h"
#include "time.h"
#include "math.h"
#include "fftw3.h"

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



int main(int argc, char **argv)
{
    time_t start_time;
    time(&start_time);

    map<string, string> inputPara;
    map<string, string> outputPara;
    const char *paraFileName = "../conf/para_normalize.conf";
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
        cerr << "No output file name, set default: tomo_normalize.st" << endl;
        output_mrc="tomo_normalize.st";
    }



    // normalize image
    cout << endl << "Performing normalization:" << endl;

    MRC stack_normalize(output_mrc.c_str(),"wb");
    float *image_now=new float[stack_orig.getNx()*stack_orig.getNy()];
    stack_normalize.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),stack_orig.getNz(),2);
    
    cout << "Scan for maximum mean value:" << endl;
    double max_mean=0.0;
    double image_mean[stack_orig.getNz()];
    for(int n=0;n<stack_orig.getNz();n++)
    {
        cout << n << ": ";
        image_mean[n]=0.0;
        stack_orig.read2DIm_32bit(image_now,n);
        for(int i=0;i<stack_orig.getNx()*stack_orig.getNy();i++)
        {
            image_mean[n]+=double(image_now[i]);
        }
        image_mean[n]/=(stack_orig.getNx()*stack_orig.getNy());
        if(image_mean[n]>max_mean)
        {
            max_mean=image_mean[n];
        }
        cout << image_mean[n] << " ";
        cout << "Done" << endl;
    }
    cout << "Done" << endl;
    
    cout << "Normalize all the means to the maximum:" << endl;
    for(int n=0;n<stack_orig.getNz();n++)
    {
        cout << n << ": ";
        stack_orig.read2DIm_32bit(image_now,n);
        for(int i=0;i<stack_orig.getNx()*stack_orig.getNy();i++)
        {
            image_now[i]=double(image_now[i])/image_mean[n]*max_mean;
        }
        stack_normalize.write2DIm(image_now,n);
        cout << "Done" << endl;
    }
    cout << "Done" << endl;

    delete [] image_now;
    stack_normalize.close();
    cout << "Finish normalization!" << endl;

    stack_orig.close();

    getTime(start_time);

    return 0;
}
