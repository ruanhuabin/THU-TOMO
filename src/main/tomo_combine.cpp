#include <stdio.h>
#include "util.h"
#include "mrc.h"
#include "time.h"
#include "math.h"

// 默认扩大到最大的mrc尺寸，均值padding

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
    const char *paraFileName = "../conf/para_combine.conf";
    readParaFile(inputPara, paraFileName);
    getAllParas(inputPara);

    // read input parameters
    map<string,string>::iterator it=inputPara.find("input_mrc_all");
    string input_mrc_all;
    if(it!=inputPara.end())
    {
        input_mrc_all=it->second;
        cout << "All input file name in txt: " << input_mrc_all << endl;
    }
    else
    {
        cerr << "No input file name!" << endl;
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
        cerr << "No output file name, set default: tomo_combine.st" << endl;
        output_mrc="tomo_combine.st";
    }

    bool is_contract=0;
    bool is_zero_padding=0;
    it=inputPara.find("contract");
    if(it!=inputPara.end())
    {
        is_contract=atoi(it->second.c_str());
    }
    if(is_contract) // contract
    {
        cout << "Contract to the minimum size" << endl;
    }
    else    // expand
    {
        cout << "Expand to the maximum size" << endl;
        it=inputPara.find("zero_padding");
        if(it!=inputPara.end())
        {
            is_zero_padding=atoi(it->second.c_str());
        }
        if(is_zero_padding)
        {
            cout << "Expand with zero padding" << endl;
        }
        else
        {
            cout << "Expand with average padding" << endl;
        }
    }

    // combine
    ifstream finput(input_mrc_all.c_str());
    if(!finput)
    {
        cerr << "Cannot open input file!" << endl;
        abort();
    }
    
    // search for the final size
    cout << "Search for output size:" << endl;
    int Nx_final,Ny_final,Nz_final=0;
    string input_mrc;
    MRC stack_now;
    int n=0;
    while(getline(finput,input_mrc))
    {
//        cout << input_mrc << endl;
        stack_now.open(input_mrc.c_str(),"rb");
        if(!stack_now.hasFile())
        {
            cerr << "Cannot open input mrc: " << input_mrc << endl;
            abort();
        }
        if(n==0)
        {
            Nx_final=stack_now.getNx();
            Ny_final=stack_now.getNy();
        }
        else
        {
            if(is_contract) // search for min
            {
                if(Nx_final>stack_now.getNx())
                {
                    Nx_final=stack_now.getNx();
                }
                if(Ny_final>stack_now.getNy())
                {
                    Ny_final=stack_now.getNy();
                }
            }
            else    // search for max
            {
                if(Nx_final<stack_now.getNx())
                {
                    Nx_final=stack_now.getNx();
                }
                if(Ny_final<stack_now.getNy())
                {
                    Ny_final=stack_now.getNy();
                }
            }
        }
        Nz_final+=stack_now.getNz();
        n++;
        stack_now.close();
    }
    cout << "Done" << endl;
    cout << "Output size: (" << Nx_final << "," << Ny_final << "," << Nz_final << ")" << endl;

    // perform combining
    MRC stack_combine(output_mrc.c_str(),"wb");
    stack_combine.createMRC_empty(Nx_final,Ny_final,Nz_final,2);
    float *image_combine=new float[Nx_final*Ny_final];
    finput.clear();
    finput.seekg(0);
    n=0;
    while(getline(finput,input_mrc))
    {
        cout << input_mrc << ":" << endl;
        stack_now.open(input_mrc.c_str(),"rb");
        float *image_now=new float[stack_now.getNx()*stack_now.getNy()];
        for(int k=0;k<stack_now.getNz();k++)
        {
            cout << k << ": ";
            stack_now.read2DIm_32bit(image_now,k);
            if(is_contract) // contract
            {
                int x_start_offset=(stack_now.getNx()-Nx_final)/2,y_start_offset=(stack_now.getNy()-Ny_final)/2;
                for(int j=0;j<Ny_final;j++)
                {
                    memcpy(image_combine+j*Nx_final,image_now+(y_start_offset+j)*stack_now.getNx()+x_start_offset,sizeof(float)*Nx_final);
                }
            }
            else    // expand
            {
                float avg=0.0;
                if(!is_zero_padding)
                {
                    for(int i=0;i<stack_now.getNx()*stack_now.getNy();i++)  // calculate average for padding
                    {
                        avg+=image_now[i];
                    }
                    avg/=(stack_now.getNx()*stack_now.getNy());
                }
                for(int i=0;i<Nx_final*Ny_final;i++)    // padding value as initial value
                {
                    image_combine[i]=avg;
                }
                int x_start_offset=(Nx_final-stack_now.getNx())/2,y_start_offset=(Ny_final-stack_now.getNy())/2;
                for(int j=0;j<stack_now.getNy();j++)
                {
                    memcpy(image_combine+(y_start_offset+j)*Nx_final+x_start_offset,image_now+j*stack_now.getNx(),sizeof(float)*stack_now.getNx());
                }
            }
            stack_combine.write2DIm(image_combine,n);
            n++;
            cout << "Done" << endl;
        }
        delete [] image_now;
        stack_now.close();
    }

    finput.close();
    stack_combine.close();
    delete [] image_combine;

    getTime(start_time);

    return 0;
}
