#include <stdio.h>
#include "util.h"
#include "mrc.h"
#include "time.h"
#include "math.h"

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
    const char *paraFileName = "../conf/para_flipyz.conf";
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
        cerr << "No output file name, set default: tomo_flipped.st" << endl;
        output_mrc="tomo_flipped.st";
    }
    MRC stack_flipped(output_mrc.c_str(),"wb");
    stack_flipped.createMRC_empty(stack_orig.getNx(),stack_orig.getNz(),stack_orig.getNy(),2);

    // flipyz
    cout << endl << "Flipping y-axis and z-axis:" << endl;
    float *strip_now=new float[stack_orig.getNx()];
    for(int k=0;k<stack_orig.getNz();k++)
    {
        for(int j=0;j<stack_orig.getNy();j++)
        {
            stack_orig.readLine(strip_now,k,j);
            stack_flipped.writeLine(strip_now,j,k);
        }
    }
    delete [] strip_now;
    cout << "Finish flipping!" << endl;

    stack_orig.close();
    stack_flipped.close();

    getTime(start_time);

    return 0;
}
