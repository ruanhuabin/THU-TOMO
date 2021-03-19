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
    const char *paraFileName = "../conf/para_extract.conf";
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
        cerr << "No output file name, set default: tomo_extract.st" << endl;
        output_mrc="tomo_extract.st";
    }

    int start_index,end_index;
    it=inputPara.find("start_index");
    if(it!=inputPara.end())
    {
        start_index=atoi(it->second.c_str());
        cout << "Start index: " << start_index << endl;
    }
    else
    {
        cout << "No start index, set default: 1" << endl;
        start_index=1;
    }
    it=inputPara.find("end_index");
    if(it!=inputPara.end())
    {
        end_index=atoi(it->second.c_str());
        cout << "End index: " << end_index << endl;
    }
    else
    {
        cout << "No end index, set default: " << stack_orig.getNz() << endl;
        end_index=stack_orig.getNz();
    }

    // extract
    if(start_index==1 && end_index==stack_orig.getNz())
    {
        cout << "Extract full stack, skip processing!" << endl;
    }
    else
    {
        cout << "Perform extracting:" << endl;
        MRC stack_extract(output_mrc.c_str(),"wb");
        stack_extract.createMRC_empty(stack_orig.getNx(),stack_orig.getNy(),end_index-start_index+1,2);
        float *image_now=new float[stack_orig.getNx()*stack_orig.getNy()];
        for(int n=start_index-1;n<end_index;n++)
        {
            cout << n << ": ";
            stack_orig.read2DIm_32bit(image_now,n);
            stack_extract.write2DIm(image_now,n-start_index+1);
            cout << "Done" << endl;
        }
        delete [] image_now;
        stack_extract.close();
    }

    stack_orig.close();

    getTime(start_time);

    return 0;
}
