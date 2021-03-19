/*******************************************************************
 *       Filename:  tomo.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/15/2020 05:48:48 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:
 *          Email:
 *        Company:
 *
 *******************************************************************/
#include <stdio.h>
#include "util.h"
#include "mrc.h"
#include "AlignmentAlgo.h"
#include "CTFAlgo.h"
#include "ReconstructionAlgo_WBP.h"
#include "ReconstructionAlgo_WBP_RAM.h"
#include "ReconstructionAlgo_SIRT.h"
#include "ReconstructionAlgo_SIRT_RAM.h"
#include "time.h"

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

    if(argc != 2)
    {
        fprintf(stderr, "\n  Usage: \n%6s%s /path/to/parameter/file\n\n", "", argv[0]);
        exit(1);
    }



    time_t start_time;
    time(&start_time);

    map<string, string> inputPara;
    map<string, string> outputPara;
    const char *paraFileName = argv[1];
    readParaFile(inputPara, paraFileName);
    getAllParas(inputPara);

    bool do_alignment=0,do_CTF=0,do_reconstruction_WBP=0,do_reconstruction_WBP_in_RAM=0,do_reconstruction_SIRT=0,do_reconstruction_SIRT_in_RAM=0;
    map<string,string>::iterator it=inputPara.find("do_alignment");
    if(it!=inputPara.end())
    {
        do_alignment=atoi(it->second.c_str());
    }
    it=inputPara.find("do_CTF");
    if(it!=inputPara.end())
    {
        do_CTF=atoi(it->second.c_str());
    }
    it=inputPara.find("do_reconstruction_WBP");
    if(it!=inputPara.end())
    {
        do_reconstruction_WBP=atoi(it->second.c_str());
    }
    it=inputPara.find("do_reconstruction_WBP_in_RAM");
    if(it!=inputPara.end())
    {
        do_reconstruction_WBP_in_RAM=atoi(it->second.c_str());
    }
    it=inputPara.find("do_reconstruction_SIRT");
    if(it!=inputPara.end())
    {
        do_reconstruction_SIRT=atoi(it->second.c_str());
    }
    it=inputPara.find("do_reconstruction_SIRT_in_RAM");
    if(it!=inputPara.end())
    {
        do_reconstruction_SIRT_in_RAM=atoi(it->second.c_str());
    }

    if(do_alignment)
    {
        cout << endl << "Do Alignment!" << endl << endl;
        AlignmentBase *ab = new AlignmentAlgo();
        ab->doAlignment(inputPara, outputPara);
        delete ab;
    }

    if(do_CTF)
    {
        cout << endl << "Do CTF!" << endl << endl;
        CTFBase *cb = new CTFAlgo();
        cb->doCTF(inputPara, outputPara);
        delete cb;
    }

    if(do_reconstruction_WBP)
    {
        cout << endl << "Do Reconstruction with WBP!" << endl << endl;
        ReconstructionBase *rb = new ReconstructionAlgo_WBP();
        rb->doReconstruction(inputPara, outputPara);
        delete rb;
    }

    if(do_reconstruction_WBP_in_RAM)
    {
        cout << endl << "Do Reconstruction with WBP in RAM!" << endl << endl;
        ReconstructionBase *rb = new ReconstructionAlgo_WBP_RAM();
        rb->doReconstruction(inputPara, outputPara);
        delete rb;
    }

    if(do_reconstruction_SIRT)
    {
        cout << endl << "Do Reconstruction with SIRT!" << endl << endl;
        ReconstructionBase *rb = new ReconstructionAlgo_SIRT();
        rb->doReconstruction(inputPara, outputPara);
        delete rb;
    }

    if(do_reconstruction_SIRT_in_RAM)
    {
        cout << endl << "Do Reconstruction with SIRT in RAM!" << endl << endl;
        ReconstructionBase *rb = new ReconstructionAlgo_SIRT_RAM();
        rb->doReconstruction(inputPara, outputPara);
        delete rb;
    }

    cout << endl << "Finish!" << endl;
    getTime(start_time);

    return 0;
}
