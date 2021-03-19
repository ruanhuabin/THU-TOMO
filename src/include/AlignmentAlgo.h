/*******************************************************************
 *       Filename:  AlignmentAlgo.h                                     
 *                                                                 
 *    Description:                                         
 *                                                                 
 *        Version:  1.0                                            
 *        Created:  07/07/2020 05:13:22 PM                                 
 *       Revision:  none                                           
 *       Compiler:  gcc                                           
 *                                                                 
 *         Author:  Ruan Huabin                                      
 *          Email:  ruanhuabin@tsinghua.edu.cn                                        
 *        Company:  Dep. of CS, Tsinghua Unversity                                      
 *                                                                 
 *******************************************************************/

#ifndef  ALIGNMENTALGO_H
#define  ALIGNMENTALGO_H

#include "AlignmentBase.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "GPUInterface.h"

using namespace std;
class AlignmentAlgo:public AlignmentBase
{
    public:
        void doAlignment(map<string, string> &inputPara, map<string, string> &outputPara);
        ~AlignmentAlgo();

};

#endif

