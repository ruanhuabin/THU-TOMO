/*******************************************************************
 *       Filename:  ReconstructionAlgo.h                                     
 *                                                                 
 *    Description:                                         
 *                                                                 
 *        Version:  1.0                                            
 *        Created:  07/07/2020 05:35:18 PM                                 
 *       Revision:  none                                           
 *       Compiler:  gcc                                           
 *                                                                 
 *         Author:  Ruan Huabin                                      
 *          Email:  ruanhuabin@tsinghua.edu.cn                                        
 *        Company:  Dep. of CS, Tsinghua Unversity                                      
 *                                                                 
 *******************************************************************/

#ifndef  RECONSTRUCTIONALGO_SIRT_H
#define  RECONSTRUCTIONALGO_SIRT_H
#include "ReconstructionBase.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "GPUInterface.h"

using namespace std;
class ReconstructionAlgo_SIRT:public ReconstructionBase
{
    public:
        void doReconstruction(map<string, string> &inputPara, map<string, string> &outputPara);
        ~ReconstructionAlgo_SIRT();

};


#endif
