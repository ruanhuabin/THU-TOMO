/*******************************************************************
 *       Filename:  gputest.cpp                                     
 *                                                                 
 *    Description:                                        
 *                                                                 
 *        Version:  1.0                                            
 *        Created:  2020年10月29日 22时42分07秒                                 
 *       Revision:  none                                           
 *       Compiler:  gcc                                           
 *                                                                 
 *         Author:  Ruan Huabin                                      
 *          Email:  ruanhuabin@tsinghua.edu.cn                                        
 *        Company:  Dep. of CS, Tsinghua Unversity                                      
 *                                                                 
 *******************************************************************/
#ifdef GPU_VERSION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
//#include "sumArray.h"
#include "GPUInterface.h"

int main ( int argc, char *argv[] )
{ 
    int nElem = 1024;
    if(argc == 2)
    {
    
        nElem = atoi(argv[1]);
    }
    //cuTomo::sumArray(nElem);
    sumArray(nElem);
    return EXIT_SUCCESS;
}

#endif
