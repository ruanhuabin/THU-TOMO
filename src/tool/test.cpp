/*******************************************************************
 *       Filename:  test.cpp                                     
 *                                                                 
 *    Description:                                        
 *                                                                 
 *        Version:  1.0                                            
 *        Created:  07/10/2020 10:07:06 AM                                 
 *       Revision:  none                                           
 *       Compiler:  gcc                                           
 *                                                                 
 *         Author:  Ruan Huabin                                      
 *          Email:  ruanhuabin@tsinghua.edu.cn                                        
 *        Company:  Dep. of CS, Tsinghua Unversity                                      
 *                                                                 
 *******************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N 100000000

int main ( int argc, char *argv[] )
{ 

    int tn = atoi(argv[1]);
    float *a = (float *)malloc(sizeof(float) * N);
    float *b = (float *)malloc(sizeof(float) * N);
    double c1 = 0.0f;
    double c2 = 0.0f;
    double c3 = 0.0f;

    int i = 0;
#pragma omp parallel for num_threads(tn) private(i) reduction(+:c1,c2,c3)
    for(i = 0; i < N; i ++)
    {
        a[i] = (float)(i + 0.5853);
        b[i] = (float)(i + 0.6048);
        c1 += a[i] * b[i];
        c2 += a[i] * a[i];
        c3 += b[i] * b[i];
    }

    printf("c1 = %f, c2 = %f, c3 = %f\n", c1, c2, c3);
    free(a);
    free(b);
    return EXIT_SUCCESS;
}

