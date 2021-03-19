#ifdef GPU_VERSION
#include "GPUInterface.h"
#include <iostream>
int main()
{

    std::vector<void*> stream;
	std::vector<int> iGPU;
    int _nGPU = InitGPU(iGPU, stream);
    
    if(_nGPU == 0)
    {
        return 0;
    }

    std::cout<<"cuda配置成功！\n" ;
    freeGPU(iGPU, stream);
    return 0;
    
}
#endif
