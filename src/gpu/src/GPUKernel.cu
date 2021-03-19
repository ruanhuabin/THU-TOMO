#include "GPUKernel.cuh"
//To remove
#include "sumArray.h"

namespace cuTomo {

int cudaInit(
	vector<int>& iGPU,		//[inout]
	vector<void*>& stream 	//[inout]
)
{
	CHECK(cudaDeviceReset());
	//cudaError_t result = cudaSetDevice(1);
	int deviceNum;
	CHECK(cudaGetDeviceCount(&deviceNum));
	std::cout << "Device Number: " << deviceNum << std::endl;
	for (int i = 0; i < deviceNum; i++)
	{
		iGPU.push_back(i);
		for (int j = 0; j < NUM_STREAM_PER_DEVICE; j++)
		{
			cudaStream_t* newStream = new cudaStream_t;
			CHECK(cudaStreamCreate(newStream));
			stream.push_back((void*)newStream);
		}
	}

	return deviceNum;
}

void cudaEndUp(vector<int>& iGPU,
	vector<void*>& stream)
{
	
	for (int j = 0; j < stream.size(); j++)
	{
		CHECK(cudaStreamDestroy(*((cudaStream_t*)(stream[j]))));
		delete (cudaStream_t*)(stream[j]);
	}	
	
	/*关于cudaDeviceReset的使用还需要再研究一下 与current device和current process都有关
	for (int i = 0; i < iGPU.size(); i++)
	{
		cudaDeviceReset();
	}
	*/
}

void volumeProj(
    bool skip_ctfcorrection,
    bool skip_3dctf,
    //MRC stack_orig,
    int stack_orig_Nx,
    int stack_orig_Ny,
    //CTF ctf_para_n,     //"n" refers to the n-th image out of Nz overall
    float *stack_recon[],
    float *proj_now,
    float psi_rad,
    float theta_rad,      
    int h_tilt_max,
    int h,
    //int threads,         //To be removed
    //fftwf_plan *plan_fft,
    //fftwf_plan *plan_ifft,
    // float *proj_omp[],
    // float *bufc[],
    float x_offset,
    float y_offset,
	float z_offset,
	vector<int>& iGPU,
    vector<void*>& stream
)
{
	//Test1 单GPU 单stream
	int blockNum = h_tilt_max/2;
	//注意CUDA中 每个block的size是有限制的，Tesla K20为(1024, 1024, 64).这里的限制是指每个维度上的限制,单个block线程数量限制为1024
	//grid size也是有限制的，但是会比block size大很多，Tesla K20为(2147483647, 65535, 65535)
	//int threadNum = stack_orig_Nx * stack_orig_Ny;
	int threadX = 0;
	int threadY = 0;

	//注意 这里会受到硬件 Maximum number of threads per block:1024的限制.
	//所以threadX*threadY要<=1024
	if(stack_orig_Nx > THREADSMAXX)
	{
		threadX = THREADSMAXX;
	}
	else
	{
		threadX = stack_orig_Nx;
	}
	if(stack_orig_Ny > THREADSMAXY)
	{
		threadY = THREADSMAXY;
	}
	else
	{
		threadY = stack_orig_Ny;
	}

	dim3 block(threadX, threadY);
	
	CHECK(
		//TODO
		cudaSetDevice(iGPU[0])
	);

	float *devStackRecon = NULL;
	float *devProj = NULL;

	std::cout<<"[CUTOMO] Device set"<<std::endl;
	CHECK(
		cudaMalloc(
			(void**)&devStackRecon, 
			h * stack_orig_Nx * stack_orig_Ny * sizeof(float)
		)
	);

	//三条流同时拷贝，将stack_recon中的数据拷贝到devStackRecon中
	for(int i = 0; i < h; i++)
	{
		CHECK(
			cudaMemcpyAsync(
				&(devStackRecon[i * stack_orig_Nx * stack_orig_Ny]),
				&(stack_recon[i][0]),
				stack_orig_Nx * stack_orig_Ny * sizeof(float),
				cudaMemcpyHostToDevice,
				*((cudaStream_t*)stream[i % NUM_STREAM_PER_DEVICE])
			)
		)
	}

	//TODO：测试一下异步多流拷贝与默认拷贝方式之间的效率区别
	// CHECK(
	// 	cudaMemcpy(
	// 		devStackRecon, 
	// 		&(stack_recon[0][0]), 
	// 		h_tilt_max*stack_orig_Nx*stack_orig_Ny*sizeof(float),
	// 		cudaMemcpyHostToDevice)
	// );
	// CHECK(
	// 	cudaMemcpyAsync(
	// 		&(devStackRecon[0 * stack_orig_Nx * stack_orig_Ny]),
	// 		&(stack_recon[0][0]),
	// 		stack_orig_Nx * stack_orig_Ny * sizeof(float),
	// 		cudaMemcpyHostToDevice,
	// 		*((cudaStream_t*)stream[0])
	// 	)
	// );

	//同步所有stream
	//ToOpti：怀疑会对性能产生影响。
	for(int i = 0; i < NUM_STREAM_PER_DEVICE;i++)
	{
		CHECK(
			cudaStreamSynchronize(*((cudaStream_t*)stream[i]))
		);
	}

	//TODO:在device上创建h_tilt_max个一维数组（本质是一个二维数组）
	//笔记：CUDA不适合二维数组，会比较慢，应该使用一维数组，然后用偏移量来访问。
	//将这个数组置全0
	std::cout<<"[CUTOMO] threadX:"<<threadX<<" threadY:"<<threadY<<"h_tilt_max:"<<h_tilt_max<<std::endl;
	
	CHECK(
		cudaMalloc(
			(void**)&devProj,
			blockNum * stack_orig_Nx * stack_orig_Ny * sizeof(float)
		)
	);

	CHECK(
		cudaMemset(
			(void*)devProj,
			0,
			blockNum * stack_orig_Nx * stack_orig_Ny * sizeof(float)
		)
	);

	kernel_volumeProj<<<
		blockNum,
		block,
		0,
		0
	>>>(
		devStackRecon,
		devProj,
		stack_orig_Nx,
		stack_orig_Ny,
		psi_rad,
		theta_rad,      
		h_tilt_max,
		h,
		// *proj_omp[],
		// *bufc[],
		x_offset,
		y_offset,
		z_offset
	);
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());

	float *tmpProjSum;
	CHECK(
		cudaMalloc(
			(void**)&tmpProjSum, 
			stack_orig_Nx * stack_orig_Ny * sizeof(float)
		)
	)
	CHECK(
		cudaMemset(
			(void*)tmpProjSum,
			0,
			stack_orig_Nx * stack_orig_Ny * sizeof(float)
		)
	);


	kernel_projAccu<<<
		blockNum,
		block,
		0,
		0
	>>>(
		devProj,
		tmpProjSum,
		stack_orig_Nx,
		stack_orig_Ny,
		h_tilt_max
	);
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(
		cudaMemcpy(
			proj_now,
			tmpProjSum,
			stack_orig_Nx * stack_orig_Ny * sizeof(float),
			cudaMemcpyDeviceToHost
		)
	);

	CHECK(
		cudaFree((void*)devStackRecon)
	);

	CHECK(
		cudaFree((void*)devProj)
	);

	CHECK(
		cudaFree((void*)tmpProjSum)
	);



	//TODO: 写程多GPU 多stream 并且要测试效果 有多大的提升 提升在哪里。
	// int nGPU = stream.size()/NUM_STREAM_PER_DEVICE;
	// for(int GPUIdx = 0; GPUIdx < nGPU; GPUIdx++)
	// {
	// 	//ToOpti 有点多此一举？THUNDER中iGPU的作用是什么？在CUDA中其实可以直接在函数内调用cudaGetDeviceCount获取GPU的数量，其实循环也完全可以直接根据这个数量开始？
	// 	//甚至可能有好处：如果有一个GPU不可用了，代码也可以根据可用GPU数量自动调整分配任务，反而如果在一开始就获取GPU数量，后续出了问题怎么办？
	// 	//这里感觉是潜在的隐患，虽然可能很难触发到。需要在整个project中统一。
	// 	//可能有机器会出现GPU device number是0，1，3，5，7这样不规则排列？或者有无效的节点？
	// 	//具体可能会出问题的function：cudaSetDevice。
	// 	CHECK(cudaSetDevice(iGPU[GPUIdx]));

	// 	for(int stmIdx = 0; stmIdx < NUM_STREAM_PER_DEVICE; stmIdx++)
	// 	{
	// 		CHECK()
	// 	}

	// }
}



}
