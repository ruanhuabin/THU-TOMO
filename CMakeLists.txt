# @file CMakeLists.txt
# @author Fande Yu 
# @version 0.1 
# @copyright 
#
# ChangeLog
# AUTHOR      | TIME       | VERSION       | DESCRIPTION
# ------      | ----       | -------       | -----------
# Fande Yu    | 2020/10/14 | 0.1.20201014  | Create this file 
# Fande Yu	  | 2020/12/26 | 0.1.20201226  | Build Seperatedly

#Notice
#1. export CC=gcc and CXX=g++ ahead of running this CMake file.
#2. If this program will be run on multi-nodes where NCCL may be necessary, this file need to be modified.

# CMAKE REQUIREMENT
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(TomoProject)

set(TomoProject_VERSION_MAJOR 0)
set(TomoProject_VERSION_MINOR 1)
set(TomoProject_VERSION_ADDIT 20201014)

#Get git commit
execute_process(
    COMMAND git rev-parse HEAD 
    OUTPUT_VARIABLE COMMIT_VERSION 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Welcome to TomoProject ${TomoProject_VERSION_MAJOR}.${TomoProject_VERSION_MINOR}.${TomoProject_VERSION_ADDIT}!")

message(STATUS "Git commit version:" ${COMMIT_VERSION})

#For TomoConfig.h.in
set(COMMIT_VERSION_QUOTE "\"${COMMIT_VERSION}\"")

#set(CMAKE_C_COMPILER gcc)
#set(CMAKE_CXX_COMPILER g++)
set(COMMON_FLAGS "-std=c++11 ${COMMON_FLAGS}")

#For FFTW
option(SINGLE_PRECISION "Whether to use single float precision in calculation?" ON)
option(ENABLE_SIMD "Whether to use SIMD to accelerate?" ON)
option(ENABLE_AVX256 "Whether to use AVX256 to accelerate?" ON)
option(ENABLE_AVX512 "Whether to use AVX512 to accelerate?" OFF)

#Only option is useful for #cmakedefine in TomoConfig.h.in
option(CPU_VERSION "Whether to use CPU to calculate?" ON)
option(GPU_VERSION "Whether to use GPU to accelerate?" ON)

if("${CPU_VERSION}" STREQUAL "OFF")
    if("${GPU_VERSION}" STREQUAL "OFF")

        message(FATAL_ERROR "Both of CPU & GPU version is OFF, exit!")

    endif("${GPU_VERSION}" STREQUAL "OFF")
endif("${CPU_VERSION}" STREQUAL "OFF")

#Add compilation flag part1: build type
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "release")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")
if("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    set(COMMON_FLAGS "${COMMON_FLAGS} -fopenmp -O2")
    message(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}, compile ${CMAKE_PROJECT_NAME} with ${COMMON_FLAGS} flag.")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    #TODO 拆成additional_flags
    set(COMMON_FLAGS "${COMMON_FLAGS} -fopenmp -O2")
    set(COMMON_FLAGS "${COMMON_FLAGS} -g")
    message(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}, compile ${CMAKE_PROJECT_NAME} with ${COMMON_FLAGS} flag.")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "release")

if("${GPU_VERSION}" AND "${CPU_VERSION}")
    message(STATUS "Try to build Tomo-Project in CPU & GPU version.")
else("${GPU_VERSION}" AND "${CPU_VERSION}")
    if("${GPU_VERSION}")
        message(STATUS "Try to build Tomo-Project in GPU version.")
    else("${GPU_VERSION}")
        message(STATUS "Try to build Tomo-Project in CPU version.")
    endif("${GPU_VERSION}")
endif("${GPU_VERSION}" AND "${CPU_VERSION}")


find_package(CUDA 8.0)
if(NOT ${CUDA_FOUND})
    if("${GPU_VERSION}")
        message(STATUS "WARNING: CUDA is not found, skip the compilation in GPU version.")
		set(${GPI_VERSION} off)
    endif("${GPU_VERSION}")
    set(GPU_VERSION off)
else(NOT ${CUDA_FOUND})
    #option GPU_VERSION has been determined at the beginning of the CMakeLists.txt
    #
    #add_definitions(-DGPU_VERSION)
endif(NOT ${CUDA_FOUND})

#Add compilation flags part2: library flags
#set(COMMON_FLAGS "${COMMON_FLAGS} -lutil -lcore -lz -lpthread -lm -lfftw3f -lfftw3f_threads")
#这里要特别注意 由于FFTW是从源文件进行编译，打包成库，在target_link_libraries中用${FFTW_LIBRARIES}变量进行指定，所以在这里指定编译器flags时，不能加入-lfftw3f等等。
#-lcore -lutil同理
set(COMMON_FLAGS "${COMMON_FLAGS} -lz -lpthread -lm ")

#Add compilation flags part3: SIMD flags
if("${ENABLE_SIMD}")
    if(APPLE)
        message(WARNING "${CMAKE_PROJECT_NAME} does not suppot SIMD on Apple.")
        set(ENABLE_SIMD OFF)
        set(CMAKE_C_FLAGS "${COMMON_FLAGS}")
        set(CMAKE_CXX_FLAGS "-D_FILE_OFFSET_BITS=64 ${COMMON_FLAGS}")
    else(APPLE)
        set(CMAKE_C_FLAGS "${COMMON_FLAGS} -mavx512f -mavx512cd")
        set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} -mavx512f -mavx512cd")
        try_compile(AVX512_SUPPORT
                    ${CMAKE_BINARY_DIR}
                    "${CMAKE_SOURCE_DIR}/cmake/SIMD/AVX512.c")
        if(AVX512_SUPPORT AND ENABLE_AVX512)
            message(STATUS "Build ${CMAKE_PROJECT_NAME} using AVX512.")
            set(ENABLE_SIMD_512 ON)

            #add_definitions is used to add MACROS into project directly, 
            #which is equivalent to making a definition "#define XXX" in a header file.
            #cmakedefine is controlled by "option()" in CMakeLists.txt
            #add_definitions(-DENABLE_SIMD_512)
        else(AVX512_SUPPORT AND ENABLE_AVX512)
            if(ENABLE_AVX512)
                message(WARNING "Compiler does not support AVX512.")
            endif(ENABLE_AVX512)
            set(CMAKE_C_FLAGS "${COMMON_FLAGS} -mavx")
            set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} -mavx")
            try_compile(AVX256_SUPPORT
                        ${CMAKE_BINARY_DIR}
                        "${CMAKE_SOURCE_DIR}/cmake/SIMD/AVX256.c")
            if(AVX256_SUPPORT AND ENABLE_AVX256)
                message(STATUS "Build ${CMAKE_PROJECT_NAME} using AVX256.")
                set(ENABLE_SIMD_256 ON)
                #add_definitions(-DENABLE_SIMD_256)
            else(AVX256_SUPPORT AND ENABLE_AVX256)
                if(ENABLE_AVX256)
                    message(WARNING "Compiler does not support AVX256.")
                endif(ENABLE_AVX256)
                set(ENABLE_SIMD OFF)
                set(CMAKE_C_FLAGS "${COMMON_FLAGS}")
                set(CMAKE_CXX_FLAGS "${COMMON_FLAGS}")
            endif(AVX256_SUPPORT AND ENABLE_AVX256)
        endif(AVX512_SUPPORT AND ENABLE_AVX512)
    endif(APPLE)
else("${ENABLE_SIMD}")
    message(STATUS "Build ${CMAKE_PROJECT_NAME} without using SIMD.")
    set(CMAKE_C_FLAGS "${COMMON_FLAGS}")
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS}")
endif("${ENABLE_SIMD}")


#output flags
message(STATUS "CMAKE_C_FLAGS : ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")

#用于载入已有的CMakeLists文件
#载入FFTW的CMakeLists文件
include(${CMAKE_SOURCE_DIR}/cmake/BuildFFTW.cmake)
message(STATUS "FFTW Path: ${FFTW_PATH}")
# 将fftw安装之后的头文件目录加入
include_directories(${FFTW_PATH}/include)

include_directories(${CMAKE_SOURCE_DIR}/external/Eigen)
include_directories(${CMAKE_SOURCE_DIR}/src/include)
include_directories(${CMAKE_SOURCE_DIR}/src/gpu/interface)
include_directories(${CMAKE_SOURCE_DIR}/src/gpu/include)



#set(CONFIG_DIR "${PROJECT_SOURCE_DIR}/Config")
#configure_file("${PROJECT_SOURCE_DIR}/TomoConfig.h.in"
#            "${CONFIG_DIR}/TomoConfig.h")

#Config

if("${CPU_VERSION}")
	set(CONFIG_CPU_DIR "${PROJECT_BINARY_DIR}/CONFIG_CPU")

	#If CPU_VERSION and GPU_VERSION is set ON in the meantime
	if("${GPU_VERSION}")
		set(GPU_VERSION off)
		configure_file("${PROJECT_SOURCE_DIR}/TomoConfig.h.in"
						"${CONFIG_CPU_DIR}/TomoConfig.h")
		set(GPU_VERSION on)
	else("${GPU_VERSION}")
		configure_file("${PROJECT_SOURCE_DIR}/TomoConfig.h.in"
						"${CONFIG_CPU_DIR}/TomoConfig.h")
	endif("${GPU_VERSION}")

	file(GLOB_RECURSE CORE_SOURCES ${CMAKE_SOURCE_DIR}/src/core/*.cpp)
	add_library(core_CPU STATIC ${CORE_SOURCES})
	add_dependencies(core_CPU FFTW)
	target_link_libraries(core_CPU PUBLIC ${FFTW_LIBRARIES})

	target_include_directories(core_CPU PUBLIC ${CONFIG_CPU_DIR})
	
	file(GLOB_RECURSE TOOL_SOURCES ${CMAKE_SOURCE_DIR}/src/tool/*.cpp)
	add_library(tool STATIC ${TOOL_SOURCES})

	file(GLOB_RECURSE UTIL_SOURCES ${CMAKE_SOURCE_DIR}/src/util/*.cpp)
	add_library(util STATIC ${UTIL_SOURCES})
	
	file(GLOB_RECURSE MAIN_SOURCES ${CMAKE_SOURCE_DIR}/src/main/*.cpp)

	foreach(MAINSRC ${MAIN_SOURCES})
	    get_filename_component(MAINNAME ${MAINSRC} NAME_WE) # file name without directories or longest extensi    on
		set(MAINNAME "${MAINNAME}_cpu")
	    add_executable(${MAINNAME} ${MAINSRC})
	    target_link_libraries(${MAINNAME} PUBLIC core_CPU util tool ${FFTW_LIBRARIES})
	endforeach()

endif("${CPU_VERSION}")

if("${GPU_VERSION}")


	message(STATUS "CUDA_VERSION : " ${CUDA_VERSION})
	message(STATUS "CUDA_TOOLKIT_ROOT_DIR : " ${CUDA_TOOLKIT_ROOT_DIR})
	message(STATUS "CUDA_INCLUDE_DIRS : " ${CUDA_INCLUDE_DIRS})
	message(STATUS "CUDA_LIBRARIES : " ${CUDA_LIBRARIES})
	message(STATUS "CUDA_CUBLAS_LIBRARIES : " ${CUDA_CUBLAS_LIBRARIES})
	message(STATUS "CUDA_curand_LIBRARY : " ${CUDA_curand_LIBRARY})
	message(STATUS "CUDA_CUFFT_LIBRARIES : " ${CUDA_CUFFT_LIBRARIES})

	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	
	# THE GPU ARCHITETURE SPECIFIC BINARY
	
	list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30")
	list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35,code=sm_35")
	list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50")
	list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52")
	list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_53,code=sm_53")
	list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60")
	string(REPLACE "." ";" VERSION_LIST ${CUDA_VERSION})
	set(l 0)
	list(GET VERSION_LIST ${l} LIST_VALUE)
	set(i 9)
	if((${LIST_VALUE} GREATER ${i}) OR (${LIST_VALUE} EQUAL ${i}))
	    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_70,code=sm_70")
	endif((${LIST_VALUE} GREATER ${i}) OR (${LIST_VALUE} EQUAL ${i}))
	list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
	list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fopenmp -lgomp")
	#for nvcc,cuda-memcheck
	if("${CMAKE_BUILD_TYPE}" STREQUAL "release")
	
	elseif("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
	    list(APPEND CUDA_NVCC_FLAGS " -G -lineinfo")
	endif("${CMAKE_BUILD_TYPE}" STREQUAL "release")
	
	message(STATUS "CUDA_NVCC_FLAGS : " ${CUDA_NVCC_FLAGS})
	
	# LINK THE CUDA LIBRARIES
	
	list(APPEND EXTERNAL_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
	list(APPEND EXTERNAL_LIBS ${CUDA_CUFFT_LIBRARIES})
	list(APPEND EXTERNAL_LIBS ${CUDA_curand_LIBRARY})
	
	if(("${CMAKE_MINOR_VERSION}" STRLESS "7") OR ("${CMAKE_MINOR_VERSION}" STREQUAL "7"))
	    find_library(CUDA_cudadevrt_LIBRARY libcudadevrt.a PATHS ${CUDA_TOOLKIT_ROOT_DIR})
	    list(APPEND EXTERNAL_LIBS ${CUDA_cudadevrt_LIBRARY})
	    #message(STATUS "CUDA_cudadevrt_LIBRARY:${CUDA_cudadevrt_LIBRARY}")
	endif(("${CMAKE_MINOR_VERSION}" STRLESS "7") OR ("${CMAKE_MINOR_VERSION}" STREQUAL "7"))

	set(CONFIG_GPU_DIR "${PROJECT_BINARY_DIR}/CONFIG_GPU")
	configure_file("${PROJECT_SOURCE_DIR}/TomoConfig.h.in"
					"${CONFIG_GPU_DIR}/TomoConfig.h")

	file(GLOB_RECURSE CORE_SOURCES ${CMAKE_SOURCE_DIR}/src/core/*.cpp)
	add_library(core_GPU STATIC ${CORE_SOURCES})
	add_dependencies(core_GPU FFTW)
	target_link_libraries(core_GPU PUBLIC ${FFTW_LIBRARIES})

	target_include_directories(core_GPU PUBLIC ${CONFIG_GPU_DIR})

	#TODO 如果后续要用到NCCL的话
	#find_package(NCCL)
	#if(NOT ${NCCL_FOUND})
	#        message(STATUS "WARNING: NCCL is not found, skip the compilation in GPU version.")
	#    endif("${GPU_VERSION}")
	#    set(GPU_VERSION off)
	#endif(NOT ${NCCL_FOUND})
	
	target_include_directories(core_GPU PUBLIC SYSTEM ${CUDA_INCLUDE_DIRS})
	
	
	#Buld CUDA core
	#TODO set output path
	#Note: INTERFACE_SOURCES is a key word, here we use INTERFACE_SOURCE
	set(KERNEL_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/src/gpu/include")
	cuda_include_directories(${CUDA_INCLUDE_DIRS} ${KERNEL_INCLUDE_PATH})
	file(GLOB_RECURSE GPU_SOURCES ${CMAKE_SOURCE_DIR}/src/gpu/src/*.cu)
	cuda_add_library(cuda_core STATIC ${GPU_SOURCES})
	target_link_libraries(cuda_core ${EXTERNAL_LIBS})
	
	file(GLOB_RECURSE INTERFACE_SOURCE ${CMAKE_SOURCE_DIR}/src/gpu/interface/*.cpp)
	add_library(interface_core STATIC ${INTERFACE_SOURCE})
	#target_link_libraries(interface_core cuda_core)
	#Notice: interface_core is dependent on core, for the use of MRC/CTF
	target_link_libraries(interface_core core_GPU cuda_core )
	#target_include_directories(interface_core PUBLIC ${CMAKE_SOURCE_DIR}/src/include)
	
	file(GLOB_RECURSE MAIN_SOURCES ${CMAKE_SOURCE_DIR}/src/main/*.cpp)

	foreach(MAINSRC ${MAIN_SOURCES})
	    get_filename_component(MAINNAME ${MAINSRC} NAME_WE) # file name without directories or longest extensi    on
		set(MAINNAME "${MAINNAME}_gpu")
	    add_executable(${MAINNAME} ${MAINSRC})
	    target_link_libraries(${MAINNAME} PUBLIC core_GPU util tool ${FFTW_LIBRARIES} cuda_core interface_core)
	endforeach()

endif("${GPU_VERSION}")

