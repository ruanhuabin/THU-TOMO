#这个值会根据当前路径的变化而变化
CURRENT_DIR := $(shell pwd)
LIB_PATH := ${CURRENT_DIR}/../../lib/
EXTERNAL_LIB_PATH := ${CURRENT_DIR}/../../3rdlib/lib/
INC_PATH := ${CURRENT_DIR}/../../src/include/
EXTERNAL_INC_PATH := $(CURRENT_DIR)/../../3rdlib/include/
EXTERNAL_INC_EIGEN_PATH := $(CURRENT_DIR)/../../external/Eigen/
BIN_PATH := ${CURRENT_DIR}/bin

MODULE_DIR_NAME := $(shell ls $(CURRENT_DIR)/src/ 2>/dev/null)
MODULE_LIBS_1 := $(subst main,,$(MODULE_DIR_NAME))
MODULE_LIBS_2 = $(subst tool,,$(MODULE_LIBS_1))
MODULE_LIBS_3 = $(subst include,,$(MODULE_LIBS_2))
MODULE_LIBS_4 = $(subst util,,$(MODULE_LIBS_3))
MODULE_LIBS_5 = $(subst core,,$(MODULE_LIBS_4))

FINAL_MODULE_LIBS := $(addprefix -l, $(MODULE_LIBS_5))
CC := g++
CC_CPP:= g++
AR := ar
OPT_LEVEL = -O2
OMP_FLAG = -fopenmp
DEBUG_FLAG = -g
LIBS := -lutil -lcore -lz -lpthread -lm -lfftw3f -lfftw3f_threads
#LIBS := -lutil -lcore $(FINAL_MODULE_LIBS)
CFLAGS := ${OPT_LEVEL} ${OMP_FLAG} -std=c++11
CFLAGS_g := ${DEBUG_FLAG} ${OMP_FLAG}
CFLAGS_CPP := ${OPT_LEVEL} ${OMP_FLAG}
CFLAGS_CPP_g := ${DEBUG_FLAG} ${OMP_FLAG}
ARFLAGS := -rcs

