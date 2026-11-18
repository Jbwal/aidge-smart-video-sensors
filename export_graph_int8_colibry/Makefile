board = EVBTINY

COLIBRY_SDK_PATH ?= ${PWD}/../../..

APP = demo


DNNDIR := dnn
DATADIR := data
OUTDIR := $(DATADIR)/export_outputs

# Compare fmaps with Aidge ref
AIDGE_CMP := false

SAVE_OUTPUTS := false

PRMFLAGS :=

ifeq ($(AIDGE_CMP), true)
	PRMFLAGS += -DAIDGE_CMP
endif

ifeq ($(SAVE_OUTPUTS), true)
	PRMFLAGS += -DSAVE_OUTPUTS
	DUMMY := $(shell mkdir -p $(OUTDIR))
endif

APP_SRCS = ${PWD}/main.cpp



INCLUDE_DIRS  = -I ${PWD}
INCLUDE_DIRS += -I ${PWD}/${DNNDIR}
INCLUDE_DIRS += -I ${PWD}/${DNNDIR}/include
INCLUDE_DIRS += -I ${PWD}/${DNNDIR}/include/kernels
INCLUDE_DIRS += -I ${PWD}/${DNNDIR}/include/layers
INCLUDE_DIRS += -I ${PWD}/${DATADIR}


CFLAGS += -O2
CFLAGS += ${PRMFLAGS}
CFLAGS += ${INCLUDE_DIRS}

CFLAGS += -fno-exceptions
CFLAGS += -fno-rtti
CFLAGS += -fno-use-cxa-atexit


include ${COLIBRY_SDK_PATH}/FIRMWARE/COMMON/BOARDS/colibry_board.mk
include ${COLIBRY_SDK_PATH}/RISCV_RUNTIME/runtime.mk

custom: