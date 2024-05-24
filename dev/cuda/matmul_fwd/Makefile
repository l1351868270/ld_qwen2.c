
GPU_TARGET=A40

NVCC=nvcc

BUILD_DIR=build

ifeq ($(GPU_TARGET),A40)
    NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_A40 -arch=sm_80 -lcuda -lcudadevrt -lcudart_static -lcublas
endif


v0: matmul_v0_fwd.cu
	$(NVCC) matmul_v0_fwd.cu $(NVCCFLAGS) -o matmul_v0_fwd


v1: matmul_v1_fwd.cu
	$(NVCC) matmul_v1_fwd.cu $(NVCCFLAGS) -o matmul_v1_fwd


cublas: matmul_cublas_fwd.cu
	$(NVCC) matmul_cublas_fwd.cu $(NVCCFLAGS) -o matmul_cublas_fwd

kittens: matmul_kittens_fwd.cu
	$(NVCC) matmul_kittens_fwd.cu $(NVCCFLAGS) -o matmul_kittens_fwd

all: v0 v1 cublas kittens

run:
	ncu --help

benchmark:
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./matmul_v0_fwd
	python stat-csv.py
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./matmul_v1_fwd
	python stat-csv.py
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./matmul_cublas_fwd
	python stat-csv.py
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./matmul_kittens_fwd
	python stat-csv.py
	ncu --csv --log-file benchmark.csv --metrics  l1tex__data_bank_conflicts_pipe_lsu_mem_shared,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./matmul_kittens_fwd

clean:
	rm -f *.csv *.log matmul_v0_fwd matmul_v0_fwd.d matmul_v1_fwd matmul_v1_fwd.d matmul_cublas_fwd matmul_cublas_fwd.d matmul_kittens_fwd matmul_kittens_fwd.d