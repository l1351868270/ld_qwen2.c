
GPU_TARGET=A40

NVCC=nvcc

BUILD_DIR=build

ifeq ($(GPU_TARGET),A40)
    NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_A40 -arch=sm_86 -arch=sm_80 -lcuda -lcudadevrt -lcudart_static -lcublas
endif


v0: gemv_v0_fwd.cu
	$(NVCC) gemv_v0_fwd.cu $(NVCCFLAGS) -o gemv_v0_fwd

v1: gemv_v1_fwd.cu
	$(NVCC) gemv_v1_fwd.cu $(NVCCFLAGS) -o gemv_v1_fwd

cublas: gemv_cublas_fwd.cu
	$(NVCC) gemv_cublas_fwd.cu $(NVCCFLAGS) -o gemv_cublas_fwd

fast: gemv_fast_fwd.cu
	$(NVCC) gemv_fast_fwd.cu $(NVCCFLAGS) -o gemv_fast_fwd

kittens: gemv_kittens_fwd.cu
	$(NVCC) gemv_kittens_fwd.cu $(NVCCFLAGS) -o gemv_kittens_fwd

all: v0 v1 cublas fast kittens

run:
	ncu --help

benchmark:
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./gemv_v0_fwd
	python stat-csv.py
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./gemv_v1_fwd
	python stat-csv.py
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./gemv_cublas_fwd
	python stat-csv.py
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./gemv_fast_fwd
	python stat-csv.py
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./gemv_kittens_fwd
	python stat-csv.py
	ncu --csv --log-file benchmark.csv --metrics  l1tex__data_bank_conflicts_pipe_lsu_mem_shared,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./gemv_kittens_fwd

clean:
	rm -f *.csv *.log gemv_v0_fwd gemv_v0_fwd.d gemv_v1_fwd gemv_v1_fwd.d gemv_cublas_fwd gemv_cublas_fwd.d gemv_fast_fwd gemv_fast_fwd.d gemv_kittens_fwd gemv_kittens_fwd.d