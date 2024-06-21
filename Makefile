CC = nvcc

EMPTY :=
ifeq ($(LD_QWEN2_HOME),$(EMPTY))
	export LD_QWEN2_HOME = $(shell pwd)/ld_qwen2_cache
endif


$(info "LD_QWEN2_HOME: $(LD_QWEN2_HOME)")

LD_QWEN2_LIB_PATH=$(LD_QWEN2_HOME)/qwen2/lib
$(info "LD_QWEN_LIB_PATH: $(LD_QWEN2_LIB_PATH)")

qwen2: src/models/qwen2/qwen2.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o libqwen2_fp16.so -O3 src/models/qwen2/qwen2.cu -I./ -lm -lcublas -lcublasLt  -gencode arch=compute_80,code=sm_80 
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so
	mv libqwen2_fp16.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so

qwen2_cpp: src/models/qwen2/qwen2.cpp
	mkdir -p $(LD_QWEN2_LIB_PATH)
	g++ --shared -fPIC --std=c++20 -o libqwen2_fp32.so -Ofast src/models/qwen2/qwen2.cpp -I./ -lm -fopenmp
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp32.so
	mv libqwen2_fp32.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp32.so

qwen2_mpi: src/models/qwen2/qwen2_mpi.cpp
	mkdir -p $(LD_QWEN2_LIB_PATH)
	mpicxx -DENABLE_MUTI --shared -fPIC --std=c++20 -o libqwen2_fp32.so -Ofast src/models/qwen2/qwen2_mpi.cpp -I./ -lm -fopenmp -lmpi
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp32.so
	mv libqwen2_fp32.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp32.so

qwen2_avx512: src/models/qwen2/qwen2.cpp
	mkdir -p $(LD_QWEN2_LIB_PATH)
	g++ --shared -DAVX512_FWD -fPIC --std=c++20 -o libqwen2_fp32.so -Ofast src/models/qwen2/qwen2.cpp -I./ -lm -fopenmp -mavx512f
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp32.so
	mv libqwen2_fp32.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp32.so
	
qwen2_neon: src/models/qwen2/qwen2.cpp
	mkdir -p $(LD_QWEN2_LIB_PATH)
	g++ --shared -DNEON_FWD -fPIC --std=c++20 -o libqwen2_fp32.so -O3 src/models/qwen2/qwen2.cpp -I./ -lm -fopenmp
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp32.so
	mv libqwen2_fp32.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp32.so

single_W16A32: single_deploy/qwen2_cublas_W16A32.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o libqwen2_fp16.so -O3 single_deploy/qwen2_cublas_W16A32.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so
	mv libqwen2_fp16.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so

single_W16A16: single_deploy/qwen2_cublas_W16A16.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o libqwen2_fp16.so -O3 single_deploy/qwen2_cublas_W16A16.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so
	mv libqwen2_fp16.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so

single_W8A16: single_deploy/qwen2_cublas_W8A16.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o libqwen2_q80.so -O3 single_deploy/qwen2_cublas_W8A16.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_q80.so
	mv libqwen2_q80.so $(LD_QWEN2_LIB_PATH)/libqwen2_q80.so

single_W8A8: single_deploy/qwen2_cublas_W8A8.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o libqwen2_q80.so -O3 single_deploy/qwen2_cublas_W8A8.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_q80.so
	mv libqwen2_q80.so $(LD_QWEN2_LIB_PATH)/libqwen2_q80.so

single_v0: single_deploy/qwen2_v0.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o libqwen2_fp16.so -O3 single_deploy/qwen2_v0.cu -lm -gencode arch=compute_80,code=sm_80
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so
	mv libqwen2_fp16.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so

single_v1: single_deploy/qwen2_v1.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC -std=c++20 -o libqwen2_fp16.so -O3 single_deploy/qwen2_v1.cu -lm -gencode arch=compute_80,code=sm_80
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so
	mv libqwen2_fp16.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so

single_v2: single_deploy/qwen2_v2.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o libqwen2_fp16.so -O3 single_deploy/qwen2_v2.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so
	mv libqwen2_fp16.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so

single_v3: single_deploy/qwen2_v3.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o libqwen2_fp16.so -O3 single_deploy/qwen2_v3.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so
	mv libqwen2_fp16.so $(LD_QWEN2_LIB_PATH)/libqwen2_fp16.so

single_q80: single_deploy/qwen2_q80.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC -std=c++20 -o libqwen2_q80.so -O3 single_deploy/qwen2_q80.cu -lm -gencode arch=compute_80,code=sm_80

single_q40: single_deploy/qwen2_q40.cu
	mkdir -p $(LD_QWEN2_LIB_PATH)
	$(CC) --shared -Xcompiler -fPIC -std=c++20 -o libqwen2_q40.so -O3 single_deploy/qwen2_q40.cu -lm -gencode arch=compute_80,code=sm_80
	rm -rf $(LD_QWEN2_LIB_PATH)/libqwen2_q40.so
	mv libqwen2_q40.so $(LD_QWEN2_LIB_PATH)/libqwen2_q40.so

all: qwen2_cpp single_W16A16 single_W8A16

run:
	ncu --help

b:
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum python run.py
	python stat-csv.py

clean:
	rm -rf *.log *.nsys-rep *.sqlite *.so *.csv $(LD_QWEN2_LIB_PATH)/* *.bin