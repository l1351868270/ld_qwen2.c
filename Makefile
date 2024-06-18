CC = nvcc

qwen2: src/models/qwen2/qwen2.cu
	$(CC) --shared -DWEIGHTS_DEBU -DARGMAX_DEBU -Xcompiler -fPIC --std=c++20 -o qwen2_fp16.so -O3 src/models/qwen2/qwen2.cu -I./ -lm -lcublas -lcublasLt  -gencode arch=compute_80,code=sm_80 

qwen2_cpp: src/models/qwen2/qwen2.cpp
	g++ --shared -DLINEAR_DEBU -D ARGMAX_DEBU -fPIC --std=c++20 -o qwen2_fp32.so -Ofast src/models/qwen2/qwen2.cpp -I./ -lm -fopenmp

qwen2_avx512: src/models/qwen2/qwen2.cpp
	g++ --shared -DAVX512_FWD -D ARGMAX_DEBU -fPIC --std=c++20 -o qwen2_fp32.so -Ofast src/models/qwen2/qwen2.cpp -I./ -lm -fopenmp -mavx512f


cublas_W16A32: qwen2_cublas_W16A32.cu
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o qwen2_fp16.so -O3 qwen2_cublas_W16A32.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 

cublas_W16A16: qwen2_cublas_W16A16.cu
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o qwen2_fp16.so -O3 qwen2_cublas_W16A16.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 

cublas_W8A16: qwen2_cublas_W8A16.cu
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o qwen2_q80.so -O3 qwen2_cublas_W8A16.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 

cublas_W8A8: qwen2_cublas_W8A8.cu
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o qwen2_q80.so -O3 qwen2_cublas_W8A8.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 

v2: qwen2_v2.cu
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o qwen2_fp16.so -O3 qwen2_v2.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80

v3: qwen2_v3.cu
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o qwen2_fp16.so -O3 qwen2_v3.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80


v0: qwen2_v0.cu
	$(CC) --shared -Xcompiler -fPIC --std=c++20 -o qwen2_fp16.so -O3 qwen2_v0.cu -lm -gencode arch=compute_80,code=sm_80

v1: qwen2_v1.cu
	$(CC) --shared -Xcompiler -fPIC -std=c++20 -o qwen2_fp16.so -O3 qwen2_v1.cu -lm -gencode arch=compute_80,code=sm_80

q80: qwen2_q80.cu
	$(CC) --shared -Xcompiler -fPIC -std=c++20 -o qwen2_q80.so -O3 qwen2_q80.cu -lm -gencode arch=compute_80,code=sm_80

q40: qwen2_q40.cu
	$(CC) --shared -Xcompiler -fPIC -std=c++20 -o qwen2_q40.so -O3 qwen2_q40.cu -lm -gencode arch=compute_80,code=sm_80

all: v0 v1 v2 v3 q80 q40 cublas

run:
	ncu --help

b:
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum python run.py
	python stat-csv.py

clean:
	rm -rf *.log *.nsys-rep *.sqlite *.so *.csv