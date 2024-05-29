CC = nvcc

cublas: qwen2_cublas.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2_cublas.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86

v0: qwen2_v0.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2_v0.cu -lm -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86

v1: qwen2_v1.cu
	$(CC) --shared -Xcompiler -fPIC -std=c++17 -o qwen2.so -O3 qwen2_v1.cu -lm -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86
all: v0 v1 cublas

clean:
	rm -rf *.log *.nsys-rep *.sqlite *.so