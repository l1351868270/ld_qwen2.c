CC = nvcc

cublas: qwen2_cublas.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2_cublas.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86

v0: qwen2_v0.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2_v0.cu -lm -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86

all: v0 cublas

clean:
	rm -rf *.log *.nsys-rep *.sqlite *.so