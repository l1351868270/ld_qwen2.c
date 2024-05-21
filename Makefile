CC = nvcc

.PHONY: qwen2_v1
qwen2_v1: qwen2_v1.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2_v1.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86

.PHONY: qwen2
qwen2: qwen2.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2.cu -lm -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86

