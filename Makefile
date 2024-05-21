CC = nvcc

.PHONY: qwen2_v1
qwen2_v1: qwen2_v1.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2_v1.cu -lm
	
.PHONY: qwen2
qwen2: qwen2.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2.cu -lm

