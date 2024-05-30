CC = nvcc

cublas: qwen2_cublas.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2_cublas.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80 

v2: qwen2_v2.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2_v2.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80

v3: qwen2_v3.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2_v3.cu -lm -lcublas -lcublasLt -gencode arch=compute_80,code=sm_80


v0: qwen2_v0.cu
	$(CC) --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2_v0.cu -lm -gencode arch=compute_80,code=sm_80

v1: qwen2_v1.cu
	$(CC) --shared -Xcompiler -fPIC -std=c++17 -o qwen2.so -O3 qwen2_v1.cu -lm -gencode arch=compute_80,code=sm_80

q80: qwen2_q80.cu
	$(CC) --shared -Xcompiler -fPIC -std=c++17 -o qwen2.so -O3 qwen2_q80.cu -lm -gencode arch=compute_80,code=sm_80

q40: qwen2_q40.cu
	$(CC) --shared -Xcompiler -fPIC -std=c++17 -o qwen2.so -O3 qwen2_q40.cu -lm -gencode arch=compute_80,code=sm_80

all: v0 v1 v2 v3 q80 q40 cublas

run:
	ncu --help

b:
	ncu --csv --log-file benchmark.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum python run.py
	python stat-csv.py

clean:
	rm -rf *.log *.nsys-rep *.sqlite *.so *.csv