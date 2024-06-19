
#!/bin/bash
quantization_type=$1
if [ -z "$quantization_type" ]; then
  quantization_type="fp32"
fi

echo "quantization_type: $quantization_type"

ld_qwen2_home=$LD_QWEN2_HOME

if [ -z "$ld_qwen2_home" ]; then
  ld_qwen2_home="$(cd $(dirname "$0"); pwd)/../ld_qwen2_cache"
  echo "LD_QWEN2_HOME is not set"
fi
echo "LD_QWEN2_HOME: $ld_qwen2_home"

qwen2_path=$ld_qwen2_home/qwen2
echo "qwen2_path: $qwen2_path"

mkdir -p $qwen2_path
mkdir -p $qwen2_path/library
mkdir -p $qwen2_path/checkpoints

model_type="Qwen/Qwen1.5-0.5B-Chat"
checkpoint_file="$qwen2_path/checkpoints/qwen1.5-0.5B-$quantization_type.bin"

if [ ! -f $checkpoint_file ]; then
  echo "export model file..."
  python tools/export.py --filepath="qwen1.5-0.5B-$quantization_type.bin" --dtype="$quantization_type" --model_type=Qwen/Qwen1.5-0.5B-Chat
fi

make qwen2_cpp

python tools/run.py -p "天空为什么是蓝色的" -m "Qwen/Qwen1.5-0.5B-Chat" -q $quantization_type --batch 1
