"""
nvcc --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2.cu -lm
python run.py
"""
import argparse
import time
import os
from transformers import AutoTokenizer
from ctypes import CDLL
from ctypes import c_int, c_char_p, POINTER, create_string_buffer

qwen2_path = os.path.dirname(os.path.abspath(__file__))
qwen2lib = CDLL(os.path.join(qwen2_path, "./qwen2.so"))

def init(batch: int, max_seq_len: int, checkpoint_path: str):
    qwen2lib.c_init.argtypes = [c_int, c_int, c_char_p]
    checkpoint_path_buffer = create_string_buffer(checkpoint_path.encode("utf-8"))
    qwen2lib.c_init(c_int(batch), c_int(max_seq_len), checkpoint_path_buffer)

def qwen2_forward(token, batch, seq_len, pos)->list:
    qwen2lib.c_qwen2_forward.restype = POINTER(c_int * batch)
    sample = qwen2lib.c_qwen2_forward(c_int(batch), c_int(seq_len), (c_int * len(token))(*token), c_int(pos)) 
    res = []
    for i in sample.contents:
        res.append(int(i))
    return res

if __name__ == '__main__':
    # python run.py --model_type=Qwen/Qwen1.5-0.5B-Chat --prompt="Give me a short introduction to large language model."
    # python run.py --model_type=Qwen/Qwen1.5-1.8B-Chat --prompt="Give me a short introduction to large language model."
    # python run.py --model_type=Qwen/Qwen1.5-4B-Chat --prompt="Give me a short introduction to large language model."
    # python run.py --model_type=Qwen/Qwen1.5-14B-Chat --prompt="Give me a short introduction to large language model."
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("-q", "--quantization_type", choices=("fp16", "q80", "q40"), type=str, default="fp16")
    parser.add_argument("-p", "--prompt", type=str, default="Give me a short introduction to large language model.")
    parser.add_argument("--max_seq_len", type=int, default=256)
    args = parser.parse_args()
    model_type = args.model_type
    quantization_type = args.quantization_type
    max_seq_len = args.max_seq_len

    model_file = "qwen1.5-14B.bin"
    
    if quantization_type == "fp16":
        if model_type == "Qwen/Qwen1.5-0.5B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-0.5B.bin")
        if model_type == "Qwen/Qwen1.5-1.8B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-1.8B.bin")
        if model_type == "Qwen/Qwen1.5-4B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-4B.bin")
        if model_type == "Qwen/Qwen1.5-14B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-14B.bin")
    if quantization_type == "q40":
        if model_type == "Qwen/Qwen1.5-0.5B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-0.5B-q40.bin")
        if model_type == "Qwen/Qwen1.5-1.8B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-1.8B-q40.bin")
        if model_type == "Qwen/Qwen1.5-4B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-4B-q40.bin")
        if model_type == "Qwen/Qwen1.5-14B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-14B-q40.bin")
    if quantization_type == "q80":
        if model_type == "Qwen/Qwen1.5-0.5B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-0.5B-q80.bin")
        if model_type == "Qwen/Qwen1.5-1.8B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-1.8B-q80.bin")
        if model_type == "Qwen/Qwen1.5-4B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-4B-q80.bin")
        if model_type == "Qwen/Qwen1.5-14B-Chat":
            model_file = os.path.join(qwen2_path, "qwen1.5-14B-q80.bin")

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    stop_token_ids = [tokenizer.eos_token_id]

    print(stop_token_ids)
    # print(tokenizer.eos_token)

    prompt = args.prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # print(model_inputs)
    model_inputs = model_inputs["input_ids"]
    # print(model_inputs)
    tokenized_prompt = model_inputs.flatten().tolist()
    # print(tokenized_prompt)
    seq_len = len(tokenized_prompt)
    batch = 1
    pos = 0
    init(batch, max_seq_len, model_file)
    print("="*50)
    print("user:")
    print(prompt)
    print("assistant:")
    output = []
    begin = time.time()
    while (pos < max_seq_len):
        if pos < seq_len:
            tokenized_prompt_c = [tokenized_prompt[pos]]
        else:
            tokenized_prompt_c = next
        next = qwen2_forward(tokenized_prompt_c, batch, 1, pos)
        # break
        if (next[0] in stop_token_ids) and pos >= seq_len:
            break
        next_text = tokenizer.decode(
                next,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
        )
        if (pos >= seq_len - 1):
            print(f"{next_text}", end="", flush=True)
        output.append(tokenized_prompt_c[0])
        pos += 1
    end = time.time()
    print("")
    output.append(next[0])
    output_text = tokenizer.decode(
        output,
        skip_special_tokens=True
    )
    print("="*50)
    # print(output_text)
    print(f"total time is:{end - begin:.2f}s, tokens:{pos}, achieved {pos / (end - begin):.2f} tokens/s")
    print("="*50)
