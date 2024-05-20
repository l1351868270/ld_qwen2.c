"""
nvcc --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2.cu -lm
python run.py
"""
import argparse
import time
from transformers import AutoTokenizer
from ctypes import CDLL
from ctypes import c_int, c_char_p, POINTER, create_string_buffer
qwem2lib = CDLL("./qwen2.so")


def init(batch: int, max_seq_len: int, checkpoint_path: str):
    qwem2lib.c_init.argtypes = [c_int, c_int, c_char_p]
    checkpoint_path_buffer = create_string_buffer(checkpoint_path.encode("utf-8"))
    qwem2lib.c_init(c_int(batch), c_int(max_seq_len), checkpoint_path_buffer)

def qwen2_forward(token, batch, seq_len, pos)->list:
    qwem2lib.c_qwen2_forward.restype = POINTER(c_int * batch)
    sample = qwem2lib.c_qwen2_forward(c_int(batch), c_int(seq_len), (c_int * len(token))(*token), c_int(pos)) 
    res = []
    for i in sample.contents:
        res.append(int(i))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="Qwen/Qwen1.5-0.5B-Chat")
    args = parser.parse_args()
    model_type = args.model_type

    model_file = "qwen1.5-14B.bin"
    if model_type == "Qwen/Qwen1.5-0.5B-Chat":
        model_file = "qwen1.5-0.5B.bin"
    if model_type == "Qwen/Qwen1.5-1.8B-Chat":
        model_file = "qwen1.5-1.8B.bin"
    if model_type == "Qwen/Qwen1.5-4B-Chat":
        model_file = "qwen1.5-4B.bin"
    if model_type == "Qwen/Qwen1.5-14B-Chat":
        model_file = "qwen1.5-14B.bin"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    prompt = "Give me a short introduction to large language model."
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
    max_seq_len = 256
    pos = 0
    init(batch, max_seq_len, "qwen1.5-0.5B.bin")
    # init(batch, max_seq_len, "qwen1.5-1.8B.bin")

    output = []
    begin = time.time()
    while (pos < max_seq_len):
        if pos < seq_len:
            tokenized_prompt_c = [tokenized_prompt[pos]]
        else:
            tokenized_prompt_c = next
        next = qwen2_forward(tokenized_prompt_c, batch, 1, pos)
        print(f"pos:{pos} {tokenized_prompt_c}")
        output.append(tokenized_prompt_c[0])
        pos += 1
    end = time.time()
    
    output.append(next[0])
    output_text = tokenizer.decode(
        output,
        skip_special_tokens=True
    )

    print(output_text)
    print(f"total time is:{end - begin:.2f}s, tokens:{max_seq_len}, achieved {max_seq_len / (end - begin):.2f} tokens/s")
