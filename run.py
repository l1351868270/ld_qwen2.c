"""
nvcc --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2.cu -lm
python run.py
"""
import time
from transformers import AutoTokenizer
from ctypes import CDLL
from ctypes import c_int, POINTER
qwem2lib = CDLL("./qwen2.so")

import sys
sys.argv.append("-resize")
def init(batch: int, max_seq_len: int):
    qwem2lib.c_init(c_int(batch), c_int(max_seq_len))

def qwen2_forward(token, batch, seq_len, pos)->list:
    qwem2lib.c_qwen2_forward.restype = POINTER(c_int * batch)
    sample = qwem2lib.c_qwen2_forward(c_int(batch), c_int(seq_len), (c_int * len(token))(*token), c_int(pos)) 
    res = []
    for i in sample.contents:
        res.append(int(i))
    return res

if __name__ == '__main__':
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

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
    init(batch, max_seq_len)

    output = []
    begin = time.time()
    while (pos < 1):
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
