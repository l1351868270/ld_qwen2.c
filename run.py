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

class CLDQwen2:
    def __init__(self, model_type: str, quantization_type: str, batch: int, max_seq_len: int):
        self.qwen2_path = os.path.dirname(os.path.abspath(__file__))
        if quantization_type == "fp16":
            self.qwen2lib = CDLL(os.path.join(self.qwen2_path, "./qwen2_fp16.so"))
        if quantization_type == "q80":
            self.qwen2lib = CDLL(os.path.join(self.qwen2_path, "./qwen2_q80.so"))
        if quantization_type == "q40":
            self.qwen2lib = CDLL(os.path.join(self.qwen2_path, "./qwen2_q40.so"))

        if quantization_type == "fp16":
            if model_type == "Qwen/Qwen1.5-0.5B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-0.5B.bin")
            if model_type == "Qwen/Qwen1.5-1.8B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-1.8B.bin")
            if model_type == "Qwen/Qwen1.5-4B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-4B.bin")
            if model_type == "Qwen/Qwen1.5-7B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-7B.bin")
            if model_type == "Qwen/Qwen1.5-14B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-14B.bin")
        if quantization_type == "q40":
            if model_type == "Qwen/Qwen1.5-0.5B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-0.5B-q40.bin")
            if model_type == "Qwen/Qwen1.5-1.8B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-1.8B-q40.bin")
            if model_type == "Qwen/Qwen1.5-4B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-4B-q40.bin")
            if model_type == "Qwen/Qwen1.5-7B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-7B-q40.bin")
            if model_type == "Qwen/Qwen1.5-14B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-14B-q40.bin")
            if model_type == "Qwen/Qwen1.5-32B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-32B-q40.bin")
        if quantization_type == "q80":
            if model_type == "Qwen/Qwen1.5-0.5B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-0.5B-q80.bin")
            if model_type == "Qwen/Qwen1.5-1.8B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-1.8B-q80.bin")
            if model_type == "Qwen/Qwen1.5-4B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-4B-q80.bin")
            if model_type == "Qwen/Qwen1.5-7B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-7B-q80.bin")
            if model_type == "Qwen/Qwen1.5-14B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-14B-q80.bin")
            if model_type == "Qwen/Qwen1.5-32B-Chat":
                self.checkpoint_path = os.path.join(self.qwen2_path, "qwen1.5-32B-q80.bin")

        self.batch = batch
        self.max_seq_len = max_seq_len

        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.stop_token_ids = [self.tokenizer.eos_token_id]
    
    def init(self):
        self.qwen2lib.c_init.argtypes = [c_int, c_int, c_char_p]
        checkpoint_path_buffer = create_string_buffer(self.checkpoint_path.encode("utf-8"))
        self.qwen2lib.c_init(c_int(self.batch), c_int(self.max_seq_len), checkpoint_path_buffer)
    
    def qwen2_forward(self, token, seq_len, pos)->list:
        self.qwen2lib.c_qwen2_forward.restype = POINTER(c_int * self.batch)
        sample = self.qwen2lib.c_qwen2_forward(c_int(self.batch), c_int(seq_len), (c_int * len(token))(*token), c_int(pos)) 
        res = []
        for i in sample.contents:
            res.append(int(i))
        return res
    
    def tokenized_prompt(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        model_inputs = model_inputs["input_ids"]
        tokenized_prompt = model_inputs.flatten().tolist()
        seq_len = len(tokenized_prompt)
        return tokenized_prompt, seq_len
    
    def tokenized_prompt_v1(self, prompt):
        model_inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        tokenized_prompt = model_inputs.flatten().tolist()
        seq_len = len(tokenized_prompt)
        return tokenized_prompt, seq_len
    
    def tokenized_prompts(self, prompts):
        texts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        model_inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        model_inputs = model_inputs["input_ids"].tolist()
        return model_inputs, len(model_inputs[0])

    def tokenized_prompts_v1(self, prompts):
        model_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        model_inputs = model_inputs["input_ids"].tolist()
        return model_inputs, len(model_inputs[0])
    
    def chat(self, tokenized_prompt, seq_len):
        output_text = ""
        pos = 0
        while (pos < self.max_seq_len):
            if pos < seq_len:
                tokenized_prompt_c = [tokenized_prompt[pos]]
            else:
                tokenized_prompt_c = next
            next = self.qwen2_forward(tokenized_prompt_c, 1, pos)

            if (next[0] in self.stop_token_ids) and pos >= seq_len:
                break
            next_text = self.tokenizer.decode(
                    next,
                    skip_special_tokens=True,
                    # spaces_between_special_tokens=False,
            )
            if (pos >= seq_len - 1):
                # yield next_text
                output_text += next_text
            pos += 1
        return output_text, pos

    def batch_chat(self, tokenized_prompt, batch, seq_len):
        output_text = ""
        pos = 0
        while (pos < self.max_seq_len):
            if pos < seq_len:
                tokenized_prompt_c = [tokenized_prompt[pos]]
            else:
                tokenized_prompt_c = next
            next = self.qwen2_forward(tokenized_prompt_c, 1, pos)

            if (next[0] in self.stop_token_ids) and pos >= seq_len:
                break
            next_text = self.tokenizer.decode(
                    next,
                    skip_special_tokens=True,
                    # spaces_between_special_tokens=False,
            )
            if (pos >= seq_len - 1):
                # yield next_text
                output_text += next_text
            pos += 1
        return output_text, pos
    
    def stream_chat(self, tokenized_prompt, seq_len):
        pos = 0
        while (pos < self.max_seq_len):
            if pos < seq_len:
                tokenized_prompt_c = [tokenized_prompt[pos]]
            else:
                tokenized_prompt_c = next
            next = self.qwen2_forward(tokenized_prompt_c, 1, pos)

            if (next[0] in self.stop_token_ids) and pos >= seq_len:
                break
            next_text = self.tokenizer.decode(
                    next,
                    skip_special_tokens=True,
                    # spaces_between_special_tokens=False,
            )
            if (pos >= seq_len - 1):
                yield next_text
            pos += 1

    def batch_stream_chat(self, tokenized_prompt, seq_len):
        pos = 0
        stop = [False for i in range(self.batch)]
        while (pos < self.max_seq_len):
            if pos < seq_len:
                tokenized_prompt_c = []
                for i in range(self.batch):
                    tokenized_prompt_c.append(tokenized_prompt[i][pos])
            else:
                tokenized_prompt_c = next

            # print(f"tokenized_prompt_c: {tokenized_prompt_c}")
            next = self.qwen2_forward(tokenized_prompt_c, self.batch, pos)
            

            if pos >= seq_len:
                for i in range(self.batch):
                    if next[i] in self.stop_token_ids:
                        stop[i] = True
            if all(stop) and pos >= seq_len:
                break

            next_text = []

            for i in range(self.batch):
                t = self.tokenizer.decode(
                    [next[i]],
                    skip_special_tokens=True,
                    # spaces_between_special_tokens=False,
                )
                next_text.append(t)
            if (pos >= seq_len - 1):
                yield next_text, next
            pos += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("-q", "--quantization_type", choices=("fp16", "q80", "q40"), type=str, default="fp16")
    parser.add_argument("-p", "--prompt", nargs='+', type=str, default="Give me a short introduction to large language model.")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    # python run.py -p "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" "天空为什么是蓝色的" --batch 16
    # python run.py -p "天空为什么是蓝色的" -m "Qwen/Qwen1.5-32B-Chat" -q q40 --batch 1
    args = parse_args()
    model_type = args.model_type
    quantization_type = args.quantization_type
    max_seq_len = args.max_seq_len
    prompt = args.prompt
    batch = args.batch

    model = CLDQwen2(model_type, quantization_type, batch, max_seq_len)
    model.init()

    print("="*50)
    print("user:")
    print(prompt[0])
    print("assistant:")
    # print("tokenized_prompt:")
    # tokenized_prompts, seq_len = model.tokenized_prompt(prompt[0])
    # print(tokenized_prompts, seq_len)
    # print("tokenized_prompt_v1:")
    # tokenized_prompts, seq_len = model.tokenized_prompt_v1(prompt[0])
    # print(tokenized_prompts, seq_len)
    # print("tokenized_prompts:")
    tokenized_prompts, seq_len = model.tokenized_prompts(prompt)
    # print(tokenized_prompts, seq_len)
    # print("tokenized_prompts_v1:")
    # tokenized_prompts, seq_len = model.tokenized_prompts_v1(prompt)
    # print(tokenized_prompts, seq_len)
    begin = time.time()
    num_tokens = seq_len

    next_texts = []
    next_ids = []
    stop_flag = False
    for text, id in model.batch_stream_chat(tokenized_prompts, seq_len):
        if id[0] in model.stop_token_ids:
            stop_flag = True
        if stop_flag == False and id[0] not in model.stop_token_ids:
            print(f"{text[0]}", end="", flush=True)
        num_tokens += 1
        next_texts.append(text)
        next_ids.append(id)
    end = time.time()
    print("")
    for b in range(1, batch):
        print("="*50)
        print("user:")
        print(prompt[b])
        print("assistant:")
        stop_flag = False
        for text, id in zip(next_texts, next_ids):
            if id[b] in model.stop_token_ids:
                stop_flag = True

            if stop_flag == False and id[b] not in model.stop_token_ids:
                print(f"{text[b]}", end="", flush=True)
        print("")
    print("="*50)
    # print(output_text)
    print(f"total time is:{end - begin:.2f}s, batch:{batch}, tokens:{num_tokens}, achieved {num_tokens / (end - begin):.2f} tokens/s")
    print("="*50)
