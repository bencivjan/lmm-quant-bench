import os
os.environ['HF_HOME'] = '/projects/bdes/bcivjan/.cache/huggingface'

from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration, \
                        LlavaNextForConditionalGeneration, AutoTokenizer, AutoModel, GenerationConfig
from awq import AutoAWQForCausalLM
from datasets import load_dataset
import torch
import time
import psutil
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np
import requests

def warmup(model):
    warm_up = torch.randn((4096,4096)).to(DEVICE)
    torch.mm(warm_up,warm_up)

def generate_torch(model, input_ids, n_generate):
    context_time = 0
    generate_time = []

    with torch.inference_mode():
        for i in range(n_generate):
            if DEVICE != "cpu":
                torch.cuda.synchronize()
            start = time.time()

            if i == 0:
                # prefill context
                inputs = torch.as_tensor(input_ids, device=DEVICE)
            else:
                # decode tokens
                inputs = torch.as_tensor(token, device=DEVICE)

            out = model(inputs, use_cache=True)

            if DEVICE != "cpu":
                torch.cuda.synchronize()
            token = out[0][:, -1].max(1)[1].unsqueeze(1)

            if i == 0:
                context_time += time.time() - start
            else:
                generate_time.append(time.time() - start)

    return context_time, generate_time

if __name__ == "__main__":
    # Parameters
    pretrained = True
    # model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
    model_path = "llava-hf/llava-1.5-7b-hf"
    DEVICE = "cuda"
    torch_dtype = torch.float32

    context = 32
    n_generate = 32

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Random image
    # url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    # inputs = processor(images=image, text=torch.randint(0, processor.tokenizer.vocab_size, (1, context)), padding=True, return_tensors="pt")
    # print(inputs.keys())
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, context)) # Batch size of 1
    # input_ids = processor(images=torch.randint(), text=torch.randint(0, processor.tokenizer.vocab_size, (1, context)), padding=True, return_tensors="pt")
    generator = generate_torch

    print(f" -- Loading model...")

    if pretrained:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            device_map=DEVICE,
            torch_dtype=torch_dtype,
        )
    else:
        raise NotImplementedError
        # model = AutoAWQForCausalLM.from_quantized(
        #     model_path, quant_file, max_seq_len=n_generate, batch_size=batch_size, safetensors=not no_safetensors
        # )

    print(f" -- Warming up...")
    warmup(model)

    print(f" -- Generating {n_generate} tokens, {input_ids.shape[1]} in context...")

    try:
        context_time, generate_time = generator(model, input_ids, n_generate)
        successful_generate = True
    except RuntimeError as ex:
        if 'out of memory' in str(ex).lower():
            successful_generate = False
        else:
            raise RuntimeError(ex)

    total_memory_used = 0
    memory_pct = 100
    if successful_generate:
        # number of tokens in context / time for processing context * batch size
        prefill_tokens_per_second = round(input_ids.shape[1] / context_time, 2)
        # 1 second / median time per token in seconds * batch size
        decode_tokens_per_second = round(1 / np.median(generate_time), 2)

        print(f" ** Speed (Prefill): {prefill_tokens_per_second:.2f} tokens/second")
        print(f" ** Speed (Decode): {decode_tokens_per_second:.2f} tokens/second")

        if DEVICE == "cpu":
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            memory_info = psutil.virtual_memory()
            memory_pct = mem_info.rss / memory_info.total
            total_memory_used = float(mem_info.rss) / (1024 ** 3)
            print(f" ** Max Memory (device: {DEVICE}): {total_memory_used:.2f} GB ({memory_pct:.2f}%)")
        else:
            for device in range(torch.cuda.device_count()):
                memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                total_memory_used += memory_used
                memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100
                print(f" ** Max Memory (device: {device}): {memory_used:.2f} GB ({memory_pct:.2f}%)")
    else:
        prefill_tokens_per_second = 'OOM'
        decode_tokens_per_second = 'OOM'

    if pretrained:
        version = "FP16" if DEVICE != "cpu" else "BF16"
    else:
        version = model.quant_config.version

    print( {
        "Batch Size": 1,
        "Prefill Length": input_ids.shape[1],
        "Decode Length": n_generate,
        "Prefill tokens/s": prefill_tokens_per_second,
        "Decode tokens/s": decode_tokens_per_second,
        "Memory (VRAM)": f"{total_memory_used:.2f} GB ({memory_pct:.2f}%)"
    })