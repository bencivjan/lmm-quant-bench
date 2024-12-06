from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def quantize_llava():
    pass

if __name__ == "__main__":
    model_path = 'llava-hf/llama3-llava-next-8b-hf'
    quant_path = 'llama3-llava-next-8b-awq'
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)