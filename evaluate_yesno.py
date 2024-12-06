import os
os.environ['HF_HOME'] = '/projects/bdes/bcivjan/.cache/huggingface'

from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration, \
                        LlavaNextForConditionalGeneration, AutoTokenizer, AutoModel, GenerationConfig
from awq import AutoAWQForCausalLM
from datasets import load_dataset
import aiohttp
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from tqdm import tqdm
from num2words import num2words
import numpy as np
import requests
import itertools


def simple_accuracy(generation: str, label2weight: dict):
    score = 0
    generation = generation.strip().lower()

    for label_id in label2weight.keys():
        if label_id in generation:
            score += label2weight[label_id]

    return score / sum(label2weight.values())

def calc_avg_simple_accuracy(generations: list[str], labels: list[dict]):
    score_sum = 0

    for generation, label in zip(generations, labels):
        score_sum += simple_accuracy(generation, label)

    return score_sum / len(generations)

def yes_no_accuracy(generation: str, label2weight: dict):
    generation = generation.strip().lower()

    if len(generation.split()) > 1:
        print(generation)
        raise AssertionError("Generation should be a single word")

    return label2weight.get(generation, 0)

def calc_avg_yes_no_accuracy(generations: list[str], labels: list[dict]):
    score_sum = 0

    for generation, label in zip(generations, labels):
        score_sum += yes_no_accuracy(generation, label)

    return score_sum / len(generations)

def get_yesno_accuracy_metrics(generations: list[str], labels: list[dict]):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for generation, label in zip(generations, labels):
        generation = generation.strip().lower()

        if len(generation.split()) > 1:
            print(f"Generation should be a single word: {generation}")

        if 'yes' in generation and 'yes' in label:
            tp += 1
        elif 'yes' in generation and 'yes' not in label:
            fp += 1
        elif 'no' in generation and 'no' in label:
            fn += 1
        elif 'no' in generation and 'no' not in label:
            tn += 1
        else:
            print(f"Invalid generation: {generation}\n")

    return tp, fp, tn, fn

def percentage_correct(correct, total):
    if total == 0:
        return 0.0  # To handle division by zero
    return (correct / total) * 100

def precision(tp, fp):
    if tp + fp == 0:
        return 0.0  # To handle division by zero
    return tp / (tp + fp)


def recall(tp, fn):
    if tp + fn == 0:
        return 0.0  # To handle division by zero
    return tp / (tp + fn)


def f1_score(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    if p + r == 0:
        return 0.0  # To handle division by zero
    return 2 * (p * r) / (p + r)

# ======= TODO: Can try to get this to work with dataset.map() =======
# def preprocess_vqa(examples):
#     images = []
#     prompts = []
#     labels = []

#     for image_id, question, dataset_labels in zip(dataset['image_id'],
#                                                   dataset['question'],
#                                                   dataset['label']):
#         image = Image.open(image_id)
#         prompt = f"USER: <image>\n{question} ASSISTANT:"
#         images.append(image)
#         prompts.append(prompt)

#         label2weight = {}
#         for i, id in enumerate(dataset_labels['ids']):
#             if id.isnumeric():
#                 id = num2words(int(id))
#             label2weight[id] = dataset_labels['weights'][i]
#         labels.append(label2weight)

#     examples['label2weight'] = label2weight
#     examples['prompt'] = prompts
#     examples['image'] = images

#     return examples

def preprocess_vqa_batch(dataset, start, end):
    images = []
    prompts = []
    labels = []

    for image_id, question, dataset_labels in zip(dataset['image_id'][start:end],
                                                  dataset['question'][start:end],
                                                  dataset['label'][start:end]):
        image = Image.open(image_id)
        prompt = f"USER: <image>\n{question} ASSISTANT:"
        images.append(image)
        prompts.append(prompt)

        label2weight = {}
        for i, id in enumerate(dataset_labels['ids']):
            if id.isnumeric():
                id = num2words(int(id))
            label2weight[id] = dataset_labels['weights'][i]
        labels.append(label2weight)

    return images, prompts, labels

# From https://huggingface.co/allenai/Molmo-7B-D-0924/discussions/7
def process_molmo_batch(
    processor: AutoProcessor,
    texts: list[str],
    images_list: list[list[Image.Image]]
) -> dict[str, torch.Tensor]:
    """
    Process in batch.
    
    Args:
        processor: The original processor.
        texts: List of text inputs
        images_list: List of lists containing PIL images.
        
    Returns:
        Dict with padded input_ids, images, image_input_idx, image_masks.
    """
    batch_size = len(texts)
    tokens_list = []
    for text in texts:
        tokens = processor.tokenizer.encode(" " + text, add_special_tokens=False)
        tokens_list.append(tokens)
    images_arrays_list = []
    image_idxs_list = []
    for images in images_list:
        if images:
            image_arrays = []
            for image in images:
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")
                    image = ImageOps.exif_transpose(image)
                    image_arrays.append(np.array(image))
                else:
                    assert len(image.shape) == 3 and image.shape[-1] == 3
                    image_arrays.append(image.astype(np.uint8))
            images_arrays_list.append(image_arrays)
            image_idx = [-1] * len(image_arrays)
            image_idxs_list.append(image_idx)
        else:
            images_arrays_list.append(None)
            image_idxs_list.append(None)
    images_kwargs = {
        "max_crops": 12,
        "overlap_margins": [4, 4],
        "base_image_input_size": [336, 336],
        "image_token_length_w": 12,
        "image_token_length_h": 12,
        "image_patch_size": 14,
        "image_padding_mask": True,
    }
    outputs_list = []
    for i in range(batch_size):
        tokens = tokens_list[i]
        images = images_arrays_list[i]
        image_idx = image_idxs_list[i]
        out = processor.image_processor.multimodal_preprocess(
            images=images,
            image_idx=image_idx,
            tokens=np.asarray(tokens).astype(np.int32),
            sequence_length=1536,
            image_patch_token_id=processor.special_token_ids["<im_patch>"],
            image_col_token_id=processor.special_token_ids["<im_col>"],
            image_start_token_id=processor.special_token_ids["<im_start>"],
            image_end_token_id=processor.special_token_ids["<im_end>"],
            **images_kwargs,
        )
        outputs_list.append(out)

    batch_outputs = {}
    for key in outputs_list[0].keys():
        tensors = [torch.from_numpy(out[key]) for out in outputs_list]
        batch_outputs[key] = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=-1
        )
    bos = processor.tokenizer.bos_token_id or processor.tokenizer.eos_token_id
    batch_outputs["input_ids"] = torch.nn.functional.pad(
        batch_outputs["input_ids"], (1, 0), value=bos
    )
    if "image_input_idx" in batch_outputs:
        image_input_idx = batch_outputs["image_input_idx"]
        batch_outputs["image_input_idx"] = torch.where(
            image_input_idx < 0, image_input_idx, image_input_idx + 1
        )
    return batch_outputs

# def evaluate_llava_1_5(images, prompts, labels, num_samples, batch_size):
#     model_path = "llava-hf/llava-1.5-7b-hf"
#     processor = AutoProcessor.from_pretrained(model_path)
#     model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

#     # Recommended with LLaVa batched generation
#     processor.tokenizer.padding_side = "left"
#     processor.patch_size = model.config.vision_config.patch_size
#     processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

#     responses = []
#     num_batches = (num_samples + batch_size - 1) // batch_size

#     for batch in tqdm(range(num_batches), desc=f"Calculating accuracy (batch size: {batch_size})"):
#         image_batch = images[batch * batch_size : (batch+1) * batch_size]
#         prompt_batch = prompts[batch * batch_size : (batch+1) * batch_size]

#         # start = batch * batch_size
#         # end = min((batch+1) * batch_size, num_samples)
#         # image_batch, prompt_batch, label_batch = preprocess_vqa_batch(dataset, start, end)

#         inputs = processor(images=image_batch, text=prompt_batch, padding=True, return_tensors="pt")
#         inputs.to(device)

#         generate_ids = model.generate(**inputs, max_new_tokens=20)
#         response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         responses += response

#         # Remove inputs from gpu memory
#         # del inputs

#     return calc_avg_simple_accuracy(responses, labels)

def get_model_prompt_yes_no(model_path, processor, question):
    if model_path == "llava-hf/llama3-llava-next-8b-hf":
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."}
                ]
            },
            {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question + " Answer in one word only."}
            ]
        }]
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    elif model_path == "llava-hf/llava-v1.6-mistral-7b-hf" \
        or model_path == "llava-hf/llava-1.5-7b-hf":
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question + " Answer in one word only."}
            ]
        }]
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    elif model_path == "allenai/Molmo-7B-D-0924" or model_path == "cyan2k/molmo-7B-D-bnb-4bit":
        return "User: " + question + " Answer in one word only." + " Assistant:"
    
    else:
        raise NotImplementedError

def evaluate_llava(model, processor, images, prompts, labels, num_samples, batch_size):
    # Recommended with LLaVa batched generation
    processor.tokenizer.padding_side = "left"
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

    responses = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch in tqdm(range(num_batches), desc=f"Calculating accuracy (batch size: {batch_size})"):
        image_batch = images[batch * batch_size : (batch+1) * batch_size]
        prompt_batch = prompts[batch * batch_size : (batch+1) * batch_size]

        # start = batch * batch_size
        # end = min((batch+1) * batch_size, num_samples)
        # image_batch, prompt_batch, label_batch = preprocess_vqa_batch(dataset, start, end)

        inputs = processor(images=image_batch, text=prompt_batch, padding=True, return_tensors="pt")
        inputs.to(device)

        print(inputs.pixel_values)
        print(inputs.pixel_values.shape, inputs.pixel_values.dtype)

        generate_ids = model.generate(**inputs, max_new_tokens=1)
        generated_texts = processor.batch_decode(generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # generated_texts = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        responses += generated_texts

    # for p, r in zip(prompts, responses):
    #     print(p, r)

    return responses

def evaluate_molmo(model, processor, images, prompts, labels, num_samples, batch_size):
    # move inputs to the correct device and make a batch of size 1
    # inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    images = [[image] for image in images]

    responses = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch in tqdm(range(num_batches), desc=f"Calculating accuracy (batch size: {batch_size})"):
        image_batch = images[batch * batch_size : (batch+1) * batch_size]
        prompt_batch = prompts[batch * batch_size : (batch+1) * batch_size]

        inputs = process_molmo_batch(processor, prompt_batch, image_batch)

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output = model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=1,
                    stop_sequences=["<|endoftext|>"],
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                ),
            tokenizer=processor.tokenizer,
        )

        generated_texts = processor.tokenizer.batch_decode(
            output[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )
        responses += generated_texts

    return responses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_dataset("Graphcore/vqa", 
                           split="validation",
                           trust_remote_code=True,
                           cache_dir="/projects/bdes/bcivjan/.cache/huggingface",
                           storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
    dataset = dataset.filter(lambda x: x['label']['ids'] != [] and x['answer_type'] == 'yes/no')
    print(len(dataset))

    num_samples = 1000

    # model_path = "llava-hf/llava-1.5-7b-hf"
    model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
    # model_path = "llava-hf/llama3-llava-next-8b-hf"
    # model_path = "allenai/Molmo-7B-D-0924"
    # model_path = "cyan2k/molmo-7B-D-bnb-4bit"

    if model_path == "llava-hf/llava-1.5-7b-hf":
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    elif model_path == "llava-hf/llava-v1.6-mistral-7b-hf" or model_path == "llava-hf/llama3-llava-next-8b-hf":
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    elif model_path == "allenai/Molmo-7B-D-0924" or model_path == "cyan2k/molmo-7B-D-bnb-4bit":
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
    else:
        raise AssertionError("Model path not supported")

    images = []
    prompts = []
    labels = []
    for image_id, question, dataset_labels in tqdm(zip(dataset['image_id'][:num_samples], dataset['question'][:num_samples], dataset['label'][:num_samples]),
                                                   total=num_samples,
                                                   desc="Preprocessing dataset"):

        image = Image.open(image_id)
        
        prompt = get_model_prompt_yes_no(model_path, processor, question)
        images.append(image)
        prompts.append(prompt)

        label2weight = {}
        for i, id in enumerate(dataset_labels['ids']):
            if id.isnumeric():
                id = num2words(int(id))
            label2weight[id] = dataset_labels['weights'][i]
        labels.append(label2weight)

    batch_size = 25

    if "llava" in model_path.lower():
        generated_responses = evaluate_llava(model, processor, images, prompts, labels, num_samples, batch_size)
    elif "molmo" in model_path.lower():
        generated_responses = evaluate_molmo(model, processor, images, prompts, labels, num_samples, 5)
    else:
        raise NotImplementedError

    acc = calc_avg_yes_no_accuracy(generated_responses, labels)
    print(f'{model_path} Accuracy: {acc * 100:.2f}%')

    tp, fp, tn, fn = get_yesno_accuracy_metrics(generated_responses, labels)
    print(f"{model_path} Precision: {precision(tp, fp) * 100:.2f}%")
    print(f"{model_path} Recall: {recall(tp, fn) * 100:.2f}%")
    print(f"{model_path} F1 Score: {f1_score(tp, fp, fn) * 100:.2f}%")