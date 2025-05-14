import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math

from datasets import load_dataset

def eval_model(args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    questions = load_dataset(args.dataset, split="test")
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for idx, line in tqdm(enumerate(questions)):
        dataset_name = args.dataset
        gt = line["answer"]

        image = line["image"]
        qs = line["question"]

        args.conv_mode = "llm_jp"

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], (qs, [image], None))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        cur_prompt = prompt

        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

        image_tensors = []
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensors.append(image_tensor.half().cuda())
        # image_tensors = torch.cat(image_tensors, dim=0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "dataset": dataset_name,
                                   "sample_id": str(idx),
                                   "prompt": cur_prompt,
                                   "pred_response": outputs,
                                   "gt_response": gt,
                                   "shortuuid": ans_id,
                                   "model_id": model_name,
                                   }, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/checkpoints/llava-onevision-siglip-so400m-patch14-384-llm-jp-3-7.2b-staircaptions-japanese_visualgenome")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="SakanaAI/JA-VLM-Bench-In-the-Wild")
    parser.add_argument("--answers-file", type=str, default="/output/eval/answers.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llm_jp")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=int, default=1.05)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)