#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import logging
import time
import argparse
import os
import psutil
from transformers import set_seed
from transformers import LlamaTokenizer

import qlinear
from utils import Utils
from model_utils import (
    warmup, 
    decode_prompt,
    decode_prompts,
    get_wikitext2,
    perplexity,
)
from profiler import ProfileAIE
import gc

from modeling_llama_amd import LlamaForCausalLM, LlamaAttention

from pre_quant import run_awq, apply_awq
from quantizer import real_quantize_model_weight
from qmodule import WQLinear

set_seed(123)


def load_model(args):
    tokenizer = LlamaTokenizer.from_pretrained("./llama3-8b-amd-npu")
    if args.awq == "none":
        model = LlamaForCausalLM.from_pretrained("./llama-2-wts-hf/7B_chat", torch_dtype=torch.bfloat16) 
    
    else:
        ckpt = "pytorch_llama3_8b_w_bit_4_awq_lm_amd.pt"
        print(f"Loading from ckpt: {ckpt}")
        if not os.path.exists(ckpt):
            print(f"\n\n ***** Run --task quantize (with/without lm_head) first to save quantized model ...!!! \n\n")
            raise SystemExit 
        model = torch.load(ckpt)

    Utils.print_model_size(model)
    _ = gc.collect()
    model.eval()
    model = model.to(torch.bfloat16)
    print(model)
    return model, tokenizer 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Dataset - wikitext2-raw-v1, wikitext2-v1", type=str, default="raw", choices=["non-raw", "raw"])
    parser.add_argument('--w_bit', help="weight bit size", type=int, default=3, choices=[3, 4])
    parser.add_argument('--awq', help="load awq scales, clips from pt or run awq", type=str, default="load", choices=["load", "run", "none"]) 
    parser.add_argument("--target", help="cpu, aie, aie_emu", type=str, default="cpu", choices=["cpu", "aie_emu", "aie"])
    parser.add_argument('--task', help="quantize: Apply AWQ and save ckpt; perplexity: Measure perplexity on wikitext2 dataset; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); decode: Decode set of prompts; interactive: Interactive session with the model", type=str, default="decode", choices=["quantize", "decode", "benchmark", "benchmark_long", "perplexity", "interactive"])
    parser.add_argument('--flash_attention', help="Enable flash attention", action='store_true')
    parser.add_argument('--lm_head', help="Enable PerGrp quantization of lm_head layer", action='store_true')
    parser.add_argument('--num_torch_threads', help="Number of torch threads", type=int, default=8, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    args = parser.parse_args()
    print(f"{args}")
    dev = os.getenv("DEVICE")

    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(args.num_torch_threads)
    
    log_dir = "./logs_awq_7B_chat"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_awq_7B_chat.log"

    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)

    model, tokenizer = load_model(args)

    if args.awq != "none":
        for n, m in model.named_modules():
            if isinstance(m, qlinear.QLinearPerGrp):
                print(f"Preparing weights of layer : {n}")
                m.device = "aie"
                m.quantize_weights()

    print(model)
    Utils.print_model_size(model)
    
    warmup(model, tokenizer)

    if (args.task == "decode"):
        decode_prompts(model, tokenizer, max_new_tokens=11)
        logging.shutdown()
        out_file = log_file.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        ProfileAIE.analyze_profiling(False, True, log_file, out_file)
        out_file.close()

    elif (args.task == "benchmark") or (args.task == "benchmark_long"):
        trainloader, testenc = get_wikitext2(tokenizer, nsamples=2, seqlen=4096)
        if (args.task == "benchmark"):
            seqlens =  [4, 8, 16, 32, 64, 128, 256]
        else:
            seqlens =  [512, 1024, 1536, 2048, 3000, 4096] 
        input_ids = next(iter(trainloader))[0][:, :4096]
        for seqlen in seqlens:
            logging.critical("*"*40)
            print("*"*40)
            print(f"Benchmarking for {seqlen} tokens ...")
            input_ids_test = input_ids[:, :seqlen]
            decode_prompt(model, tokenizer, prompt=None, input_ids = input_ids_test, max_new_tokens=11)
            
        logging.shutdown()
        out_file = log_file.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        ProfileAIE.analyze_profiling(False, True, log_file, out_file)
        out_file.close()

    elif (args.task == "perplexity"):
        start = time.time()
        perplexity(model, tokenizer, dataset=args.dataset)
        print(f"Time taken to measure ppl on RyzenAI: {time.time() - start}s")

    elif (args.task == "interactive"):
        print("Starting interactive session. Type 'exit' to quit.")
        while True:
            prompt = input("User: ")
            if prompt.lower() == "exit":
                break
            output = decode_prompt(model, tokenizer, prompt=prompt, max_new_tokens=512)
            print(f"Assistant: {output}")