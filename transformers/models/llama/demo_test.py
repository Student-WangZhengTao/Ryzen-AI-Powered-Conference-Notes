import torch
import time
import os
import psutil
import transformers
from transformers import AutoTokenizer, set_seed, AutoModel
import qlinear
import logging

set_seed(123)
transformers.logging.set_verbosity_error()
logging.disable(logging.CRITICAL)

messages = [
    {"role": "system", "content": "你是一个中文会议纪要小助手，主要功能是提炼会议纪要和翻译文字，主要文字是中文。"},
]

def generate_response(prompt):
    start_time = time.time()
    input = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    outputs = model.generate(input['input_ids'],
                            max_new_tokens=600,
                            eos_token_id=terminators,
                            attention_mask=input['attention_mask'],
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9)

    response = outputs[0][input['input_ids'].shape[-1]:]
    response_message = tokenizer.decode(response, skip_special_tokens=True)
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"响应时间: {generation_time:.2f} 秒")
    return response_message

if __name__ == "__main__":
    p = psutil.Process()
    p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(4)

    tokenizer = AutoTokenizer.from_pretrained("llama3-8b-amd-npu")
    ckpt = "llama3-8b-amd-npu/pytorch_llama3_8b_w_bit_4_awq_lm_amd.pt"
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    model = torch.load(ckpt)
    model.eval()
    model = model.to(torch.bfloat16)

    for n, m in model.named_modules():
        if isinstance(m, qlinear.QLinearPerGrp):
            print(f"Preparing weights of layer : {n}")
            m.device = "aie"
            m.quantize_weights()

    print("系统: " + messages[0]['content'])

    while True:
        user_prompt = input("用户: ")
        if user_prompt.lower() == "exit":
            break
        response = generate_response(user_prompt)
        print("助手: " + response)
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "system", "content": response})