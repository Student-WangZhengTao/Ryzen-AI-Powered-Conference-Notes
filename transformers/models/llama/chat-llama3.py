import torch
import time
import os
import psutil
import transformers
from transformers import AutoTokenizer, set_seed, AutoModel
import qlinear
import logging
from flask import Flask, request, jsonify

set_seed(123)
transformers.logging.set_verbosity_error()
logging.disable(logging.CRITICAL)

app = Flask(__name__)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
]

def generate_response(prompt):
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
    return response_message

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data['prompt']
    response = generate_response(prompt)
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "system", "content": response})
    return jsonify({'response': response})

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

    print("system: " + messages[0]['content'])

    app.run(host='0.0.0.0', port=5000)