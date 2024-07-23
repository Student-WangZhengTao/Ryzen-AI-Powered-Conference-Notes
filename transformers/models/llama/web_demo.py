from flask import Flask, render_template, request
import torch
import time
import os
import psutil
import transformers
from transformers import AutoTokenizer, set_seed, AutoModel
import qlinear
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)

set_seed(123)
transformers.logging.set_verbosity_error()
logging.disable(logging.CRITICAL)

messages = [
    {"role": "system", "content": "你是一名专业的会议记录分析员,你的任务是根据给定的会议纪要文档或内容提取重要信息并以简洁明了的方式总结报告。你可以选择用中文或英文回答。"},
]

def generate_response_from_file(file_path, language):
    # 读取会议纪要文件内容
    meeting_minutes = ""
    encoding_list = ['utf-8', 'gb18030', 'gbk', 'big5', 'latin-1']
    for encoding in encoding_list:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                meeting_minutes = file.read()
            break
        except UnicodeDecodeError:
            continue
    if not meeting_minutes:
        return "无法解码文件内容,请检查文件是否存在或尝试其他编码格式。" if language == 'zh' else "Failed to decode file content, please check if the file exists or try other encoding formats."

    # 使用语言模型提取重要信息并生成响应
    input = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"根据会议纪要文档,请仅使用语言({language})提取重要信息并简洁地总结报告。文档内容如下:\n{meeting_minutes}"}],
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
    
    if language == 'en':
        return "Based on the meeting minutes, the key information is summarized as follows:\n\n" + response_message
    else:
        return "根据会议纪要,重要信息总结如下:\n\n" + response_message

def generate_response_from_text(meeting_minutes, language):
    # 使用语言模型提取重要信息并生成响应
    input = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"根据会议纪要内容,请仅使用语言({language})提取重要信息并简洁地总结报告。内容如下:\n{meeting_minutes}"}],
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
    
    if language == 'en':
        return "Based on the meeting minutes, the key information is summarized as follows:\n\n" + response_message
    else:
        return "根据会议纪要,重要信息总结如下:\n\n" + response_message

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        language = request.form.get('language', 'zh')
        if 'meeting_file' in request.files and request.files['meeting_file']:
            file = request.files['meeting_file']
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            response = generate_response_from_file(file_path, language)
        elif 'meeting_content' in request.form and request.form['meeting_content']:
            meeting_content = request.form['meeting_content']
            response = generate_response_from_text(meeting_content, language)
        else:
            response = "请上传会议纪要文件或直接输入会议内容。" if language == 'zh' else "Please upload the meeting minutes file or directly input the meeting content."
        return render_template('index.html', response=response, language=language)
    return render_template('index.html', language='zh')

if __name__ == '__main__':
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

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)