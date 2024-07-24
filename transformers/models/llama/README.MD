# Llama - 3 - 8b - pytorch

<p align="center">
  <img src="https://github.com/Student-WangZhengTao/Ryzen-AI-Powered-Conference-Notes/blob/main/transformers/models/llama/llama.jpg" alt="Llama">
</p>

**_NOTE:_**  Please ensure that you followed the environment setup instructions from the [Transformer folder readme](../../README.md) before following the steps here.



### Prepare Llama3 Weights to use with HF

The weights of Llama-3 models can be obtained by requesting permission with Meta. Check this [Huggingface page](https://huggingface.co/docs/transformers/main/model_doc/llama3) on Llama-3 for details. 

<p align="center">
  <img src="https://github.com/Student-WangZhengTao/Ryzen-AI-Powered-Conference-Notes/blob/main/transformers/models/llama/original.png" alt="Llama">
</p>

Once weights are obtained, use Huggingface's converter to convert the weights to be compatible to be loaded with HF interface. 

```

# Run the converter
$ python <condainstall>/envs/ryzenai-transformers/lib/python3.9/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ./llama-3-wts/ --model_size 8B --output_dir ./llama-3-wts-hf/8B

```

<p align="center">
  <img src="https://github.com/Student-WangZhengTao/Ryzen-AI-Powered-Conference-Notes/blob/main/transformers/models/llama/update.png" alt="Llama">
</p>


## Quantization

### w4abf16 with AWQ + PerGrp Quantization

AWQ enables 3-bit and 4-bit weights for LLMs. This reduces model size of Llama3 8B from 52-58% of int8 model depending on group size and whether the last layer is quantized. 

AWQ scales are obtained from MIT-Han-Lab. THe script also contains hooks to calculate scales instead of using precomputed scales. All layers other than "lm_head" are quantized using AWQ. This software stack of RyzenAI can also quantize lm_head layer using per group quantization scheme, with group sizes varying from 32-256. 

Linear layers are replaced with QLinearPerGrp custom int4 compute layer afetr AWQ.

Matmul grouping in done when flash attention is enabled, which reduces number of dispatches to AIE by grouping matmuls of QKV matrices into a single grouped matmul.
This in addition to static memory allocation in token phases provides 8-10% better performance than vanilla attention.


#### Save AWQ checkpoints
4-bit AWQ has higher perplexity than 3-bit AWQ with same performance.

##### 4-bit AWQ

```
AWQ + Quantize lm_head
python run_awq.py --w_bit 4 --task quantize --lm_head --flash_attention
```

## Deployment

```
python run_awq.py --task decode --target aie --w_bit 4

****************************************
prompt: What is the meaning of life?
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: What is the meaning of life? This is a question that has puzzled philosophers, theologians, scientists, and many others for
****************************************
prompt: Tell me something you don't know.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: Tell me something you don't know.
The only thing I can think of is that I don't know how to make you disappear
****************************************
*
```

## Results

#### Latency : Ryzen9 7945HS
```
python web_demo.py
```


### Features
1. w4a16: AWQ on CPU, AIE **(AWQ for Llama 3 his from MIT-Han-lab)**
4. Static memory allocation for token phase speed up
5. Matmul grouping for Llama3Attn
6. Model save and load from checkpoints
7. Profiling instrumentation to measure prefill/token times during mission mode



### Support modes on NPU

Currently the supported precision on NPU is "w4abf16" and "w4abf16 + FA". **w4abf16** uses **AWQ** PerGrp quantization
