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

<p align="center">
  <img src="https://github.com/Student-WangZhengTao/Ryzen-AI-Powered-Conference-Notes/blob/main/transformers/models/llama/weight.png" alt="Llama">
</p>

##### 4-bit AWQ

```
# AWQ + Quantize lm_head

$ python run_awq.py --w_bit 4 --task quantize --lm_head --flash_attention
```

#### Save AWQ checkpoints

4-bit AWQ has higher perplexity than 3-bit AWQ with same performance.

<p align="center">
  <img src="https://github.com/Student-WangZhengTao/Ryzen-AI-Powered-Conference-Notes/blob/main/transformers/models/llama/quant_weight.png" alt="Llama">
</p>

## Test

```
python run_awq.py --task decode --target aie --w_bit 4
```
<p align="center">
  <img src="https://github.com/Student-WangZhengTao/Ryzen-AI-Powered-Conference-Notes/blob/main/transformers/models/llama/test.png" alt="Llama">
</p>

## Operation

```
python web_demo.py
```

<p align="center">
  <img src="https://github.com/Student-WangZhengTao/Ryzen-AI-Powered-Conference-Notes/blob/main/transformers/models/llama/Result.png" alt="Llama">
</p>

## Result

<p align="center">
  <img src="https://github.com/Student-WangZhengTao/Ryzen-AI-Powered-Conference-Notes/blob/main/transformers/models/llama/System.png" alt="Llama">
</p>

**_NOTE:_**  The supported precision on NPU is "w4abf16" and "w4abf16 + FA". **w4abf16** uses **AWQ** PerGrp quantization
