## Transformers

This repository consists of methods to run Transformers in PyTorch and ONNX with operators dispatch to NPU

## Setup Transformers

### Step 1: Setup environment

Open Anaconda prompt on Windows PC.

```
conda env create --file=env.yaml
conda activate ryzenai-transformers

setup.bat
```

AWQ Model zoo has precomputed scales, clips and zeros for various LLMs including OPT, Llama. Get the precomputed results:

```
git lfs install
git clone https://huggingface.co/datasets/mit-han-lab/awq-model-zoo awq_cache
```

### Step 2: Build dependencies

```
pip install ops\cpp --force-reinstall
```

### Step 3: Install ONNX EP for running ONNX based flows

Download [Ryzen-AI Software package](https://ryzenai.docs.amd.com/en/latest/manual_installation.html#download-the-package) and extract.

**_NOTE:_** This step is not required for Pytorch based flows

```
pip install onnxruntime
cd ryzen-ai-sw-1.1\ryzen-ai-sw-1.1\voe-4.0-win_amd64
pip install voe-0.1.0-cp39-cp39-win_amd64.whl
pip install onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl
python installer.py
```
