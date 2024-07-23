# Ryzen-AI-Powered-Conference-Notes

This project aims to develop a high-performance, low-power conference note-taking system using the AMD Ryzen AI processor and Ryzen AI software platform. The system utilizes the HuggingFace LLaMA3-8B-Struct pre-trained model, optimized with w4abf16 quantization algorithm and AWQ+PerGrp methods, and is ultimately deployed on AMD Ryzen AI-powered PC devices. This provides users with fast and accurate conference note generation services. The project leverages the hardware acceleration capabilities of the Ryzen AI processor and the intelligent load optimization of the Ryzen AI software to ensure excellent performance under low power conditions.

## Transformers

This repository consists of methods to run Transformers in PyTorch and ONNX with operators dispatch to NPU

## Setup Transformers

### Step 1: Setup environment

Open Anaconda prompt on Windows PC.

```
git clone https://github.com/amd/RyzenAI-SW.git
cd RyzenAI-SW\example\transformers
conda env create --file=env.yaml
conda activate ryzenai-transformers

cd RyzenAI-SW\example\transformers\
setup.bat
```

AWQ Model zoo has precomputed scales, clips and zeros for various LLMs including OPT, Llama. Get the precomputed results:

```
git lfs install
cd RyzenAI-SW\example\transformers\ext
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

