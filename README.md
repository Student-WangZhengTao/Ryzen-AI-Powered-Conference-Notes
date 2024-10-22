# Ryzen-AI-Powered-Conference-Notes

<p align="center">
  <img src="https://github.com/user-attachments/assets/c930b2d1-0727-4a98-90d8-39d102a778d8" alt="Ryzen AI Powered Conference Notes">
</p>

This project aims to develop a high-performance, low-power conference note-taking system using the AMD Ryzen AI processor and Ryzen AI software platform. The system utilizes the HuggingFace LLaMA3-8B-Struct pre-trained model, optimized with w4abf16 quantization algorithm and AWQ+PerGrp methods, and is ultimately deployed on AMD Ryzen AI-powered PC devices. This provides users with fast and accurate conference note generation services. The project leverages the hardware acceleration capabilities of the Ryzen AI processor and the intelligent load optimization of the Ryzen AI software to ensure excellent performance under low power conditions.

## System Interface

![image](https://github.com/user-attachments/assets/65531f6d-f987-479f-99ad-c77e1cb49ef3)


## Bill of Materials

The following is a list of components for creating the Ryzen AI Conference Notes system:

### Hardware:

![5d16a3c5119d9a42e44e61d3c35306d](https://github.com/user-attachments/assets/3b094298-1c52-4009-87f8-41be783c4a18)

1. UM790 Pro
2. CPU AMD Ryzen 9 7940HS Processor
3. GPU AMD Radeon 780M
4. System Memory 16GB x 2
5. Storage 512GB

### Software:

1. Windows 11 Professional Operating System x 1
2. AMD Ryzen AI Software Platform x 1
3. HuggingFace LLaMA3-8B-Struct Pre-trained Model x 1
4. w4abf16 Quantization Algorithm x 1
5. AWQ + PerGrp Quantization Optimization Tools x 1
6. ONNX Runtime x 1
7. Conference Notes Generation Application x 1 (self-developed)

### Additionally, the following tools and resources are required:

1. Development Tools: Python, PyTorch, TensorFlow, ONNX Runtime
2. Testing Device: Laptop x 1
3. Conference Record Sample Dataset x 1
4. Documentation Tools: Microsoft Office Suite

With this complete hardware and software setup, we can fully exploit the performance advantages of the AMD Ryzen AI processor, deploy the optimized LLaMA model, and achieve efficient and accurate conference notes generation. The system design focuses on performance, power efficiency, and usability, providing users with a highly efficient conference recording experience.

## Complete Description (BOM)

### Schematics

<p align="center">
  <img src="https://github.com/user-attachments/assets/e6c610c7-9d7e-4624-9281-305d1cf45e20" alt="Schematics">
</p>

First, we install and configure the Ryzen AI software platform on a PC equipped with an AMD Ryzen AI processor. The Ryzen AI software provides a set of libraries and tools optimized for the Ryzen AI processor, helping us to fully leverage the hardware's performance potential.

Next, we prepare the pre-trained LLaMA3-8B-Struct model. We use the open-source model provided by HuggingFace as the foundation, then fine-tune and optimize it for the conference recording application. To further enhance deployment efficiency and performance, we use the w4abf16 quantization algorithm and advanced quantization techniques like AWQ+PerGrp to compress and accelerate the model.

After optimizing the model, we use the ONNX runtime provided by the Ryzen AI software to deploy the model on AMD Ryzen AI-powered PC devices. The Ryzen AI software can automatically detect hardware resources and intelligently optimize the AI task load, ensuring excellent performance under low power consumption conditions.

Finally, we design a user-friendly conference notes generation application that integrates the optimized LLaMA model. Users can start the application during a conference and submit the meeting data, and the system will automatically generate high-quality meeting notes, significantly improving work efficiency.

By fully leveraging the powerful capabilities of the AMD Ryzen AI processor and the Ryzen AI software platform, our "Ryzen AI-Powered Conference Notes" system can provide users with fast, accurate, and low-power conference recording services, offering a new experience for enterprise office work and collaboration.

## Using System
    
To run the demo in this repository, please follow the instructions of README.md in each directory. 

- [Environment Configuration](transformers)

- [Run LLM Llama 3 model with PyTorch on Ryzen AI](transformers/models/llama)

## Reference

- [Ryzen™ AI Developer Guide](https://ryzenai.docs.amd.com/en/latest)
- [ONNXRuntime Vitis-AI EP](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- [AMD AI Developer Forum](https://community.amd.com/t5/ai/ct-p/amd_ai)

