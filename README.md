# Applied Transfer Learning & LLM Fine-Tuning Labs 

This repository contains a collection of Jupyter Notebooks focusing on Transfer Learning, Large Language Models (LLMs), and Computer Vision. The projects range from fundamental concepts of text generation and prompt engineering to advanced efficient fine-tuning techniques (PEFT, QLoRA) applied to medical and general domains.

## Repository Contents

The notebooks are organized by topic, covering the transition from basic usage of Transformers to training custom models on specific datasets.

| File | Category | Description |
| :--- | :--- | :--- |
| `TL06_LLMS.ipynb` | LLM Basics | Introduction to text generation strategies (greedy, beam search), prompt engineering, and benchmarking using Qwen. |
| `TL07_CHATS.ipynb` | Chat & RAG | Implementation of chat templates, tool usage, and Retrieval-Augmented Generation (RAG) pipelines using Transformers. |
| `TL08_TRAINER.ipynb` | Vision | Fine-tuning a Vision Transformer (ViT) for image classification on the Food101 dataset. |
| `TL09_PEFT.ipynb` | Optimization | Introduction to PEFT (Parameter-Efficient Fine-Tuning). Demonstrates LoRA (Low-Rank Adaptation) on simple MLPs vs. full training. |
| `CardioLLM_Finetuning.ipynb` | Medical AI | A complete workflow for fine-tuning Llama 3.1 8B (4-bit quantized) using QLoRA for the specialized Cardiology domain. |
| `TL10_EXAM.ipynb` | Assessment | Practical exam notebook applying Transfer Learning concepts to the Canadian-streetview-cities dataset. |

## Key Concepts & Technologies

This repository demonstrates practical implementations of the following technologies:

* **Libraries:** PyTorch, Hugging Face Transformers, PEFT, BitsAndBytes, Unsloth.
* **Techniques:**
    * Supervised Fine-Tuning (SFT) for specialized tasks.
    * QLoRA (Quantized LoRA) for memory-efficient training.
    * RAG (Retrieval-Augmented Generation) for context-aware chat.
    * Vision Transformers (ViT) for image classification.
* **Models Used:** Llama 3.1, Qwen, ViT-Base.

## Getting Started

Most notebooks are designed to run in Google Colab with GPU support (T4 or A100 is recommended for Llama 3 training).

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/transfer-learning-llm-vision.git](https://github.com/YOUR_USERNAME/transfer-learning-llm-vision.git)
    ```
2.  Open any `.ipynb` file in Jupyter Lab, VS Code, or upload it directly to Google Colab.
3.  Install dependencies at the beginning of each notebook.

## Highlights

### CardioLLM Fine-Tuning
The `CardioLLM_Finetuning.ipynb` is a standout project that shows how to take a general-purpose 8B model and specialize it for medical inquiries using minimal hardware resources via 4-bit quantization.

### Vision Transformers
`TL08_TRAINER.ipynb` provides a clear template for adapting pre-trained Google ViT models to custom image datasets, showcasing the power of Transfer Learning in Computer Vision.

---
*Disclaimer: These notebooks are for educational and research purposes.*
