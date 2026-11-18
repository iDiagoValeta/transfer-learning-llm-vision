# transfer-learning-llm-vision

**Jupyter notebooks on transfer learning with Hugging Face: text generation, chat templates, tool calling, ViT image classification, LoRA, and a QLoRA fine-tune of Llama 3.1 8B on cardiology Q&A.**

> **Context.** Built in December 2025 as lab work for the APR course (ETSINF, DSIC, UPV) plus one personal notebook, when I was finishing my Computer Engineering degree at UPV while working at Grupo CEU. These were my first hands-on fine-tuning exercises with transformers, and they came before the local RAG work in MonkeyGrab and the tools I now build with coding agents. It is kept as it was submitted; the code is not maintained.

## What it does

- `TL06_LLMS.ipynb`: text generation with `Qwen/Qwen3-0.6B` (greedy, sampling, beam search), zero-shot, few-shot and chain-of-thought prompting, and a benchmark example on MMLU-ProX (Spanish).
- `TL07_CHATS.ipynb`: chats with `TextGenerationPipeline`, `apply_chat_template`, custom Jinja2 templates, a `QwenChatbot` class, and tool calling with a weather function (wttr.in), using `Qwen/Qwen3-4B`.
- `TL08_TRAINER.ipynb`: full fine-tuning of `google/vit-base-patch16-224-in21k` with the `Trainer` API, first on a 5,000-image subset of Food101 and then on CIFAR10.
- `TL09_PEFT.ipynb`: LoRA with `peft`, first on a toy MLP (with and without LoRA), then on the same ViT for Food101 and CIFAR10.
- `TL10_EXAM.ipynb`: the APR lab exam of 17-12-2025: LoRA fine-tuning of the ViT on `canada-guesser/Canadian-streetview-cities` (15 Canadian cities).
- `CardioLLM_Finetuning.ipynb`: QLoRA supervised fine-tuning of `unsloth/Meta-Llama-3.1-8B-bnb-4bit` on 11 hand-written cardiology instruction/response examples, with a before and after comparison on 5 unseen questions.

## How it works

The TL notebooks follow a common pattern: load a pre-trained checkpoint from the Hugging Face Hub, preprocess the data with its image processor or tokenizer, and adapt it to a new task. For vision, `Trainer` fine-tunes the whole ViT (TL08) or only low-rank adapters on the attention layers plus the classifier head (TL09, TL10). For text, TL06 and TL07 only use the models for inference to explore generation and chat features.

`CardioLLM_Finetuning.ipynb` loads Llama 3.1 8B in 4-bit, adds LoRA adapters (r=16, all attention and MLP projections) with Unsloth, and trains with `SFTTrainer` for 80 steps (batch size 2, gradient accumulation 4, learning rate 2e-4, bf16). It generates answers to the same questions before and after training and compares them by response length and by how many cardiology terms from a fixed list appear. The notebooks' own text is in Spanish (TL notebooks) and English (CardioLLM).

## Results

All numbers are printed in the notebook outputs saved in this repo; none were recomputed.

| Notebook | Setup | Printed result |
| :--- | :--- | :--- |
| `TL08_TRAINER` | ViT full fine-tuning, Food101 (5,000 images), 5 epochs | eval accuracy 0.58, train time 0:16:49 |
| `TL08_TRAINER` | ViT full fine-tuning, CIFAR10, 3 epochs | eval accuracy 0.987, train time 1:33:23 |
| `TL09_PEFT` | LoRA r=16, Food101, 10 epochs, 667,493 trainable of 86,543,818 params (0.77%) | eval accuracy 0.782, train time 0:10:29 |
| `TL09_PEFT` | LoRA, CIFAR10 (5,000 train, 1,000 test), 10 epochs | eval accuracy 0.935, train time 0:07:43 |
| `TL10_EXAM` | LoRA, 601,359 trainable of 86,411,550 params (0.70%) | trainable parameter count only; no eval metric is saved |
| `CardioLLM_Finetuning` | QLoRA, 41,943,040 trainable of 8,072,204,288 params (0.52%), 80 steps | trained in 304 s on an RTX 4060 Laptop (8 GB); mean answer length 696 to 879 chars, cardiology terms 5 to 7 |

The CardioLLM numbers come from a tiny training set and a crude length and keyword count, so they show the pipeline working, not a validated medical model. The notebook's closing message about a 26% quality improvement is its own wording, not a clinical evaluation. Do not use this model for medical advice.

## Run it

The notebooks have no requirements file; each one installs what it needs in its first cells (`!pip install ...`). Not verified by re-running here.

1. Clone the repo and open a notebook in Jupyter, VS Code or Google Colab. TL06 and the ViT notebooks were saved with a Colab T4 GPU; CardioLLM was last run on a local RTX 4060 Laptop (8 GB).
2. Suggested order: `TL06_LLMS`, `TL07_CHATS`, `TL08_TRAINER`, `TL09_PEFT`, `TL10_EXAM`, then `CardioLLM_Finetuning`.
3. Datasets (Food101, CIFAR10, MMLU-ProX, Canadian-streetview-cities) and models (Qwen3, ViT, Llama 3.1 4-bit) download from the Hugging Face Hub on first run. TL08 on CIFAR10 took about 1.5 hours in the saved run.

## Repository layout

```
TL06_LLMS.ipynb              text generation, prompting, benchmarking
TL07_CHATS.ipynb             chat templates, tools, chatbot
TL08_TRAINER.ipynb           ViT fine-tuning with Trainer (Food101, CIFAR10)
TL09_PEFT.ipynb              LoRA on an MLP and on ViT (Food101, CIFAR10)
TL10_EXAM.ipynb              APR lab exam, LoRA on Canadian street views
CardioLLM_Finetuning.ipynb   QLoRA on Llama 3.1 8B with Unsloth
```

## Stack

Python, PyTorch, Hugging Face Transformers, Datasets, PEFT, TRL, Unsloth, bitsandbytes, Google Colab.
