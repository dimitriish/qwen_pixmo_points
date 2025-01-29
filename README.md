# Fine-tuning and Evaluation of Qwen2-VL Model

This repository contains scripts for fine-tuning and evaluating the Qwen2-VL-2B-Instruct model for pointing task. Pointing task is very important now, because recent agent systems can use VLMs to interact with any screens.

## Data
We utilized **allenai/pixmo-pointing** to create instruction dataset similar to Molmo model training. As an output of the model we expect parsable html with coordinates of the searched object scaled in 0-100. Here is an example:

 - **Question**: 'Mark the dark storm clouds in this image.'
 - **Answer**: '<points x0="37.77" y0="3.86"" alt="dark storm clouds">dark storm clouds</points>'

## Installing dependencies

```bash
pip install -r requirements -U
```

Additionally, make sure your system has sufficient GPU memory for training and evaluation. We used A100 GPU for training.

---

## Fine-tuning the Model

### Script: `finetune.py`

This script fine-tunes the Qwen2-VL-2B model using `SFTTrainer`. It supports dataset loading and tracks experiments using `wandb`.

### Usage

```bash
python finetune.py
```

### Features
- Supports LoRA for efficient fine-tuning.
- Uses `wandb` for tracking training progress.
- Loads datasets using Hugging Face's `datasets` library.
- Implements memory management functions to optimize GPU usage.

---

## Evaluating the Model

### Notebook: `eval.ipynb`
It contains inference code for finetuned model

After training model always generates parsable html. Unfortunately, there is no benchmarks for evaluating the model, so we tested it manually. Actually it's obvious that model finds objects with precision higher than just random. But it needs much more training for getting good model. Here is an example:

 - **Question**: 'Mark the dark storm clouds in this image.'
 - **Answer**: '<points x0="51.17" y0="11.17" alt="dark storm clouds">dark storm clouds</points>'

![image](https://github.com/user-attachments/assets/306a8780-e352-4336-acf7-bd902d4d4b45)

