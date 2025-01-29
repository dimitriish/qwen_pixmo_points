import os
import gc
import time
import torch
import wandb
from datasets import load_dataset, load_from_disk
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
    AutoProcessor
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info

def clear_memory():
    """Clears GPU and CPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def format_data(sample, system_message):
    """Formats a single dataset sample into the chatbot structure."""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image_url"],
                },
                {
                    "type": "text",
                    "text": sample['question'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["answer"]
                }
            ],
        },
    ]


def collate_fn(examples, processor, image_token_ids):
    """Collates and preprocesses a batch of examples."""
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example)[0] for example in examples]

    batch = processor(
        text=texts, 
        images=image_inputs, 
        return_tensors="pt", 
        padding=True
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    for image_token_id in image_token_ids:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels

    return batch

def main():
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    DATASET_ID = "data/generated_dataset"
    OUTPUT_DIR = "qwen2-2b-pointing"
    PROJECT_NAME = "qwen2-2b-pointing"

    # System message for the VLM
    system_message = """You are a Vision Language Model specialized in pointing to searched object
    Poiniting uses HTML-like format. (x,y) coordinates are scaled to 0-100. For a single point, the format is:
<point x="10.0" y="10.0" alt="alt text">Inline text</point>
For multiple points the format is:
<points x1="10.0" y1="10.0" x2="20.0" y1="20.0" ...  alt="alt text">Inline text</points>
It is also possble that searched object is not presented"""

    clear_memory()

    print("Loading dataset...")
    train_dataset = load_from_disk(DATASET_ID)
    print("Formatting dataset...")
    train_dataset = [format_data(sample, system_message) for sample in train_dataset]

    print("Loading model and processor...")


    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )

    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )

    
    model.gradient_checkpointing_enable()

    print("Setting up training configuration...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        max_steps=2000,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    training_args.remove_unused_columns = False
    training_args.deepspeed = "deepspeed_config.json"

    print("Initializing Weights & Biases...")
    wandb.init(
        project=PROJECT_NAME,
        name=PROJECT_NAME,
        config=training_args.__dict__,
    )


    print("Setting up data collator...")
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652, 151653, 151655]  # Specific to Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

    def custom_collate_fn(examples):
        return collate_fn(examples, processor, image_tokens)


    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
        tokenizer=processor.tokenizer,
    )


    print("Starting training...")
    trainer.train()

    print("Saving the model...")
    trainer.save_model(training_args.output_dir)
    wandb.finish()
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()