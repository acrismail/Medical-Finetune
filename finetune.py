import os
import sys
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 3/3.2 on medical QA dataset using QLoRA.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit", 
        help="Hugging Face model ID (pre-quantized 4-bit model recommended)."
    )
    parser.add_argument(
        "--dataset_id", 
        type=str, 
        default="ruslanmv/ai-medical-chatbot", 
        help="Hugging Face dataset ID."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./medical_llama3_lora", 
        help="Directory to save checkpoints."
    )
    parser.add_argument(
        "--subset_size", 
        type=int, 
        default=5000, 
        help="Number of examples from the dataset to use for training (dataset has 256k+ entries)."
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=100, 
        help="Maximum number of training steps."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size per device."
    )
    parser.add_argument(
        "--grad_accum_steps", 
        type=int, 
        default=8, 
        help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-4, 
        help="Learning rate."
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=512, 
        help="Maximum sequence length."
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None, 
        help="Hugging Face token for gated models (can also use HF_TOKEN env var)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Hugging Face Authentication
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        print("Logging in to Hugging Face...")
        login(token=token)
        
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. A GPU is required for this training.")
        sys.exit(1)
        
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Check for bfloat16 support
    has_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if has_bf16 else torch.float16
    print(f"Bfloat16 supported: {has_bf16}. Using compute dtype: {compute_dtype}")

    # 2. BitsAndBytes 4-bit Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # 3. Load Model and Tokenizer
    print(f"Loading base model: {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=token
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, 
        token=token,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for sft/decoder-only models

    # 4. Define PEFT (QLoRA) Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("PEFT/LoRA configuration defined.")

    # 5. Load and Preprocess Dataset
    print(f"Loading dataset: {args.dataset_id}...")
    # Load raw dataset
    raw_dataset = load_dataset(args.dataset_id, split="train")
    
    # Shuffle and select a subset to speed up training
    print(f"Dataset size: {len(raw_dataset)} examples. Selecting a subset of {args.subset_size}...")
    shuffled_dataset = raw_dataset.shuffle(seed=42).select(range(args.subset_size))
    
    # Format to standard conversation format (messages)
    # LLaMA 3 Chat Template expects this message structure
    def format_conversations(example):
        return {
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a professional medical assistant. Provide accurate, helpful, and concise medical information to the patient's questions."
                },
                {"role": "user", "content": example["Patient"]},
                {"role": "assistant", "content": example["Doctor"]}
            ]
        }
        
    print("Mapping dataset to chat template format...")
    formatted_dataset = shuffled_dataset.map(format_conversations, remove_columns=raw_dataset.column_names)
    
    # Split into train and evaluation splits (90% train, 10% eval)
    split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Train split size: {len(train_dataset)}, Validation split size: {len(eval_dataset)}")

    # 6. Training Arguments (SFTConfig)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=1,
        max_steps=args.max_steps,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        fp16=not has_bf16,
        bf16=has_bf16,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=2,
        load_best_model_at_end=True,
        max_length=args.max_seq_length,
    )

    # 7. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    # 8. Start Fine-Tuning
    print("Starting fine-tuning...")
    trainer.train()
    
    # 9. Save Trained Adapter
    print(f"Saving final trained adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
