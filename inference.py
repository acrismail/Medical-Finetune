import os
import sys
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLaMA 3 model.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit", 
        help="Base model ID used for fine-tuning."
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        default="./medical_llama3_lora", 
        help="Path to the saved LoRA adapter."
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None, 
        help="Hugging Face token."
    )
    return parser.parse_args()

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    messages = [
        {
            "role": "system", 
            "content": "You are a professional medical assistant. Provide accurate, helpful, and concise medical information to the patient's questions."
        },
        {"role": "user", "content": prompt}
    ]
    
    # Format using LLaMA 3 Chat Template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
    # Extract only the newly generated tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def main():
    args = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. GPU is required for inference.")
        sys.exit(1)
        
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Check for bfloat16 support
    has_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if has_bf16 else torch.float16
    
    # 1. Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    # 2. Load Base Model and Tokenizer
    print(f"Loading Base Model: {args.model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=token
    )
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, 
        token=token,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Sample questions to compare before/after
    test_prompts = [
        "What are the typical symptoms of a migraine, and how does it differ from a tension headache?",
        "My child has a mild fever (38C) and a runny nose. When should I contact a pediatrician?"
    ]
    
    print("\n" + "="*50)
    print("TESTING BASE MODEL (Before Fine-Tuning)")
    print("="*50)
    
    base_responses = []
    for i, prompt in enumerate(test_prompts):
        print(f"\n[Prompt {i+1}]: {prompt}")
        response = generate_response(base_model, tokenizer, prompt)
        base_responses.append(response)
        print(f"[Base Model Response]:\n{response}")
        print("-" * 40)
        
    # 3. Load LoRA Adapter
    if not os.path.exists(args.adapter_path):
        print(f"\nWARNING: Adapter path '{args.adapter_path}' does not exist.")
        print("We cannot compare with fine-tuned model because it hasn't been trained yet.")
        print("To run fine-tuning first, execute: python finetune.py")
        sys.exit(0)
        
    print(f"\nLoading LoRA Adapter from: {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    print("Adapter successfully loaded.")
    
    print("\n" + "="*50)
    print("TESTING FINE-TUNED MODEL (After Fine-Tuning)")
    print("="*50)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n[Prompt {i+1}]: {prompt}")
        print(f"[Before Fine-Tuning]:\n{base_responses[i]}")
        print("-" * 20)
        response = generate_response(model, tokenizer, prompt)
        print(f"[After Fine-Tuning]:\n{response}")
        print("-" * 40)
        
    # 4. Interactive loop
    print("\n" + "="*50)
    print("INTERACTIVE MEDICAL CHAT MODE (Type 'exit' to quit)")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nPatient question: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                break
                
            print("\nThinking...")
            response = generate_response(model, tokenizer, user_input)
            print(f"\nDoctor's Response:\n{response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
if __name__ == "__main__":
    main()
