import os
import sys

# Ensure UTF-8 encoding on Windows to prevent Jinja/TRL import crashes
os.environ["PYTHONUTF8"] = "1"

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time

# ---------------------------------------------------------
# Page Configurations and Premium Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="MediLlama - AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium medical style CSS injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Title Gradient Styling */
.main-title {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 5px;
}
.sub-title {
    font-family: 'Inter', sans-serif;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 25px;
}

/* Premium Medical Disclaimer Alert Box */
.disclaimer-container {
    background-color: rgba(239, 68, 68, 0.08);
    border: 1px dashed rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 25px;
    color: #f87171;
    font-size: 0.88rem;
    line-height: 1.4;
    text-align: center;
}

/* Custom cards for sidebar metrics */
.metric-card {
    background-color: #1e293b;
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #334155;
    margin-bottom: 12px;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
}
.metric-val {
    color: #38bdf8;
    font-size: 1rem;
    font-weight: 700;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Panel: Configurations & Device Monitoring
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size: 1.8rem; text-align: left;">🩺 MediLlama</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title" style="font-size: 0.9rem; text-align: left; margin-bottom: 15px;">AI Medical Assistant Dashboard</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Model Loading configurations
    st.subheader("Model Configuration")
    model_id = st.text_input("Base Model ID", value="unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
    adapter_path = st.text_input("Adapter Path", value="./medical_llama3_lora")
    
    # Model Hyperparameters
    st.subheader("Generation Settings")
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.05,
                            help="Higher values make outputs more creative but less predictable.")
    top_p = st.slider("Top-P (Nucleus)", min_value=0.1, max_value=1.0, value=0.9, step=0.05,
                      help="Tokens matching top-p cumulative probability are kept.")
    max_new_tokens = st.slider("Max New Tokens", min_value=64, max_value=512, value=256, step=32,
                               help="Maximum number of tokens the model generates.")
    
    st.divider()
    
    # Mode selector
    st.subheader("Interactive Mode")
    chat_mode = st.radio(
        "Select Response Mode",
        options=["🩺 Fine-Tuned Doctor (LoRA)", "🤖 Base Model (General)", "⚖️ Compare Side-by-Side"],
        index=0
    )

# ---------------------------------------------------------
# Caching Model Loading to avoid reloading on user input
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading LLaMA model & LoRA adapter. Please wait (this can take 1-2 minutes)...")
def load_llama_model(model_id_val, adapter_path_val):
    if not torch.cuda.is_available():
        return None, None, "CUDA GPU not found. Streamlit requires a GPU to load this 3B model."
    
    device_name = torch.cuda.get_device_name(0)
    has_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if has_bf16 else torch.float16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    try:
        # Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id_val,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id_val, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load Adapter if present
        if os.path.exists(adapter_path_val):
            model = PeftModel.from_pretrained(base_model, adapter_path_val)
            adapter_loaded = True
        else:
            model = base_model
            adapter_loaded = False
            
        return model, tokenizer, {
            "device": device_name,
            "has_bf16": has_bf16,
            "adapter_loaded": adapter_loaded,
            "error": None
        }
    except Exception as e:
        return None, None, {
            "device": device_name,
            "has_bf16": has_bf16,
            "adapter_loaded": False,
            "error": str(e)
        }

# Start loading the model
model, tokenizer, metadata = load_llama_model(model_id, adapter_path)

# Update sidebar GPU statistics
with st.sidebar:
    st.divider()
    st.subheader("System Status")
    if metadata.get("error"):
        st.error(f"Error loading model: {metadata['error']}")
    else:
        # GPU Device Card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active GPU</div>
            <div class="metric-val">{metadata['device']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Compute Precision Card
        precision = "Bfloat16" if metadata['has_bf16'] else "Float16"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precision & Quantization</div>
            <div class="metric-val">{precision} / NF4 (4-bit)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Adapter Status Card
        status_txt = "Active (medical_llama3_lora)" if metadata['adapter_loaded'] else "Not Found (Using Base)"
        status_color = "#4ade80" if metadata['adapter_loaded'] else "#fca5a5"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">LoRA Adapter</div>
            <div class="metric-val" style="color: {status_color};">{status_txt}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Page Header & Layout
# ---------------------------------------------------------
st.markdown('<div class="main-title">MediLlama AI Medical Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Providing fine-tuned medical chatbot dialogue powered by LLaMA 3 QLoRA</div>', unsafe_allow_html=True)

# Medical Disclaimer Banner
st.markdown("""
<div class="disclaimer-container">
    ⚠️ <strong>Medical Disclaimer:</strong> This chatbot is an experimental AI research demonstration fine-tuned on medical dialogue datasets.
    It does not provide professional medical advice, diagnosis, or treatment. Always consult a certified healthcare professional.
</div>
""", unsafe_allow_html=True)

# Response generator helper
def generate_text(model, tokenizer, prompt, use_base_only=False):
    messages = [
        {
            "role": "system", 
            "content": "You are a professional medical assistant. Provide accurate, helpful, and concise medical information to the patient's questions."
        },
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Check if we need to temporarily disable LoRA
    if isinstance(model, PeftModel) and use_base_only:
        context_manager = model.disable_adapter()
    else:
        # Dummy context manager if not disabling
        class DummyContext:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        context_manager = DummyContext()
        
    with context_manager:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

# Initializing Session State for Chats
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------
# Dynamic Chat Rendering
# ---------------------------------------------------------
if chat_mode != "⚖️ Compare Side-by-Side":
    # Regular Chat Session
    # Display historical chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # React to user input
    if user_prompt := st.chat_input("Ask a medical question... (e.g. Symptoms of a common cold?)"):
        # Display user question
        with st.chat_message("user"):
            st.write(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
        # Display response loader
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                t_start = time.time()
                # Run with/without adapter
                use_base = (chat_mode == "🤖 Base Model (General)")
                assistant_response = generate_text(model, tokenizer, user_prompt, use_base_only=use_base)
                duration = time.time() - t_start
                
            response_placeholder.write(assistant_response)
            st.caption(f"Generated in {duration:.2f} seconds.")
            
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

else:
    # Compare Side-by-Side layout
    st.info("💡 Side-by-side mode allows you to ask a single question and see both Base Model and Fine-Tuned Model responses simultaneously.")
    
    # Large comparative input box
    compare_prompt = st.text_area("Ask a medical question to compare models:", height=80, placeholder="Describe symptoms or ask medical questions here...")
    
    if st.button("Generate Comparison", type="primary"):
        if not compare_prompt.strip():
            st.warning("Please enter a question first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 Base LLaMA 3 (Before Fine-Tuning)")
                with st.spinner("Base model thinking..."):
                    t0 = time.time()
                    base_resp = generate_text(model, tokenizer, compare_prompt, use_base_only=True)
                    t1 = time.time()
                st.markdown(f'<div style="background-color: #1e293b; border-radius: 8px; padding: 15px; border: 1px solid #334155; min-height: 180px;">{base_resp}</div>', unsafe_allow_html=True)
                st.caption(f"Generated in {t1-t0:.2f} seconds.")
                
            with col2:
                st.subheader("🩺 Fine-Tuned Medical LLaMA (After Fine-Tuning)")
                with st.spinner("Fine-tuned doctor model thinking..."):
                    t0 = time.time()
                    tuned_resp = generate_text(model, tokenizer, compare_prompt, use_base_only=False)
                    t1 = time.time()
                st.markdown(f'<div style="background-color: rgba(6, 182, 212, 0.05); border-radius: 8px; padding: 15px; border: 1px solid rgba(6, 182, 212, 0.2); min-height: 180px;">{tuned_resp}</div>', unsafe_allow_html=True)
                st.caption(f"Generated in {t1-t0:.2f} seconds.")
