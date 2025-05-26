import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
import re
import os
import requests
import pickle
from urllib.parse import urlparse

# --- Configuration ---
ORIGINAL_MODEL_NAME = "vinai/phobert-base"
LOCAL_CONFIG_TOKENIZER_PATH = "./phobert_local_files"

# Fixed URL - use the direct download link from Hugging Face
MODEL_URL = "https://huggingface.co/binhkhang3169/phoBERTtoxic/resolve/main/model_best_valacc.pt"
MODEL_FILENAME = "model_best_valacc.pt"

MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions ---
def clean_text(text):
    """Clean and preprocess Vietnamese text"""
    # Remove unwanted characters (keep letters, numbers, Vietnamese diacritics)
    text = re.sub(r"[^\w\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỳỵỷỹýÝỲỴỶỸ]", ' ', text)
    
    # Convert to lowercase
    text = text.lower()

    # Remove pronouns (optional)
    pronouns_to_remove = ["bạn", "tôi", "cậu", "tớ", "mình"]
    pattern = r'\b(' + '|'.join(re.escape(pronoun) for pronoun in pronouns_to_remove) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Model Definition ---
class BERTClassifier(torch.nn.Module):
    def __init__(self, num_labels=2, config_path=LOCAL_CONFIG_TOKENIZER_PATH, architecture_name=ORIGINAL_MODEL_NAME):
        super(BERTClassifier, self).__init__()
        
        # Load config
        try:
            if os.path.exists(config_path):
                bert_classifier_config = RobertaConfig.from_pretrained(
                    config_path,
                    num_labels=num_labels,
                    output_hidden_states=False,
                )
                st.info(f"✅ Loaded config from local path: {config_path}")
            else:
                raise FileNotFoundError("Local config not found")
        except Exception as e:
            st.warning(f"⚠️ Loading config from Hugging Face: {architecture_name}")
            bert_classifier_config = RobertaConfig.from_pretrained(
                architecture_name,
                num_labels=num_labels,
                output_hidden_states=False,
            )

        # Initialize model architecture
        self.bert_classifier = RobertaForSequenceClassification.from_pretrained(
            architecture_name,
            config=bert_classifier_config
        )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert_classifier(
            input_ids=input_ids,
            token_type_ids=None,  # RoBERTa doesn't use token_type_ids
            attention_mask=attention_mask,
            labels=labels
        )
        return output

# --- Download Function ---
def download_file_with_progress(url, destination_path, chunk_size=8192):
    """Download file with Streamlit progress bar"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_path) if os.path.dirname(destination_path) else '.', exist_ok=True)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded_size = 0
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = min(downloaded_size / total_size, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded: {downloaded_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
        
        progress_bar.empty()
        status_text.success(f"✅ Download completed: {os.path.basename(destination_path)}")
        return True
        
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        if os.path.exists(destination_path):
            os.remove(destination_path)
        return False

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer with caching"""
    
    with st.spinner("🔄 Loading model and tokenizer..."):
        local_model_path = MODEL_FILENAME

        # Download model if not exists
        if not os.path.exists(local_model_path):
            st.info(f"📥 Downloading model from Hugging Face...")
            if not download_file_with_progress(MODEL_URL, local_model_path):
                st.error("❌ Failed to download model. Please check your internet connection.")
                st.stop()
        else:
            st.success(f"✅ Found local model: {local_model_path}")

        # Load tokenizer
        try:
            if os.path.exists(LOCAL_CONFIG_TOKENIZER_PATH):
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_CONFIG_TOKENIZER_PATH, use_fast=False)
                st.success(f"✅ Loaded tokenizer from local path")
            else:
                tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME, use_fast=False)
                st.success(f"✅ Loaded tokenizer from Hugging Face")
        except Exception as e:
            st.error(f"❌ Failed to load tokenizer: {str(e)}")
            st.stop()

        # Load model
        try:
            # Initialize model structure
            model = BERTClassifier(num_labels=2)
            
            # Load checkpoint
            st.info(f"📂 Loading checkpoint from: {local_model_path}")
            
            try:
                checkpoint = torch.load(local_model_path, map_location=DEVICE, weights_only=False)
            except TypeError:
                # For older PyTorch versions
                checkpoint = torch.load(local_model_path, map_location=DEVICE)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                st.success("✅ Loaded model_state_dict from checkpoint")
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
                st.success("✅ Loaded state_dict directly from checkpoint")
            else:
                st.error("❌ Unrecognized checkpoint format")
                st.stop()
            
            model = model.to(DEVICE)
            model.eval()
            st.success("✅ Model loaded and ready for inference")
            
            return model, tokenizer
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.exception(e)
            st.stop()

# --- Main App ---
def main():
    st.set_page_config(
        page_title="Vietnamese Toxic Comment Detector",
        page_icon="🛡️",
        layout="centered"
    )
    
    st.title("🛡️ Vietnamese Toxic Comment Detector")
    st.markdown("Enter Vietnamese text below to check if it contains toxic content.")
    st.markdown(f"**Device:** `{DEVICE}` | **Max Length:** `{MAX_LEN}`")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Initialize session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # User input
    user_input = st.text_area(
        "Enter your Vietnamese comment here:",
        height=150,
        placeholder="Nhập bình luận tiếng Việt của bạn ở đây...",
        max_chars=500
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.prediction_history = []
        st.rerun()
    
    if analyze_button:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔄 Analyzing..."):
                # Clean text
                cleaned_text = clean_text(user_input)
                
                # Tokenize
                inputs = tokenizer(
                    cleaned_text,
                    add_special_tokens=True,
                    max_length=MAX_LEN,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                input_ids = inputs['input_ids'].to(DEVICE)
                attention_mask = inputs['attention_mask'].to(DEVICE)
                
                # Predict
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
                
                # Results
                is_toxic = predicted_class.item() == 1
                confidence_score = confidence.item()
                
                # Store in history
                result = {
                    'text': user_input,
                    'is_toxic': is_toxic,
                    'confidence': confidence_score,
                    'cleaned_text': cleaned_text
                }
                st.session_state.prediction_history.insert(0, result)
                
                # Keep only last 5 results
                if len(st.session_state.prediction_history) > 5:
                    st.session_state.prediction_history = st.session_state.prediction_history[:5]
    
    # Display results
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        for i, result in enumerate(st.session_state.prediction_history):
            with st.container():
                st.markdown(f"**Result #{i+1}:**")
                
                # Display text
                st.markdown(f"*Input:* {result['text']}")
                
                # Display prediction
                if result['is_toxic']:
                    st.error(f"🚨 **Toxic** (Confidence: {result['confidence']*100:.1f}%)")
                else:
                    st.success(f"✅ **Non-toxic** (Confidence: {result['confidence']*100:.1f}%)")
                
                # Show cleaned text if different
                if result['cleaned_text'] != result['text'].lower():
                    with st.expander("Show processed text"):
                        st.code(result['cleaned_text'])
                
                if i < len(st.session_state.prediction_history) - 1:
                    st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Powered by:** [PhoBERT](https://huggingface.co/vinai/phobert-base) | "
        "**Model by:** [binhkhang3169](https://huggingface.co/binhkhang3169) | "
        "**Built with:** [Streamlit](https://streamlit.io)"
    )

if __name__ == "__main__":
    main()