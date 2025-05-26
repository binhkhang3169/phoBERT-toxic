import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
import re
import os
import requests # Để tải file từ URL
from tqdm import tqdm # Để hiển thị thanh tiến trình (tùy chọn)

# --- Cấu hình Model và Các hằng số ---
MODEL_NAME_FOR_TOKENIZER_AND_ARCHITECTURE = "vinai/phobert-base"
# URL trực tiếp để tải file .pt từ Hugging Face (sử dụng link "raw")
MODEL_URL = "https://huggingface.co/binhkhang3169/phoBERTtoxic/raw/main/model_best_valacc.pt"
MODEL_FILENAME = "model_best_valacc.pt" # Tên file model sẽ lưu cục bộ

MAX_LEN = 256  # Phải khớp với MAX_LEN khi training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Các hàm Helper ---
def make_mask(batch_ids):
    batch_mask = []
    for ids in batch_ids:
        mask = [int(token_id > 0) for token_id in ids]
        batch_mask.append(mask)
    return torch.tensor(batch_mask)

def clean_text(text):
    # 1. Loại bỏ các ký tự không mong muốn (giữ lại chữ cái, số và tiếng Việt có dấu)
    text = re.sub(r"[^\w\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỳỵỷỹýÝỲỴỶỸ]", ' ', text)
    
    # 2. Chuyển về chữ thường trước khi loại bỏ đại từ để bắt tất cả các trường hợp (ví dụ: "Bạn", "bạn")
    text = text.lower()

    # 3. Danh sách các đại từ cần loại bỏ
    pronouns_to_remove = [
        "bạn", "tôi", "cậu", "tớ", "mình",
    ]
    
    pattern = r'\b(' + '|'.join(re.escape(pronoun) for pronoun in pronouns_to_remove) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 4. Chuẩn hóa khoảng trắng (loại bỏ khoảng trắng thừa)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def custom_pad_sequences(sequences, maxlen, dtype='long', padding='post', truncating='post', value=0):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            elif truncating == 'pre':
                seq = seq[-maxlen:]
            else:
                raise ValueError(f"Truncating type {truncating} not understood")
        
        if len(seq) < maxlen:
            if padding == 'post':
                padded_seq = seq + [value] * (maxlen - len(seq))
            elif padding == 'pre':
                padded_seq = [value] * (maxlen - len(seq)) + seq
            else:
                raise ValueError(f"Padding type {padding} not understood")
        else: 
            padded_seq = seq
            
        padded_sequences.append(padded_seq)
    return padded_sequences

# --- Định nghĩa Model ---
class BERTClassifier(torch.nn.Module):
    def __init__(self, num_labels=2, model_name=MODEL_NAME_FOR_TOKENIZER_AND_ARCHITECTURE):
        super(BERTClassifier, self).__init__()
        bert_classifier_config = RobertaConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=False,
        )
        self.bert_classifier = RobertaForSequenceClassification.from_pretrained(
            model_name,
            config=bert_classifier_config
        )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert_classifier(
            input_ids=input_ids,
            token_type_ids=None,
            attention_mask=attention_mask,
            labels=labels
        )
        return output

# --- Hàm tải file từ URL với thanh tiến trình ---
def download_file_from_url(url, destination_path, chunk_size=8192):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar_placeholder = st.empty()
        status_text_placeholder = st.empty()

        downloaded_size = 0
        status_text_placeholder.info(f"Downloading {os.path.basename(destination_path)}...")

        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = min(int((downloaded_size / total_size) * 100), 100)
                        progress_bar_placeholder.progress(progress)
        
        status_text_placeholder.success(f"Download complete: {os.path.basename(destination_path)}")
        progress_bar_placeholder.empty() 
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        if os.path.exists(destination_path):
            os.remove(destination_path)
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        if os.path.exists(destination_path):
            os.remove(destination_path)
        return False

# --- Hàm tải Model và Tokenizer (Cached) ---
@st.cache_resource
def load_model_and_tokenizer():
    local_model_path = MODEL_FILENAME

    if not os.path.exists(local_model_path):
        st.info(f"Model file '{local_model_path}' not found locally. Attempting to download...")
        if not MODEL_URL:
            st.error("MODEL_URL is not configured.")
            st.stop()
        if not download_file_from_url(MODEL_URL, local_model_path):
            st.error("Failed to download model. Please check the URL and network connection.")
            st.stop()
    
    try:
        phobert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FOR_TOKENIZER_AND_ARCHITECTURE, use_fast=False)
        phobert_model_structure = BERTClassifier(num_labels=2, model_name=MODEL_NAME_FOR_TOKENIZER_AND_ARCHITECTURE)
        checkpoint = torch.load(local_model_path, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            phobert_model_structure.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
            phobert_model_structure.load_state_dict(checkpoint)
        else:
            st.error("Model checkpoint format not recognized.")
            st.stop()
            
        phobert_model = phobert_model_structure.to(DEVICE)
        phobert_model.eval()
        return phobert_model, phobert_tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.stop()
        return None, None

# --- Tải model và tokenizer khi ứng dụng khởi chạy ---
with st.spinner(f"Loading model from {MODEL_NAME_FOR_TOKENIZER_AND_ARCHITECTURE} and weights... This may take a moment."):
    phobert_model, phobert_tokenizer = load_model_and_tokenizer()

# --- Giao diện người dùng Streamlit ---
st.set_page_config(page_title="Toxic Comment Detector", layout="centered")
st.title("Toxic Comment Detector")
st.markdown("Enter Vietnamese text below to check if it's toxic or non-toxic.")

if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = None
if 'submitted_text' not in st.session_state:
    st.session_state.submitted_text = ""

user_input = st.text_area("Enter your comment here:", 
                          value=st.session_state.text_input, 
                          height=150, 
                          placeholder="Nhập bình luận tiếng Việt của bạn ở đây...",
                          key="main_text_input")

if st.button("Check Toxicity", use_container_width=True, type="primary"):
    if phobert_model is None or phobert_tokenizer is None:
        st.error("Model is not loaded. Please check for errors during startup.")
    elif not user_input.strip():
        st.warning("Please enter some text to analyze.")
        st.session_state.prediction_result = None
        st.session_state.confidence_score = None
        st.session_state.submitted_text = ""
    else:
        st.session_state.text_input = user_input
        st.session_state.submitted_text = user_input

        with st.spinner("Analyzing..."):
            cleaned_text = clean_text(user_input)
            encoded_text = phobert_tokenizer.encode(cleaned_text, truncation=True, max_length=MAX_LEN, add_special_tokens=True)
            
            pad_value = phobert_tokenizer.pad_token_id if phobert_tokenizer.pad_token_id is not None else 0
            ids_padded = custom_pad_sequences([encoded_text], maxlen=MAX_LEN,
                                               value=pad_value,
                                               padding="post", truncating="post")

            input_ids_tensor = torch.tensor(ids_padded, dtype=torch.long).to(DEVICE)
            attention_mask_tensor = make_mask(ids_padded).to(DEVICE)

            with torch.no_grad():
                outputs = phobert_model(input_ids_tensor, attention_mask=attention_mask_tensor)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)

            predicted_label = "Toxic" if predicted_class_idx.item() == 1 else "Non-toxic"
            
            st.session_state.prediction_result = predicted_label
            st.session_state.confidence_score = confidence.item()

if st.session_state.prediction_result and st.session_state.submitted_text:
    st.markdown("---")
    st.subheader("Result:")
    st.markdown(f"**Your input:**")
    st.markdown(f"> _{st.session_state.submitted_text}_")

    if st.session_state.prediction_result == "Toxic":
        st.error(f"**Prediction:** {st.session_state.prediction_result}")
    else:
        st.success(f"**Prediction:** {st.session_state.prediction_result}")
    
    st.info(f"**Confidence:** {st.session_state.confidence_score*100:.2f}%")

st.markdown("---")
st.markdown("Powered by [Streamlit](https://streamlit.io) | Model based on [PhoBERT](https://huggingface.co/vinai/phobert-base) | Weights by [binhkhang3169](https://huggingface.co/binhkhang3169)")