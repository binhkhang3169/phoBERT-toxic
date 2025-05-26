import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences # Vẫn dùng từ code gốc
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

# --- Các hàm Helper (Giữ nguyên từ code gốc) ---
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
        # Bạn có thể thêm các biến thể hoặc từ khác nếu muốn
        # Ví dụ: "tao", "mày", "chúng ta", "chúng tôi", "chúng nó"
    ]
    
    # Tạo pattern regex để khớp chính xác các từ này (dùng \b để khớp ranh giới từ)
    # Ví dụ: \bbạn\b sẽ khớp "bạn" nhưng không khớp "bạnhiền"
    # Dùng re.escape để đảm bảo các ký tự đặc biệt trong từ (nếu có) được xử lý đúng
    pattern = r'\b(' + '|'.join(re.escape(pronoun) for pronoun in pronouns_to_remove) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE) # flags=re.IGNORECASE có thể không cần nếu đã lower() ở trên

    # 4. Chuẩn hóa khoảng trắng (loại bỏ khoảng trắng thừa)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text # Không cần .lower() nữa vì đã làm ở bước 2

# --- Định nghĩa Model (Giữ nguyên từ code gốc) ---
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
        
        # Sử dụng st.empty() để tạo một placeholder cho progress bar
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
        progress_bar_placeholder.empty() # Xóa progress bar sau khi hoàn tất
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
@st.cache_resource  # Cache resource này để không phải tải lại model mỗi lần
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
        # Tải Tokenizer
        phobert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FOR_TOKENIZER_AND_ARCHITECTURE, use_fast=False)

        # Khởi tạo cấu trúc model
        phobert_model_structure = BERTClassifier(num_labels=2, model_name=MODEL_NAME_FOR_TOKENIZER_AND_ARCHITECTURE)

        # Tải trọng số đã train
        checkpoint = torch.load(local_model_path, map_location=DEVICE)
        
        # Kiểm tra định dạng checkpoint
        if 'model_state_dict' in checkpoint:
            phobert_model_structure.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()): # Nếu là state_dict trực tiếp
            phobert_model_structure.load_state_dict(checkpoint)
        else:
            st.error("Model checkpoint format not recognized. Expected a state_dict or a dict with 'model_state_dict'.")
            st.stop()
            
        phobert_model = phobert_model_structure.to(DEVICE)
        phobert_model.eval() # Chuyển sang chế độ đánh giá
        
        # st.success(f"Model and tokenizer loaded successfully on {DEVICE}!") # Thông báo khi tải xong (có thể ẩn đi)
        return phobert_model, phobert_tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        # Cân nhắc xóa file nếu tải về mà load lỗi để lần sau tải lại
        # if os.path.exists(local_model_path) and MODEL_URL:
        #     os.remove(local_model_path)
        #     st.warning(f"Removed potentially corrupted local model file '{local_model_path}'. Please try reloading.")
        st.stop()
        return None, None

# --- Tải model và tokenizer khi ứng dụng khởi chạy ---
# Thông báo cho người dùng biết quá trình tải model có thể mất thời gian
with st.spinner(f"Loading model from {MODEL_NAME_FOR_TOKENIZER_AND_ARCHITECTURE} and weights... This may take a moment."):
    phobert_model, phobert_tokenizer = load_model_and_tokenizer()

# --- Giao diện người dùng Streamlit ---
st.set_page_config(page_title="Toxic Comment Detector", layout="centered")
st.title("💬 Toxic Comment Detector")
st.markdown("Enter Vietnamese text below to check if it's toxic or non-toxic.")

# Sử dụng session state để giữ lại input và kết quả sau mỗi lần tương tác
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = None
if 'submitted_text' not in st.session_state:
    st.session_state.submitted_text = ""


# Input text area
user_input = st.text_area("Enter your comment here:", 
                          value=st.session_state.text_input, 
                          height=150, 
                          placeholder="Nhập bình luận tiếng Việt của bạn ở đây...",
                          key="main_text_input")

if st.button("🔍 Check Toxicity", use_container_width=True, type="primary"):
    if phobert_model is None or phobert_tokenizer is None:
        st.error("Model is not loaded. Please check for errors during startup.")
    elif not user_input.strip():
        st.warning("Please enter some text to analyze.")
        st.session_state.prediction_result = None # Xóa kết quả cũ nếu input rỗng
        st.session_state.confidence_score = None
        st.session_state.submitted_text = ""
    else:
        st.session_state.text_input = user_input # Lưu input hiện tại
        st.session_state.submitted_text = user_input # Lưu input đã submit để hiển thị

        with st.spinner("Analyzing..."):
            cleaned_text = clean_text(user_input)

            encoded_text = phobert_tokenizer.encode(cleaned_text, truncation=True, max_length=MAX_LEN, add_special_tokens=True)
            ids_padded = pad_sequences([encoded_text], maxlen=MAX_LEN, dtype="long",
                                       value=phobert_tokenizer.pad_token_id if phobert_tokenizer.pad_token_id is not None else 0,
                                       truncating="post", padding="post")

            input_ids_tensor = torch.tensor(ids_padded).to(DEVICE)
            attention_mask_tensor = make_mask(ids_padded).to(DEVICE)

            with torch.no_grad():
                outputs = phobert_model(input_ids_tensor, attention_mask=attention_mask_tensor)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)

            predicted_label = "Toxic ☠️" if predicted_class_idx.item() == 1 else "Non-toxic 👍"
            
            st.session_state.prediction_result = predicted_label
            st.session_state.confidence_score = confidence.item()

# Hiển thị kết quả
if st.session_state.prediction_result and st.session_state.submitted_text:
    st.markdown("---")
    st.subheader("📝 Result:")
    
    # Hiển thị lại text đã nhập
    st.markdown(f"**Your input:**")
    st.markdown(f"> _{st.session_state.submitted_text}_")

    # Hiển thị kết quả
    if st.session_state.prediction_result == "Toxic ☠️":
        st.error(f"**Prediction:** {st.session_state.prediction_result}")
    else:
        st.success(f"**Prediction:** {st.session_state.prediction_result}")
    
    st.info(f"**Confidence:** {st.session_state.confidence_score*100:.2f}%")

st.markdown("---")
st.markdown("Powered by [Streamlit](https://streamlit.io) | Model based on [PhoBERT](https://huggingface.co/vinai/phobert-base) | Weights by [binhkhang3169](https://huggingface.co/binhkhang3169)")