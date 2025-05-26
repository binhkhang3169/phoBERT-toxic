import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
import re
import os
import requests
# from tqdm import tqdm # Có thể không cần thiết nếu dùng st.progress_bar

# --- Cấu hình Model và Các hằng số ---
# Tên model gốc trên Hugging Face (dùng để tham chiếu kiến trúc nếu cần)
ORIGINAL_MODEL_NAME = "vinai/phobert-base"
# Đường dẫn cục bộ đến tokenizer và config đã tải về (trong repo GitHub)
LOCAL_CONFIG_TOKENIZER_PATH = "./phobert_local_files"

# URL trực tiếp để tải file .pt từ Hugging Face (sử dụng link "raw")
MODEL_URL = "https://huggingface.co/binhkhang3169/phoBERTtoxic/raw/main/model_best_valacc.pt"
MODEL_FILENAME = "model_best_valacc.pt" # Tên file model sẽ lưu cục bộ

MAX_LEN = 256 # Phải khớp với MAX_LEN khi training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Các hàm Helper ---
def clean_text(text):
    # 1. Loại bỏ các ký tự không mong muốn (giữ lại chữ cái, số và tiếng Việt có dấu)
    text = re.sub(r"[^\w\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỳỵỷỹýÝỲỴỶỸ]", ' ', text)
    
    # 2. Chuyển về chữ thường trước khi loại bỏ đại từ để bắt tất cả các trường hợp (ví dụ: "Bạn", "bạn")
    text = text.lower()

    # 3. Danh sách các đại từ cần loại bỏ (Tùy chọn - có thể bỏ qua nếu không muốn)
    pronouns_to_remove = [
        "bạn", "tôi", "cậu", "tớ", "mình",
    ]
    pattern = r'\b(' + '|'.join(re.escape(pronoun) for pronoun in pronouns_to_remove) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 4. Chuẩn hóa khoảng trắng (loại bỏ khoảng trắng thừa)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Định nghĩa Model ---
class BERTClassifier(torch.nn.Module):
    def __init__(self, num_labels=2, config_path=LOCAL_CONFIG_TOKENIZER_PATH, architecture_name=ORIGINAL_MODEL_NAME):
        super(BERTClassifier, self).__init__()
        try:
            # Ưu tiên tải config từ đường dẫn cục bộ
            bert_classifier_config = RobertaConfig.from_pretrained(
                config_path, # Tải config từ file cục bộ
                num_labels=num_labels,
                output_hidden_states=False,
            )
            st.write(f"INFO: Loaded RobertaConfig from local path: {config_path}")
        except Exception as e_local:
            st.warning(f"WARN: Could not load RobertaConfig from local path '{config_path}': {e_local}. Falling back to '{architecture_name}' from Hugging Face Hub.")
            # Nếu không có file config cục bộ, thử tải từ Hugging Face Hub (cần thiết cho lần chạy đầu hoặc nếu file bị thiếu)
            bert_classifier_config = RobertaConfig.from_pretrained(
                architecture_name,
                num_labels=num_labels,
                output_hidden_states=False,
            )
            st.write(f"INFO: Loaded RobertaConfig from Hugging Face Hub: {architecture_name}")

        # Tải kiến trúc model (có thể vẫn dùng tên model gốc để lấy kiến trúc base)
        # Trọng số sẽ được load từ checkpoint sau
        self.bert_classifier = RobertaForSequenceClassification.from_pretrained(
            architecture_name,
            config=bert_classifier_config
        )
        st.write(f"INFO: Initialized RobertaForSequenceClassification with architecture from '{architecture_name}'.")


    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert_classifier(
            input_ids=input_ids,
            token_type_ids=None, # RoBERTa không sử dụng token_type_ids
            attention_mask=attention_mask,
            labels=labels
        )
        return output

# --- Hàm tải file từ URL với thanh tiến trình Streamlit ---
def download_file_from_url(url, destination_path, chunk_size=8192):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Kiểm tra lỗi HTTP
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.info(f"Downloading {os.path.basename(destination_path)}...")

        downloaded_size = 0
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = min(int((downloaded_size / total_size) * 100), 100)
                        progress_bar.progress(progress)
        
        status_text.success(f"Download complete: {os.path.basename(destination_path)}")
        progress_bar.empty() # Xóa thanh progress sau khi hoàn tất
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        if os.path.exists(destination_path):
            os.remove(destination_path) # Xóa file nếu tải lỗi
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        if os.path.exists(destination_path):
            os.remove(destination_path)
        return False

# --- Hàm tải Model và Tokenizer (Cached) ---
@st.cache_resource # Sử dụng cache_resource cho model và tokenizer
def load_model_and_tokenizer():
    st.write("Attempting to load model and tokenizer...")
    local_model_path = MODEL_FILENAME

    # 1. Tải model weights nếu chưa có
    if not os.path.exists(local_model_path):
        st.info(f"Model file '{local_model_path}' not found locally. Attempting to download from {MODEL_URL}...")
        if not MODEL_URL:
            st.error("MODEL_URL is not configured.")
            st.stop()
        if not download_file_from_url(MODEL_URL, local_model_path):
            st.error("Failed to download model weights. Please check the URL and network connection.")
            st.stop()
    else:
        st.write(f"Found local model weights: {local_model_path}")

    # 2. Tải tokenizer
    try:
        # Ưu tiên tải tokenizer từ đường dẫn cục bộ
        phobert_tokenizer = AutoTokenizer.from_pretrained(LOCAL_CONFIG_TOKENIZER_PATH, use_fast=False)
        st.write(f"INFO: Loaded PhoBERT Tokenizer from local path: {LOCAL_CONFIG_TOKENIZER_PATH}")
    except Exception as e_local_tokenizer:
        st.warning(f"WARN: Could not load PhoBERT Tokenizer from local path '{LOCAL_CONFIG_TOKENIZER_PATH}': {e_local_tokenizer}. Falling back to '{ORIGINAL_MODEL_NAME}' from Hugging Face Hub.")
        # Nếu không có file tokenizer cục bộ, thử tải từ Hugging Face Hub
        try:
            phobert_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME, use_fast=False)
            st.write(f"INFO: Loaded PhoBERT Tokenizer from Hugging Face Hub: {ORIGINAL_MODEL_NAME}")
        except Exception as e_hub_tokenizer:
            st.error(f"FATAL: Could not load PhoBERT Tokenizer from local path or Hugging Face Hub. Error (Hub): {e_hub_tokenizer}")
            st.stop()
            return None, None


    # 3. Khởi tạo cấu trúc model và tải trọng số
    try:
        phobert_model_structure = BERTClassifier(
            num_labels=2, 
            config_path=LOCAL_CONFIG_TOKENIZER_PATH, # Truyền đường dẫn config cục bộ
            architecture_name=ORIGINAL_MODEL_NAME
        )
        
        st.write(f"Loading checkpoint from: {local_model_path} to device: {DEVICE}")
        checkpoint = torch.load(local_model_path, map_location=DEVICE)
        
        # Kiểm tra định dạng checkpoint
        if 'model_state_dict' in checkpoint:
            phobert_model_structure.load_state_dict(checkpoint['model_state_dict'])
            st.write("Loaded model_state_dict from checkpoint.")
        elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
            # Giả định checkpoint là một state_dict trực tiếp
            phobert_model_structure.load_state_dict(checkpoint)
            st.write("Loaded state_dict directly from checkpoint.")
        else:
            st.error("Model checkpoint format not recognized. Expected a dict or a dict with 'model_state_dict' key.")
            st.stop()
            return None, None
            
        phobert_model = phobert_model_structure.to(DEVICE)
        phobert_model.eval() # Chuyển model sang chế độ inference
        st.write("Model loaded and set to evaluation mode successfully.")
        return phobert_model, phobert_tokenizer
        
    except Exception as e:
        st.error(f"Error loading model structure or weights: {e}")
        st.exception(e) # In ra traceback đầy đủ cho debug
        st.stop()
        return None, None

# --- Tải model và tokenizer khi ứng dụng khởi chạy ---
# Các thông báo st.write() bên trong load_model_and_tokenizer sẽ giúp theo dõi tiến trình
phobert_model, phobert_tokenizer = load_model_and_tokenizer()

# --- Giao diện người dùng Streamlit ---
st.set_page_config(page_title="Toxic Comment Detector", layout="centered")
st.title("Toxic Comment Detector")
st.markdown("Enter Vietnamese text below to check if it's toxic or non-toxic.")
st.markdown(f"Running on device: `{DEVICE}`")
if phobert_model is None or phobert_tokenizer is None:
    st.error("Model or Tokenizer not loaded. Application cannot proceed.")
    st.stop()

# Khởi tạo session state nếu chưa có
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = None
if 'submitted_text' not in st.session_state: # Lưu lại text đã submit để hiển thị lại
    st.session_state.submitted_text = ""


user_input = st.text_area("Enter your comment here:", 
                            value=st.session_state.text_input, 
                            height=150, 
                            placeholder="Nhập bình luận tiếng Việt của bạn ở đây...",
                            key="main_text_input_area" # Thêm key để tránh lỗi nếu có nhiều text_area
                            )

if st.button("Check Toxicity", use_container_width=True, type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
        st.session_state.prediction_result = None
        st.session_state.confidence_score = None
        st.session_state.submitted_text = ""
    else:
        st.session_state.text_input = user_input # Lưu lại nội dung người dùng nhập
        st.session_state.submitted_text = user_input # Lưu text để hiển thị lại sau khi xử lý

        with st.spinner("Analyzing..."):
            cleaned_text = clean_text(user_input)
            
            # Tokenize sử dụng tokenizer của Hugging Face (tự động padding và tạo attention mask)
            inputs = phobert_tokenizer(
                cleaned_text,
                add_special_tokens=True,       # Thêm [CLS] và [SEP] (hoặc <s>, </s> cho RoBERTa)
                max_length=MAX_LEN,            # Giới hạn độ dài tối đa
                padding='max_length',          # Pad đến max_length
                truncation=True,               # Cắt nếu dài hơn max_length
                return_attention_mask=True,    # Trả về attention_mask
                return_tensors='pt'            # Trả về PyTorch tensors
            )

            input_ids_tensor = inputs['input_ids'].to(DEVICE)
            attention_mask_tensor = inputs['attention_mask'].to(DEVICE)

            with torch.no_grad(): # Không tính gradient khi inference
                outputs = phobert_model(input_ids_tensor, attention_mask=attention_mask_tensor)
            
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits # Lấy logits từ output của model
            probabilities = torch.nn.functional.softmax(logits, dim=1) # Tính xác suất
            confidence, predicted_class_idx = torch.max(probabilities, dim=1) # Lấy lớp có xác suất cao nhất

            # Giả sử: 0 là Non-toxic, 1 là Toxic (cần khớp với lúc training)
            predicted_label = "Toxic" if predicted_class_idx.item() == 1 else "Non-toxic" 
            
            st.session_state.prediction_result = predicted_label
            st.session_state.confidence_score = confidence.item()

# Hiển thị kết quả nếu có
if st.session_state.prediction_result and st.session_state.submitted_text:
    st.markdown("---")
    st.subheader("Result:")
    st.markdown(f"**Your input:**")
    # Dùng st.text hoặc st.markdown trong block quote để hiển thị an toàn hơn
    st.markdown(f"> _{st.session_state.submitted_text}_")


    if st.session_state.prediction_result == "Toxic":
        st.error(f"**Prediction:** {st.session_state.prediction_result}")
    else:
        st.success(f"**Prediction:** {st.session_state.prediction_result}")
    
    st.info(f"**Confidence:** {st.session_state.confidence_score*100:.2f}%")

st.markdown("---")
st.markdown("Powered by [Streamlit](https://streamlit.io) | Model based on [PhoBERT](https://huggingface.co/vinai/phobert-base) | Weights by [binhkhang3169](https://huggingface.co/binhkhang3169)")