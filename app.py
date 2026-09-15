import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_eff
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Weather Multi-Model AI", page_icon="🌤️", layout="wide")

@st.cache_resource
def load_all_models():
    models = {
        "InceptionV3": load_model('weather_inceptionv3_best.h5'),
        "ResNet50V2": load_model('weather_resnet50v2_best.h5'),
        "EfficientNetV2S": load_model('weather_efficientnetv2s_best.h5')
    }
    return models

with st.spinner('Đang khởi động hệ thống và tải mô hình...'):
    models = load_all_models()

class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

st.markdown("<h2 style='text-align: center;'>🌤️ Hệ Thống Nhận Diện Thời Tiết Đa Mô Hình</h2>", unsafe_allow_html=True)
st.write("")

left_col, right_col = st.columns([1, 1.5], gap="medium")

with left_col:
    st.markdown("### 📤 Tải Ảnh Đầu Vào")
    uploaded_file = st.file_uploader("Chọn một bức ảnh thời tiết...", type=["jpg", "jpeg", "png"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Thao Tác")
    analyze_btn = st.button("🔄 Phân tích qua 3 Model", use_container_width=True)
    clear_btn = st.button("🗑️ Xoá / Làm mới", use_container_width=True)
    
    if clear_btn:
        st.session_state.prediction_results = None
        st.rerun()

with right_col:
    st.markdown("### 📊 Kết Quả Phân Tích")
    
    if uploaded_file is not None:
        img_display = Image.open(uploaded_file)
        st.image(img_display, caption='Ảnh bạn đã tải lên', use_container_width=True)
        
        if analyze_btn:
            with st.spinner('Đang chạy dự đoán trên cả 3 mô hình...'):
                progress_bar = st.progress(0, text="Đang khởi tạo...")
                
                progress_bar.progress(30, text="Đang dự đoán với InceptionV3...")
                img_inc = img_display.resize((299, 299))
                arr_inc = preprocess_inception(np.expand_dims(image.img_to_array(img_inc), axis=0))
                pred_inc = models["InceptionV3"].predict(arr_inc)
                
                progress_bar.progress(60, text="Đang dự đoán với ResNet50V2...")
                img_res = img_display.resize((224, 224))
                arr_res = preprocess_resnet(np.expand_dims(image.img_to_array(img_res), axis=0))
                pred_res = models["ResNet50V2"].predict(arr_res)
                
                progress_bar.progress(90, text="Đang dự đoán với EfficientNetV2S...")
                img_eff = img_display.resize((300, 300))
                arr_eff = preprocess_eff(np.expand_dims(image.img_to_array(img_eff), axis=0))
                pred_eff = models["EfficientNetV2S"].predict(arr_eff)
                
                progress_bar.progress(100, text="Hoàn tất phân tích!")
                time.sleep(0.3)
                progress_bar.empty()
                
                st.session_state.prediction_results = {
                    "InceptionV3": (class_names[np.argmax(pred_inc[0])], float(np.max(pred_inc[0])), pred_inc[0]),
                    "ResNet50V2": (class_names[np.argmax(pred_res[0])], float(np.max(pred_res[0])), pred_res[0]),
                    "EfficientNetV2S": (class_names[np.argmax(pred_eff[0])], float(np.max(pred_eff[0])), pred_eff[0])
                }

        if st.session_state.prediction_results is not None:
            res = st.session_state.prediction_results
            
            st.markdown("---")
            st.markdown("#### 🔍 Kết quả từng mô hình:")
            
            for model_name, (cls, conf, _) in res.items():
                st.success(f"**{model_name}:** Dự đoán là **{cls}** (Độ tin cậy: **{conf*100:.2f}%**)")
            
            # Tìm model tốt nhất dựa vào điểm confidence [1]
            best_model_name = max(res.items(), key=lambda x: x[1][1])[0]
            best_conf = res[best_model_name][1]
            
            st.info(f"🏆 **Model Tốt Nhất:** **{best_model_name}** với độ tự tin cao nhất (**{best_conf*100:.2f}%**)")
            
            # Expandable section xem chi tiết xác suất
            with st.expander("📈 Xem chi tiết phân phối xác suất 4 lớp (Model tốt nhất)"):
                best_probs = res[best_model_name][2]
                for idx, name in enumerate(class_names):
                    prob_val = float(best_probs[idx])
                    st.progress(prob_val, text=f"{name}: {prob_val*100:.2f}%")
    else:
        st.info("👈 Vui lòng tải lên một bức ảnh ở cột bên trái để bắt đầu.")
