import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw
import numpy as np
# ✅ Safer import that avoids unnecessary modules like FastSAM
from ultralytics import YOLO

import base64

import streamlit as st
import base64

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DermaVision",
    page_icon="🔬",
    layout="wide"
)

if 'user_name' not in st.session_state or st.session_state.user_name == '':
    st.markdown("""
    <style>
    .stApp {
        background-color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    .header-box {
        background-color: #0056b3;
        padding: 60px 30px;
        border-radius: 0 0 40px 40px;
        color: white;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 40px;
    }

    .header-box h1 {
        font-size: 46px;
        font-weight: 800;
        margin: 0;
    }

    .input-wrapper {
        padding: 40px 60px;
    }

    .input-label {
        font-size: 20px;
        font-weight: 600;
        color: #333;
        margin-bottom: 10px;
    }

    .stTextInput {
        width: 350px !important;
    }

    .stTextInput input {
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
        border: 2px solid #ccc;
        background-color: #f9f9f9;
        color: #000;
    }

    .stButton > button {
        background-color: #0056b3;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 8px;
        margin-top: 16px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #004494;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='header-box'><h1> Welcome to DermaVision!</h1></div>", unsafe_allow_html=True)

    # Input area aligned left
    st.markdown("<div class='input-wrapper'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>Enter your name to begin:</div>", unsafe_allow_html=True)
    name_input = st.text_input(label="", key="name_input")
    if st.button("Start"):
        if name_input.strip():
            st.session_state.user_name = name_input.strip()
            st.rerun()
        else:
            st.warning("Please enter a valid name.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.stop()



    if st.button("Start"):
        if name_input.strip():
            st.session_state.user_name = name_input.strip()
            st.rerun()
        else:
            st.warning("Please enter a valid name.")
    st.stop()


st.title(f"Hello {st.session_state.user_name}, welcome to the HAM10000 Classifier!")

# Class labels
class_names_resnet = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_names_yolo = [ "AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

# Load model
@st.cache_resource
def load_model(model_choice):
    num_classes = len(class_names_resnet)

    if model_choice == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load('final_resnet50_model.pt', map_location=torch.device('cpu')))
        model.eval()
        return model

    elif model_choice == "DenseNet121":
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        model.load_state_dict(torch.load('final_densenet121_model.pt', map_location=torch.device('cpu')))
        model.eval()
        return model

    elif model_choice == "YOLOv8":
        return YOLO('yolo_best.pt')

# Transform for ResNet and DenseNet
transform_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# UI
st.title("Skin Lesion Classifier (HAM10000)")
model_choice = st.selectbox("Choose a model:", ["ResNet50", "DenseNet121", "YOLOv8"])
model = load_model(model_choice)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            if model_choice in ["ResNet50", "DenseNet121"]:
                input_tensor = transform_tensor(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    prediction = torch.nn.functional.softmax(output[0], dim=0)
                    predicted_class = class_names_resnet[torch.argmax(prediction)]
                    confidence = torch.max(prediction).item()
                st.success(f"Predicted: **{predicted_class}** ({confidence:.2%} confidence)")

            elif model_choice == "YOLOv8":
                temp_image_path = "temp_uploaded_image.jpg"
                image.save(temp_image_path)

                results = model.predict(source=temp_image_path, save=False, stream=False)
                boxes = results[0].boxes

                if boxes is not None and len(boxes) > 0:
                    image_with_boxes = image.copy()
                    draw = ImageDraw.Draw(image_with_boxes)

                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = f"{class_names_yolo[class_id]} ({confidence:.2%})"
                        st.write(f"Detected class index: {class_id}, confidence: {confidence:.2%}")

                        draw.rectangle(xyxy, outline="red", width=3)
                        draw.text((xyxy[0], xyxy[1] - 10), label, fill="red")

                    st.image(image_with_boxes, caption="YOLOv8 Detected Lesions", use_container_width=True)
                else:
                    st.warning("No lesion detected in the image.")
