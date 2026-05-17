from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference.predict import predict_text


MODEL_PATH = Path("outputs/checkpoints/best_crnn.pth")
CHARSET_PATH = Path("outputs/charset.json")


st.set_page_config(
    page_title="Handwritten Text Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("Handwritten Text Recognition")
st.write(
    "Upload a handwritten text line image and the OCR model will predict the text."
)

uploaded_file = st.file_uploader(
    "Upload handwritten text image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if MODEL_PATH.exists() and CHARSET_PATH.exists():
        with st.spinner("Recognizing handwritten text..."):
            prediction = predict_text(
                image=image,
                model_path=MODEL_PATH,
                charset_path=CHARSET_PATH,
                device="cpu",
            )

        st.subheader("Predicted Text")
        st.success(prediction)
    else:
        st.info(
            "Model checkpoint is not available yet. "
            "Train the CRNN-BiLSTM-CTC model first."
        )
else:
    st.warning("Please upload an image to test the OCR system.")