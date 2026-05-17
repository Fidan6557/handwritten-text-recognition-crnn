from pathlib import Path

import streamlit as st
from PIL import Image

from src.data.segmentation import segment_text_lines
from src.inference.predict import predict_text
from src.inference.predict_page import predict_page_text


MODEL_PATH = Path("outputs/checkpoints/best_crnn.pth")
CHARSET_PATH = Path("outputs/charset.json")


st.set_page_config(
    page_title="Handwritten Text Recognition",
    page_icon="✍️",
    layout="centered",
)

st.title("Handwritten Text Recognition")

st.write(
    "Upload a handwritten line image or a full handwritten page. "
    "For full-page images, the system segments the page into text lines first."
)

mode = st.radio(
    "Select OCR mode",
    ["Line image OCR", "Full page OCR"],
)

uploaded_file = st.file_uploader(
    "Upload handwritten image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True,
    )

    if not MODEL_PATH.exists() or not CHARSET_PATH.exists():
        st.error(
            "Trained CRNN model is not available yet. "
            "Please train the model first to generate best_crnn.pth and charset.json."
        )

    else:
        if mode == "Line image OCR":
            with st.spinner("Recognizing handwritten line..."):
                prediction = predict_text(
                    image=image,
                    model_path=MODEL_PATH,
                    charset_path=CHARSET_PATH,
                    device="cpu",
                )

            st.subheader("Predicted Text")
            st.success(prediction)

        else:
            with st.spinner("Segmenting page into text lines..."):
                line_images = segment_text_lines(image)

            st.subheader("Detected Text Lines")
            st.write(f"Detected {len(line_images)} line(s).")

            for index, line_image in enumerate(line_images, start=1):
                st.image(
                    line_image,
                    caption=f"Line {index}",
                    use_column_width=True,
                )

            with st.spinner("Recognizing full page text..."):
                prediction = predict_page_text(
                    image=image,
                    model_path=MODEL_PATH,
                    charset_path=CHARSET_PATH,
                    device="cpu",
                )

            st.subheader("Predicted Full Text")
            st.text_area(
                "OCR Output",
                value=prediction,
                height=250,
            )

else:
    st.warning("Please upload an image to test the OCR system.")