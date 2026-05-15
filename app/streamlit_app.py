import streamlit as st


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
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    st.info(
        "Model inference will be available after training the CRNN-BiLSTM-CTC model."
    )
else:
    st.warning("Please upload an image to test the OCR system.")