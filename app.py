from typing import BinaryIO
import numpy as np
import cv2

import streamlit as st

from src.onnx_model import get_prediction
from src.mask2show import red_mask_over_face, color_mask_over_face
from src.probs2show import CLASSES, plot_model_confidence


def decode_image(upload_image):
    decoded = cv2.imdecode(np.frombuffer(upload_image.getvalue(), np.uint8), -1)
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return rgb

def run(uploaded_file: BinaryIO, model_name: str):
    rgb = decode_image(uploaded_file)

    mask, probs = get_prediction(image=rgb, model_name=model_name)

    img2show = rgb.astype(np.uint16) 
    mask2show = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))

    rm_over_face = red_mask_over_face(image=img2show, mask=mask2show)
    clrm_over_face = color_mask_over_face(image=img2show, mask=mask2show)

    prob_figure = plot_model_confidence(probs=probs)

    mask_preds_ = st.button("Prediction lesion area")
    if mask_preds_:
        st.text(f"Prediction severity: {CLASSES[np.argmax(probs)]}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img2show, caption="Input image", clamp=True)
        with col2:
            st.image(rm_over_face, caption="Red mask", clamp=True)
        with col3:
            st.image(clrm_over_face, caption="Color mask", clamp=True)

    confidence_ = st.button("Model confidence")
    if confidence_:
        st.plotly_chart(prob_figure)

def main():
    st.header("Demo for Detection and Classification of acne severity")

    with st.sidebar:

        m = st.selectbox("Choose one of models", 
                         ("pan-resnest-ldl", "pan-resnest-sce"))
        model_name = m.split("-")[-1]

        if model_name == "ldl":
            st.caption(
                """This model is trained with label disribution learning.
                """)
        else:
            st.caption(
                """This model is trained with smoothing crossentropy.
                """)

        st.subheader("NOTE")
        st.write(
            """In this simple demo, you can try to assess the severity of acne.
            The diagnostics takes place using two neural networks, one of which you can choose yourself.
            """)
        st.markdown(
            """
            Not recommended to upload:
            * Selfie with `makeup`.
            * `Foreign objects` in front of the face.
            * Selfies where `more than one person` is present. \
                *Estimates are made only for one person in the photo, not two, not three, only for one person.*
            * `Other body parts` ~like ass~ like arm etc.
            """)

    type_input = st.radio("Choose a convenient photo upload method", ("Upload file", "Take a photo"))
    
    if type_input == "Upload file":
        uploaded_file = st.file_uploader("Choose a photo", type=["jpeg", "png", "jpg"])
    else:
        uploaded_file = st.camera_input("Take a selfie")

    if uploaded_file is not None:
        run(uploaded_file=uploaded_file, model_name=model_name)
   
if __name__ == '__main__':
    main()
