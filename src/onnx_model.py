import os
import numpy as np
import onnxruntime
import dropbox
import streamlit as st
from albumentations import Resize, Normalize, Compose
from torch import tensor, sigmoid


API_TOKEN = st.secrets["API_TOKEN"]
THRESHOLD = 0.7

def _load_models():
    dbx = dropbox.Dropbox(API_TOKEN)
    _, sce_model = dbx.files_download("/pan-resnest26d-sce.onnx")
    _, ldl_model = dbx.files_download("/pan-resnest50d-ldl.onnx")

    models = {
        "ldl": ldl_model.content, 
        "sce": sce_model.content
        }

    return models


MODELS = _load_models() # cringe


def _transform_image(image: np.ndarray) -> np.ndarray:
    transform = Compose([Resize(256, 256), Normalize()])
    img2transform = transform(image=image)
    img = img2transform["image"]
    img2input = np.transpose(img, (2, 0, 1))

    return img2input[None, :]

def _output_preprocessing(mask: np.ndarray, logits: np.ndarray) -> tuple:
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

    probs = softmax(logits)

    mask = tensor(mask)[0][0]
    mask_ = (sigmoid(mask) > THRESHOLD).numpy().astype(np.uint8)

    return mask_, probs

def get_prediction(image: np.ndarray, model_name: str) -> tuple:
    img2input = _transform_image(image)

    session = onnxruntime.InferenceSession(MODELS[model_name])

    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    mask_, logits_ = session.run(output_names, {input_name: img2input})

    if model_name == "sce":
        mask, probs = _output_preprocessing(mask_, logits_)
    else:
        probs = logits_
        mask, _ = _output_preprocessing(mask_, logits_)

    return mask, probs

