import os
import numpy as np
import onnxruntime
from albumentations import Resize, Normalize, Compose
from torch import tensor, sigmoid


MODEL_PATH_SCE = os.path.join(os.getcwd(), "src/pan-resnest26d-sce.onnx")
MODEL_PATH_LDL = os.path.join(os.getcwd(), "src/pan-resnest50d-ldl.onnx")
MODEL_PATH = {
    "ldl": MODEL_PATH_LDL, 
    "sce": MODEL_PATH_SCE
    }

THRESHOLD = 0.7

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

    session = onnxruntime.InferenceSession(MODEL_PATH[model_name])

    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    mask_, logits_ = session.run(output_names, {input_name: img2input})

    if model_name == "sce":
        mask, probs = _output_preprocessing(mask_, logits_)
    else:
        probs = logits_
        mask, _ = _output_preprocessing(mask_, logits_)

    return mask, probs

