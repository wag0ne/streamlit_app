import numpy as np
import cv2


COLOR = [200, 0, 0]

def red_mask_over_face(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    assert image.shape[:-1] == mask.shape, \
        f"image.shape[:-1]: {image.shape[:-1]} and mask.shape {mask.shape}"
    
    idx = np.where(mask == 1)

    red_mask = np.zeros(image.shape) + 255
    red_mask[idx[0], idx[1], :] = COLOR

    img_with_mask = cv2.addWeighted(red_mask.astype(np.uint16), 0.15, image, 0.9, 0)

    return img_with_mask

def color_mask_over_face(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    assert image.shape[:-1] == mask.shape, \
        f"image.shape[:-1]: {image.shape[:-1]} and mask.shape {mask.shape}"

    img_with_mask = cv2.addWeighted(image, 0.9, image*mask[:, :, None], 0.3, 0)

    return img_with_mask


