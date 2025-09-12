import numpy as np
import cv2

def dominant_color_swatch(pil_img, k=3):
    img = np.array(pil_img.resize((256,256)))
    Z = img.reshape(-1,3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    counts = np.bincount(labels.flatten())
    dom = centers[np.argmax(counts)]
    return tuple(int(c) for c in dom)  # (R,G,B)

def rgb2hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)
