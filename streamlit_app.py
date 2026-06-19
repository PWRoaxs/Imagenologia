# -*- coding: utf-8 -*-

import io
import numpy as np
from PIL import Image, ImageFilter
import streamlit as st

MASK_COLOR = (255, 107, 129)


def otsu(arr):
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total, sum_total = arr.size, np.dot(np.arange(256), hist)
    sum_b = w_b = best_var = 0.0
    thr = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        var = w_b * w_f * (sum_b / w_b - (sum_total - sum_b) / w_f) ** 2
        if var > best_var:
            best_var, thr = var, t
    return int(thr)


def segment(img, sensitivity=1.0, target_dark=False, max_side=512):
    g = img.convert("L")
    w, h = g.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        g = g.resize((int(w * s), int(h * s)))
    arr = np.asarray(g.filter(ImageFilter.GaussianBlur(2))).astype(np.uint8)
    t = int(np.clip(otsu(arr) * sensitivity, 1, 254))
    mask = (arr < t) if target_dark else (arr > t)
    m = Image.fromarray((mask * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    return g, np.asarray(m) > 127


def overlay(gray, mask, alpha=0.45):
    base = np.asarray(gray.convert("RGB")).astype(np.float32)
    base[mask] = base[mask] * (1 - alpha) + np.array(MASK_COLOR) * alpha
    return Image.fromarray(base.astype(np.uint8))


# ----------------------------- Interfaz ----------------------------------- #
st.set_page_config(page_title="MedSegDiff · Demostración", page_icon="🧠")
st.title(" MedSegDiff · Demostración de segmentación")


with st.sidebar:
    target_dark = st.radio("Región de interés",
                           ["Estructuras claras", "Estructuras oscuras"]) == "Estructuras oscuras"
    sensitivity = st.slider("Sensibilidad del umbral", 0.5, 1.5, 1.0, 0.05)

upload = st.file_uploader("Sube una imagen (PNG / JPG)", type=["png", "jpg", "jpeg"])

if upload is None:
    st.info("Esperando imagen…")
    st.stop()

image = Image.open(io.BytesIO(upload.read()))
image.load()

if st.button("▶ Ejecutar segmentación", type="primary"):
    gray, mask = segment(image, sensitivity, target_dark)
    pct = round(100 * mask.mean(), 2)
    c1, c2, c3 = st.columns(3)
    c1.image(gray, caption="Entrada", use_container_width=True)
    c2.image(Image.fromarray((mask[..., None] * np.array(MASK_COLOR)).astype(np.uint8)),
             caption="Máscara", use_container_width=True)
    c3.image(overlay(gray, mask), caption="Overlay", use_container_width=True)
    st.metric("Región segmentada", f"{pct} %")
else:
    st.image(image, caption="Vista previa", width=360)
