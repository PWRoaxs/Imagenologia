# -*- coding: utf-8 -*-

import io
import os
import numpy as np
from PIL import Image
import torch
import streamlit as st

from model import UNet

DEVICE = "cpu"
MASK_COLOR = (255, 107, 129)
DEFAULT_WEIGHTS = "model_demo.pt"


# --------------------------------------------------------------------------- #
#  Carga del modelo (PyTorch)                                                  #
# --------------------------------------------------------------------------- #
def _extract_state_dict(ckpt):
    """Acepta state_dict directo o dentro de un dict ('state_dict'/'model'/'net')."""
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], ckpt
        # ¿el propio dict ya es un state_dict? (valores tensores)
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt, {}
    return ckpt, {}


def _clean_keys(sd):
    """Quita prefijos comunes como 'module.' (de DataParallel)."""
    out = {}
    for k, v in sd.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


@st.cache_resource(show_spinner=False)
def load_model(weights_bytes: bytes | None, in_ch: int, base: int):
    """Construye la U-Net y carga pesos.

    weights_bytes=None → usa el archivo DEFAULT_WEIGHTS si existe.
    Devuelve (modelo, info_str).
    """
    net = UNet(in_ch=in_ch, out_ch=1, base=base).to(DEVICE)

    raw = None
    source = None
    if weights_bytes is not None:
        raw, source = weights_bytes, "pesos subidos"
    elif os.path.exists(DEFAULT_WEIGHTS):
        with open(DEFAULT_WEIGHTS, "rb") as f:
            raw, source = f.read(), f"pesos incluidos ({DEFAULT_WEIGHTS})"

    if raw is None:
        net.eval()
        return net, "⚠️ Sin pesos: la red está SIN entrenar y la salida no tiene sentido."

    ckpt = torch.load(io.BytesIO(raw), map_location=DEVICE, weights_only=False)
    sd, _ = _extract_state_dict(ckpt)
    sd = _clean_keys(sd)
    result = net.load_state_dict(sd, strict=False)
    net.eval()

    info = f"Modelo cargado desde {source}."
    if result.missing_keys or result.unexpected_keys:
        info += (f" Coincidencia parcial: {len(result.missing_keys)} claves "
                 f"faltantes, {len(result.unexpected_keys)} inesperadas "
                 "(¿la arquitectura no coincide con tu checkpoint?).")
    return net, info



def preprocess(image: Image.Image, size: int, in_ch: int):
    size = (size // 8) * 8  # múltiplo de 8 para la U-Net
    mode = "L" if in_ch == 1 else "RGB"
    img = image.convert(mode).resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    if in_ch == 1:
        arr = arr[None, None, :, :]               # [1,1,H,W]
    else:
        arr = arr.transpose(2, 0, 1)[None, ...]   # [1,3,H,W]
    return torch.from_numpy(arr), img.convert("L")


@torch.no_grad()
def infer(net, x, threshold: float):
    prob = torch.sigmoid(net(x.to(DEVICE)))[0, 0].cpu().numpy()
    mask = prob > threshold
    return prob, mask


def overlay(gray: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(gray.convert("RGB")).astype(np.float32)
    if mask.shape != base.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                          .resize((base.shape[1], base.shape[0]))) > 127
    base[mask] = base[mask] * (1 - alpha) + np.array(MASK_COLOR) * alpha
    return Image.fromarray(base.astype(np.uint8))


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask[..., None] * np.array(MASK_COLOR)).astype(np.uint8))



#  Interfaz                                                                    #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="MedSegDiff · PyTorch", page_icon="🧠", layout="wide")
st.title(" MedSegDiff · Segmentación con PyTorch")
st.caption("U-Net de PyTorch · herramienta educativa para estudiantes.")
st.error("**Demostración educativa.** No es un dispositivo médico ni hace "
         "diagnóstico. Los pesos incluidos se entrenaron con datos sintéticos; "
         "reemplázalos por un modelo real para resultados clínicamente útiles.")

with st.sidebar:
    st.header("Parámetros")
    in_ch = 1 if st.radio("Canales de entrada", ["1 (gris)", "3 (RGB)"]) == "1 (gris)" else 3
    base = st.select_slider("Canales base de la U-Net", [8, 16, 32, 64], value=16,
                            help="Debe coincidir con el modelo de tus pesos.")
    size = st.select_slider("Tamaño de inferencia", [128, 160, 192, 256], value=192)
    threshold = st.slider("Umbral de la máscara", 0.05, 0.95, 0.5, 0.05)
    st.markdown("---")
    st.caption("Pesos del modelo (.pt / .pth)")
    wfile = st.file_uploader("Subir pesos propios (opcional)", type=["pt", "pth"])
    st.caption(f"Versión de PyTorch: {torch.__version__}")

weights_bytes = wfile.read() if wfile is not None else None

try:
    model, info = load_model(weights_bytes, in_ch, base)
    (st.success if "Sin pesos" not in info else st.warning)(info)
except Exception as e:
    st.exception(e)
    st.stop()

st.subheader("1 · Sube una imagen")
upload = st.file_uploader("Imagen médica (PNG / JPG)", type=["png", "jpg", "jpeg"])

if upload is None:
    st.info("Esperando imagen…")
    st.stop()

image = Image.open(io.BytesIO(upload.read()))
image.load()

if not st.button("▶ Ejecutar segmentación", type="primary", use_container_width=True):
    st.image(image, caption="Vista previa", width=360)
    st.stop()

with st.spinner("Inferencia con PyTorch…"):
    x, gray = preprocess(image, size, in_ch)
    prob, mask = infer(model, x, threshold)

st.subheader("2 · Resultado")
c1, c2, c3, c4 = st.columns(4)
c1.image(gray, caption="Entrada", use_container_width=True)
c2.image((prob * 255).astype(np.uint8), caption="Probabilidad", use_container_width=True)
c3.image(mask_to_image(mask), caption="Máscara", use_container_width=True)
c4.image(overlay(gray, mask), caption="Overlay", use_container_width=True)

st.metric("Región segmentada", f"{round(100 * mask.mean(), 2)} %")
st.caption("La máscara proviene de una U-Net; con los pesos de demostración "
           "resalta regiones brillantes prominentes, no detecta enfermedades.")
