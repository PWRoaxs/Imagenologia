# -*- coding: utf-8 -*-
"""
Entrena la U-Net de DEMOSTRACIÓN con datos sintéticos y guarda model_demo.pt.

Los datos son artificiales (una "anatomía" gris + una "lesión" brillante
irregular + ruido). El objetivo NO es precisión clínica, sino tener pesos
reales para que la app de Streamlit haga inferencia con PyTorch y segmente
regiones brillantes prominentes. Reemplaza estos pesos por los tuyos si tienes
un modelo entrenado de verdad.

Uso:  python train_demo.py
"""
import math
import numpy as np
import torch
import torch.nn as nn

from model import UNet

SIZE = 128
torch.manual_seed(0)
np.random.seed(0)


def make_batch(n, size=SIZE):
    """Genera (imagen[n,1,H,W], máscara[n,1,H,W]) sintéticas en [0,1]."""
    imgs = np.zeros((n, size, size), np.float32)
    masks = np.zeros((n, size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        img = np.random.rand(size, size).astype(np.float32) * 0.12  # fondo
        # anatomía: elipse gris centrada
        acx, acy = size / 2 + np.random.uniform(-8, 8), size / 2 + np.random.uniform(-8, 8)
        arx, ary = size * 0.36, size * 0.30
        anat = (((xx - acx) / arx) ** 2 + ((yy - acy) / ary) ** 2) <= 1
        img[anat] += 0.45 + np.random.uniform(-0.05, 0.05)
        # lesión: blob brillante irregular dentro de la anatomía
        lcx = acx + np.random.uniform(-arx * 0.4, arx * 0.4)
        lcy = acy + np.random.uniform(-ary * 0.4, ary * 0.4)
        base_r = np.random.uniform(size * 0.07, size * 0.16)
        ang = np.arctan2(yy - lcy, xx - lcx)
        wob = 1 + 0.35 * np.sin(ang * np.random.randint(3, 6) + np.random.uniform(0, 6))
        dist = np.sqrt((xx - lcx) ** 2 + (yy - lcy) ** 2)
        les = dist <= (base_r * wob)
        img[les] = 0.9 + np.random.uniform(-0.05, 0.05)
        img += np.random.randn(size, size).astype(np.float32) * 0.04
        img = np.clip(img, 0, 1)
        imgs[i] = img
        masks[i] = les.astype(np.float32)
    x = torch.from_numpy(imgs[:, None, :, :])
    y = torch.from_numpy(masks[:, None, :, :])
    return x, y


def main():
    device = "cpu"
    net = UNet(in_ch=1, out_ch=1, base=16).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    steps, bs = 400, 12
    net.train()
    for step in range(1, steps + 1):
        x, y = make_batch(bs)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = net(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter = (pred * y).sum()
                dice = (2 * inter / (pred.sum() + y.sum() + 1e-6)).item()
            print(f"step {step:4d}/{steps}  loss {loss.item():.4f}  dice {dice:.3f}")

    net.eval()
    torch.save({"state_dict": net.state_dict(),
                "in_ch": 1, "base": 16, "note": "synthetic-demo"},
               "model_demo.pt")
    print("Guardado model_demo.pt")


if __name__ == "__main__":
    main()
