# MedSegDiff · Segmentación con PyTorch (Streamlit)

App web educativa: el estudiante sube una imagen médica y una **U-Net de
PyTorch** genera una máscara de segmentación. Trae pesos de demostración
entrenados con datos sintéticos; puedes reemplazarlos por un modelo real.

> ⚠️ **Demostración educativa.** No es un dispositivo médico ni realiza
> diagnóstico. Los pesos incluidos solo resaltan regiones brillantes
> prominentes.

## Archivos
```
streamlit_app.py    # interfaz + inferencia PyTorch
model.py            # definición de la U-Net (compártela con tu entrenamiento)
model_demo.pt       # pesos de demostración (entrenados con datos sintéticos)
train_demo.py       # script que generó model_demo.pt (opcional)
requirements.txt
```

## Desplegar en Streamlit Cloud
1. Sube **todos** los archivos a un repositorio de GitHub (en la raíz).
2. Entra a https://streamlit.io/cloud → **Create app** y elige tu repo.
3. *Main file path*: `streamlit_app.py` → **Deploy**.
4. La primera compilación tarda unos minutos (PyTorch es grande). Listo: ya
   tienes la URL pública para los estudiantes.

## Probar en local
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Usar tus propios pesos
- Desde la barra lateral de la app puedes **subir un archivo `.pt`/`.pth`**.
- O reemplaza `model_demo.pt` en el repo por el tuyo.
- Ajusta en la barra lateral **Canales de entrada** y **Canales base** para que
  coincidan con tu arquitectura.

### El error de tu captura ("Weights only load failed")
PyTorch ≥ 2.6 cambió `torch.load` a `weights_only=True` por defecto. Si tu
checkpoint guarda objetos que no son tensores, falla con
*“Unsupported operand …”*. La app ya lo carga con `weights_only=False`
(`streamlit_app.py`), lo cual es seguro **solo** si confías en el archivo.

> Si tu modelo real es el de difusión MedSegDiff completo, su arquitectura y su
> bucle de muestreo (100+ pasos) son distintos a esta U-Net de un solo paso y
> demasiado pesados para el tier gratuito. En ese caso reemplaza `model.py` por
> tu clase de modelo y añade el muestreo; conviene una GPU.

## Regenerar los pesos de demostración (opcional)
```bash
python train_demo.py   # vuelve a crear model_demo.pt
```
