# RMS

Repositorio para entrenamiento y evaluacion de deteccion de personas con imagenes RGB+IR (4 canales). Incluye scripts para fusionar canales, calcular estadisticas, entrenar y predecir con Faster R-CNN y DETR.

**Datos y modelos (no incluidos)**
Por tamaño, ni el dataset LLVIP ni los pesos entrenados se incluyen en el repo. Hay que descargar el dataset, fusionar RGB+IR, entrenar para generar los `.pth` y luego usar esos pesos para predecir.

**Requisitos**
- Python 3
- PyTorch + torchvision
- transformers, timm
- numpy, opencv-python, pillow, matplotlib, scipy

**Ejecucion Local**
1. Entra a `RMS/src`.
2. Prepara el dataset fusionado con esta estructura:
```
fused/
  train/
  test/
  Annotations/   # XML en formato VOC
```
3. (Opcional) Genera las imagenes fusionadas desde RGB+IR con `channel_stacking_fusion.py` y mueve o copia el resultado a `fused/train`. Si tu dataset esta en otra ruta, ajusta `rgb_dir`, `ir_dir` y `out_dir` dentro del script.
4. (Opcional) Calcula media/desviacion por canal:
```
python3 compute_mean_std_4ch.py --dirs fused/train
```
5. Entrena:
```
python3 train_frcnn.py
python3 train_detr.py
```
6. Predice y evalua:
```
python3 predict_frcnn.py
python3 predict_detr.py
```

Notas:
- Las rutas estan dentro de los scripts. Si cambias ubicacion del dataset, ajusta `train_dir`, `valid_dir`, `annotations_dir`, `input_dir` y `weights`.
- Los resultados se guardan en `results/` con un subdirectorio por corrida.

**Ejecucion Con Docker**
1. Desde la raiz del repo:
```
docker compose build
```
2. Entrenar Faster R-CNN (comando por defecto):
```
docker compose run --rm rms_pytorch
```
3. Ejecutar otros scripts:
```
docker compose run --rm rms_pytorch python3 train_detr.py
docker compose run --rm rms_pytorch python3 predict_frcnn.py
docker compose run --rm rms_pytorch python3 predict_detr.py
```

Notas:
- El `docker-compose.yaml` usa GPU, requiere NVIDIA Container Runtime.
- Se monta `./src` en `/home/code`, por eso los comandos se ejecutan desde esa carpeta.

**Estructura Del Repo**
- `README.md` — este documento.
- `docker/Dockerfile` — imagen con dependencias (Jetson L4T + PyTorch) y comando por defecto de entrenamiento.
- `docker-compose.yaml` — servicio Docker con GPU y volumen `./src -> /home/code`.
- `src/channel_stacking_fusion.py` — fusiona RGB+IR en una imagen de 4 canales.
- `src/compute_mean_std_4ch.py` — calcula media y desviacion por canal.
- `src/train_frcnn.py` — entrenamiento de Faster R-CNN con imagenes 4 canales y XML VOC.
- `src/train_detr.py` — entrenamiento de DETR con imagenes 4 canales y XML VOC.
- `src/predict_frcnn.py` — inferencia/evaluacion con Faster R-CNN y guardado de metricas/imagenes.
- `src/predict_detr.py` — inferencia/evaluacion con DETR y guardado de metricas/imagenes.
- `src/frcnn_personas.pth` — pesos generados por entrenamiento (no incluidos en un repo limpio).
- `src/detr_personas.pth` — pesos generados por entrenamiento (no incluidos en un repo limpio).
- `src/fused/` — dataset fusionado esperado (train/test/Annotations).
- `src/Annotation/` — carpeta auxiliar para anotaciones (opcional).
- `src/results/` — salidas de entrenamiento y prediccion.
