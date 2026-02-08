FROM dustynv/l4t-pytorch:r36.4.0

WORKDIR /home/code

ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1


RUN python3 -m pip install -U pip setuptools wheel && \
    python3 -m pip install --no-cache-dir transformers timm matplotlib scipy

# Ejecutar entrenamiento al arrancar el contenedor
CMD ["python3", "/home/code/train_frcnn.py"]
