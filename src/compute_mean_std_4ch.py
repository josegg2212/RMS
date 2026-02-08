import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF


# ============================================================================
# Calculo de media y desviacion por canal para imagenes 4 canales (RGBA).
# Pasos:
# - Recorre carpetas y recolecta rutas de imagenes.
# - Carga cada imagen como RGBA y acumula estadisticas por canal.
# - Devuelve y muestra los 4 valores para usar en entrenamiento/evaluacion.
# ============================================================================


# --- Recorrido de imagenes por directorios y extensiones ---
# Devuelve rutas de imagen validas segun las extensiones indicadas.

def iter_images(dirs, exts):
    for d in dirs:
        root = Path(d)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p


# --- Acumulacion de estadisticas por canal (4ch) ---
# Calcula media y desviacion estandar por canal con una sola pasada.

def compute_mean_std(image_paths):
    channel_sum = torch.zeros(4, dtype=torch.float64)
    channel_sum_sq = torch.zeros(4, dtype=torch.float64)
    total_pixels = 0

    for i, path in enumerate(image_paths, start=1):
        img = Image.open(path).convert("RGBA")
        t = TF.to_tensor(img).to(torch.float64)
        pixels = t.shape[1] * t.shape[2]
        channel_sum += t.view(4, -1).sum(dim=1)
        channel_sum_sq += (t * t).view(4, -1).sum(dim=1)
        total_pixels += pixels

        if i % 200 == 0:
            print(f"Processed {i} images...")

    # Evita division por cero si no hay imagenes
    if total_pixels == 0:
        raise RuntimeError("No images found. Check your paths and extensions.")

    mean = channel_sum / total_pixels
    std = (channel_sum_sq / total_pixels - mean * mean).sqrt()
    return mean, std


# --- CLI principal ---
# Lee carpetas/extensiones, calcula estadisticas y las imprime.

def main():
    # --- Argumentos CLI ---
    parser = argparse.ArgumentParser(
        description="Compute mean/std for 4-channel images (RGB+IR)."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["fused/train"],
        help="One or more directories with images.",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="File extensions to include.",
    )
    args = parser.parse_args()

    # --- Recoleccion de imagenes ---
    exts = {e.lower() for e in args.exts}
    image_paths = list(iter_images(args.dirs, exts))
    print(f"Found {len(image_paths)} images.")

    # --- Calculo de estadisticas por canal ---
    mean, std = compute_mean_std(image_paths)

    mean_list = [float(x) for x in mean]
    std_list = [float(x) for x in std]

    # --- Salida en formato facil de usar ---
    print("image_mean =", mean_list)
    print("image_std  =", std_list)
    print("Use these in train_frcnn_voc_eval.py (image_mean/image_std).")


if __name__ == "__main__":
    main()
