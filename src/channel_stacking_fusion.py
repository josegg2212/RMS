import cv2
import numpy as np
import os

# ============================================================================
# Fusion por apilado de canales (RGB + IR).
# - Lee pares RGB e IR con el mismo nombre.
# - Normaliza a [0,1], apila 3 canales RGB + 1 canal IR.
# - Guarda una imagen de 4 canales (RGBA) en la carpeta fused.
# ============================================================================

# Rutas a las imágenes RGB e IR
rgb_dir = './data/LLVIP/visible/train'
ir_dir = './data/LLVIP/infrared/train'
out_dir = './data/LLVIP/fused/train'

# Si no existe la carpeta de salida, la creamos
os.makedirs(out_dir, exist_ok=True)

# Lista solo imágenes válidas (evita archivos raros)
rgb_images = sorted([
    f for f in os.listdir(rgb_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
])

for rgb_name in rgb_images:
    # Ruta del RGB y su IR correspondiente
    rgb_path = os.path.join(rgb_dir, rgb_name)

    # En LLVIP, el IR normalmente tiene el mismo nombre
    ir_path = os.path.join(ir_dir, rgb_name)

    # Si no existe el par IR, se omite
    if not os.path.exists(ir_path):
        print(f"[SKIP] No IR pair for: {rgb_name}")
        continue

    # Cargar imágenes
    rgb_image = cv2.imread(rgb_path)  # BGR, 3 canales
    ir_image = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)  # 1 canal

    if rgb_image is None:
        print(f"[SKIP] Could not load RGB: {rgb_path}")
        continue
    if ir_image is None:
        print(f"[SKIP] Could not load IR: {ir_path}")
        continue

    # Redimensionar IR al tamaño de RGB
    ir_resized = cv2.resize(ir_image, (rgb_image.shape[1], rgb_image.shape[0]))

    # Normalizar a [0,1]
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    ir_norm = ir_resized.astype(np.float32) / 255.0

    # Apilar canales => [H, W, 4] (3 RGB + 1 IR)
    fused = np.dstack((rgb_norm, ir_norm))
    fused_u8 = np.clip(fused * 255, 0, 255).astype(np.uint8)

    # Nombre de salida con el mismo base name
    base = os.path.splitext(rgb_name)[0]  # ej: "250423" o "190001"
    fused_path = os.path.join(out_dir, f"{base}.png")

    # Guardar
    cv2.imwrite(fused_path, fused_u8)
    print(f"[OK] Saved: {fused_path}")
