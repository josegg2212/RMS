import time
import json
import csv
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.ops import box_iou

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# -----------------------------
# Logging
# -----------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger("train_frcnn")


logger = setup_logging()


# ============================================================================
# Entrenamiento y validacion Faster R-CNN para 1 clase (Person) con VOC.
# Pipeline general:
# - Dataset lee imagenes RGBA y cajas desde XML VOC.
# - Modelo preentrenado se adapta a 4 canales y a 2 clases (bg + Person).
# - Entrenamiento por epoca con validacion y metricas basicas.
# - Guardado del mejor modelo segun mAP@0.5 (precision promedio con solape 0.5).
# Cambios vs pipeline 3ch:
# - Imagenes en RGBA (4 canales).
# - Conv1 acepta 4 canales y el canal extra hereda promedio RGB.
# - image_mean/image_std contienen 4 valores.
# ============================================================================


# --- Dataset VOC en carpeta ---
# Lee imagenes y cajas desde XML. Filtra solo la clase objetivo.

class VOCFolderDataset(Dataset):
    def __init__(self, folder: str, class_name: str = "Person", annotations_dir: str = None):
        self.folder = Path(folder)
        self.class_name = class_name.strip().lower()
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.images = sorted([p for p in self.folder.iterdir() if p.suffix.lower() in exts])

    def __len__(self):
        return len(self.images)

    # Lee el XML y devuelve cajas + labels de la clase objetivo
    def _parse_xml(self, xml_path: Path):
        boxes = []
        labels = []

        if not xml_path.exists():
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name != self.class_name:
                continue

            bnd = obj.find("bndbox")
            if bnd is None:
                continue

            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)

        if len(boxes) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )

    def __getitem__(self, idx):
        img_path = self.images[idx]
        if self.annotations_dir:
            xml_path = self.annotations_dir / f"{img_path.stem}.xml"
        else:
            xml_path = img_path.with_suffix(".xml")

        # Carga en RGBA y pasa a tensor (C,H,W)
        img = Image.open(img_path).convert("RGBA")
        image = TF.to_tensor(img)

        boxes, labels = self._parse_xml(xml_path)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        return image, target


# --- Collate para listas de imagenes/targets (Faster R-CNN) ---
# El modelo espera una lista de imagenes y una lista de targets.

def collate_fn(batch):
    return tuple(zip(*batch))


# --- Modelo: adaptacion a 4 canales + cabeza de 2 clases ---
# Ajusta la primera capa para aceptar 4 canales, manteniendo pesos.

# Reemplaza la primera convolucion por una de 4 canales
def _patch_first_conv_for_4ch(model):
    old_conv = model.backbone.body.conv1
    new_conv = torch.nn.Conv2d(
        4,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

    model.backbone.body.conv1 = new_conv


# Crea el modelo base y ajusta normalizacion/clases

def get_model(num_classes: int = 2, image_mean=None, image_std=None):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    _patch_first_conv_for_4ch(model)

    if image_mean is not None:
        model.transform.image_mean = image_mean
    if image_std is not None:
        model.transform.image_std = image_std

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# --- Metricas de deteccion (1 clase) ---
# IoU = solape entre caja predicha y real. mAP = precision promedio.
# Precision/Recall con umbral y AP promediando la curva de precision.

@torch.no_grad()
def precision_recall_at(preds, gts, conf_th=0.25, iou_th=0.5):
    tp = 0
    fp = 0
    total_gt = sum(int(gt["boxes"].shape[0]) for gt in gts)

    for p, gt in zip(preds, gts):
        p_boxes = p["boxes"]
        p_scores = p["scores"]
        gt_boxes = gt["boxes"]

        keep = p_scores >= conf_th
        p_boxes = p_boxes[keep]
        p_scores = p_scores[keep]

        if p_scores.numel() > 0:
            order = torch.argsort(p_scores, descending=True)
            p_boxes = p_boxes[order]
            p_scores = p_scores[order]

        matched = set()
        for box in p_boxes:
            if gt_boxes.numel() == 0:
                fp += 1
                continue

            ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_idx = torch.max(ious, dim=0)
            best_idx = int(best_idx.item())

            if best_iou.item() >= iou_th and best_idx not in matched:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (total_gt + 1e-9)
    return precision, recall


# AP@IoU con interpolacion (101 puntos)
def ap_at_iou(preds, gts, iou_th=0.5, min_score_for_map=0.05):
    records = []
    total_gt = sum(int(gt["boxes"].shape[0]) for gt in gts)

    for img_id, (p, gt) in enumerate(zip(preds, gts)):
        boxes = p["boxes"]
        scores = p["scores"]

        keep = scores >= min_score_for_map
        boxes = boxes[keep]
        scores = scores[keep]

        for s, b in zip(scores.tolist(), boxes):
            records.append((s, img_id, b))

    records.sort(key=lambda x: x[0], reverse=True)
    matched = [set() for _ in range(len(gts))]

    tps = []
    fps = []

    for score, img_id, box in records:
        gt_boxes = gts[img_id]["boxes"]

        if gt_boxes.numel() == 0:
            tps.append(0)
            fps.append(1)
            continue

        ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_idx = torch.max(ious, dim=0)
        best_idx = int(best_idx.item())

        if best_iou.item() >= iou_th and best_idx not in matched[img_id]:
            tps.append(1)
            fps.append(0)
            matched[img_id].add(best_idx)
        else:
            tps.append(0)
            fps.append(1)

    if len(tps) == 0:
        return 0.0

    tps = torch.tensor(tps, dtype=torch.float32)
    fps = torch.tensor(fps, dtype=torch.float32)

    tp_cum = torch.cumsum(tps, dim=0)
    fp_cum = torch.cumsum(fps, dim=0)

    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall = tp_cum / (total_gt + 1e-9)

    ap = 0.0
    for r in torch.linspace(0, 1, 101):
        mask = recall >= r
        p = torch.max(precision[mask]).item() if torch.any(mask) else 0.0
        ap += p
    ap /= 101.0
    return ap


# --- Evaluacion de deteccion sobre un DataLoader ---
# Devuelve precision/recall y mAP con varios umbrales de IoU (solape).

@torch.no_grad()
def evaluate_detection(model, dataloader, device, conf_th=0.25, min_score_for_map=0.05):
    model.eval()

    preds = []
    gts = []

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            labels = out["labels"].detach().cpu()
            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()

            keep = labels == 1
            boxes = boxes[keep]
            scores = scores[keep]

            preds.append({"boxes": boxes, "scores": scores})
            gts.append({"boxes": tgt["boxes"].detach().cpu()})

    prec, rec = precision_recall_at(preds, gts, conf_th=conf_th, iou_th=0.5)

    ap50 = ap_at_iou(preds, gts, iou_th=0.5, min_score_for_map=min_score_for_map)

    iou_ths = [0.50 + 0.05 * i for i in range(10)]
    aps = [ap_at_iou(preds, gts, iou_th=t, min_score_for_map=min_score_for_map) for t in iou_ths]
    map5095 = float(sum(aps) / len(aps))

    return {
        "precision": float(prec),
        "recall": float(rec),
        "map50": float(ap50),
        "map5095": float(map5095),
    }


# Crea carpeta de corrida con timestamp
def make_run_dir(root: str, prefix: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{prefix}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# Guarda diccionario en JSON formateado
def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# Guarda filas (dict) en CSV
def save_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# Grafica curvas de entrenamiento y metricas
def plot_training_curves(history, run_dir: Path):
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    precision = [h["precision"] for h in history]
    recall = [h["recall"] for h in history]
    map50 = [h["map50"] for h in history]
    map5095 = [h["map5095"] for h in history]

    plt.figure()
    plt.plot(epochs, train_loss, marker="o")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "train_loss.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, precision, marker="o", label="Precision")
    plt.plot(epochs, recall, marker="o", label="Recall")
    plt.title("Precision / Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "precision_recall.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, map50, marker="o", label="mAP@0.5")
    plt.plot(epochs, map5095, marker="o", label="mAP@0.5:0.95")
    plt.title("mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "map.png", dpi=160)
    plt.close()


# Sincroniza CUDA para medir tiempos con precision
def _safe_cuda_sync(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# Snapshot simple de uso/memoria GPU (si hay CUDA)
def _gpu_snapshot():
    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "max_mem_allocated_bytes": None,
            "max_mem_reserved_bytes": None,
            "nvidia_smi_util_gpu_pct": None,
            "nvidia_smi_util_mem_pct": None,
        }

    max_alloc = int(torch.cuda.max_memory_allocated())
    max_reserved = int(torch.cuda.max_memory_reserved())

    util_gpu = None
    util_mem = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        if out:
            parts = [p.strip() for p in out[0].split(",")]
            if len(parts) >= 2:
                util_gpu = int(parts[0])
                util_mem = int(parts[1])
    except Exception:
        pass

    return {
        "gpu_available": True,
        "max_mem_allocated_bytes": max_alloc,
        "max_mem_reserved_bytes": max_reserved,
        "nvidia_smi_util_gpu_pct": util_gpu,
        "nvidia_smi_util_mem_pct": util_mem,
    }


# --- Entrenamiento principal ---
# Configura rutas, parametros y ejecuta train/valid por epoca.

def main():
    # --- Configuracion principal ---
    train_dir = "fused/train"
    valid_dir = "fused/test"
    annotations_dir = "fused/Annotations"
    epochs = 10
    batch_size = 2
    lr = 0.005
    device = "cuda"
    num_workers = 0
    out = "frcnn_personas.pth"
    save_best = True
    conf_th = 0.9
    min_score_for_map = 0.05
    image_mean = [0.19630877966050161, 0.1903079616632919, 0.12459238259621531, 0.2914455310556421]
    image_std = [0.19535953989083024, 0.19516226791592045, 0.18131376178315223, 0.19002960530299567]
    results_root = "results"
    run_dir = make_run_dir(results_root, "train_frcnn")
    out_path = Path(out)
    best_weights_path = run_dir / out_path.name
    run_started_at = datetime.now().isoformat()

    # --- Dispositivo ---
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA no disponible, usando CPU.")

    # --- Dataset y DataLoaders ---
    # Imagenes RGBA + XML VOC, solo clase Person.
    train_ds = VOCFolderDataset(train_dir, class_name="Person", annotations_dir=annotations_dir)
    valid_ds = VOCFolderDataset(valid_dir, class_name="Person", annotations_dir=annotations_dir)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )

    config = {
        "train_dir": train_dir,
        "valid_dir": valid_dir,
        "annotations_dir": annotations_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": device,
        "num_workers": num_workers,
        "out": out,
        "save_best": save_best,
        "conf_th": conf_th,
        "min_score_for_map": min_score_for_map,
        "image_mean": image_mean,
        "image_std": image_std,
        "run_dir": str(run_dir),
        "train_images": len(train_ds),
        "valid_images": len(valid_ds),
        "started_at": run_started_at,
    }
    save_json(run_dir / "config.json", config)

    # --- Modelo + Optimizador ---
    # Modelo adaptado a 4 canales y normalizado con mean/std de 4 valores.
    model = get_model(num_classes=2, image_mean=image_mean, image_std=image_std).to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    logger.info("Train images: %s | Valid images: %s | device: %s", len(train_ds), len(valid_ds), device)
    logger.info("epochs=%s batch_size=%s lr=%s conf_th=%s", epochs, batch_size, lr, conf_th)

    history = []
    best_map50 = -1.0
    best_epoch = -1

    t0_total = time.time()
    batch_times_s = []

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- Loop de entrenamiento ---
    for epoch in range(1, epochs + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        running_loss = 0.0

        for step, (images, targets) in enumerate(train_dl, start=1):
            _safe_cuda_sync(device)
            t_batch0 = time.perf_counter()
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _safe_cuda_sync(device)
            t_batch1 = time.perf_counter()
            batch_times_s.append(t_batch1 - t_batch0)

            if step == 1 or step % 20 == 0:
                loss_parts = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
                avg_loss = running_loss / step
                logger.info(
                    "[Epoch %s/%s] step %s/%s avg_loss=%.4f parts=%s",
                    epoch, epochs, step, len(train_dl), avg_loss, loss_parts
                )

        avg_train_loss = running_loss / max(1, len(train_dl))

        # --- Validacion (mAP/Precision/Recall) ---
        val_metrics = evaluate_detection(
            model, valid_dl, device=device,
            conf_th=conf_th,
            min_score_for_map=min_score_for_map
        )

        if device == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            **val_metrics,
            "epoch_time_s": float(epoch_time),
        }
        history.append(row)

        logger.info(
            "== Epoch %s done | time=%.1fs | train_loss=%.4f | val: P=%.3f R=%.3f mAP50=%.3f mAP50-95=%.3f",
            epoch, epoch_time, avg_train_loss,
            val_metrics["precision"], val_metrics["recall"],
            val_metrics["map50"], val_metrics["map5095"]
        )

        # --- Checkpoint del mejor modelo ---
        if save_best and val_metrics["map50"] > best_map50:
            best_map50 = val_metrics["map50"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_weights_path)
            if out_path != best_weights_path:
                torch.save(model.state_dict(), out_path)
            logger.info("  -> NEW BEST (mAP50=%.3f) saved to %s", best_map50, best_weights_path)

    total_time = time.time() - t0_total

    if not save_best:
        torch.save(model.state_dict(), best_weights_path)
        if out_path != best_weights_path:
            torch.save(model.state_dict(), out_path)

    # --- Resumen final ---
    best = max(history, key=lambda x: x["map50"]) if history else None
    logger.info("================= TRAIN SUMMARY =================")
    logger.info("Total time: %.2f min | saved: %s", total_time / 60.0, best_weights_path)
    if best is not None:
        logger.info(
            "Best epoch (by mAP50): %s | P=%.3f R=%.3f mAP50=%.3f mAP50-95=%.3f train_loss=%.4f time=%.1fs",
            best["epoch"], best["precision"], best["recall"],
            best["map50"], best["map5095"], best["train_loss"], best["epoch_time_s"]
        )
    last = history[-1] if history else None
    if last is not None:
        logger.info(
            "Last epoch: %s | P=%.3f R=%.3f mAP50=%.3f mAP50-95=%.3f train_loss=%.4f time=%.1fs",
            last["epoch"], last["precision"], last["recall"],
            last["map50"], last["map5095"], last["train_loss"], last["epoch_time_s"]
        )
    logger.info("================================================")

    summary = {
        "total_time_s": float(total_time),
        "total_time_min": float(total_time / 60.0),
        "best_epoch": best["epoch"] if best else None,
        "best_map50": best["map50"] if best else None,
        "best_map5095": best["map5095"] if best else None,
        "best_precision": best["precision"] if best else None,
        "best_recall": best["recall"] if best else None,
        "last_epoch": last["epoch"] if last else None,
        "last_map50": last["map50"] if last else None,
        "last_map5095": last["map5095"] if last else None,
        "last_precision": last["precision"] if last else None,
        "last_recall": last["recall"] if last else None,
        "weights_path": str(best_weights_path),
        "weights_alias": str(out_path),
        "ended_at": datetime.now().isoformat(),
    }

    if history:
        save_json(run_dir / "history.json", history)
        save_csv(run_dir / "history.csv", history, fieldnames=list(history[0].keys()))

    save_json(run_dir / "summary.json", summary)
    (run_dir / "summary.txt").write_text(
        "\n".join([
            "TRAIN SUMMARY",
            f"run_dir: {run_dir}",
            f"total_time_min: {summary['total_time_min']:.2f}",
            f"best_epoch: {summary['best_epoch']}",
            f"best_map50: {summary['best_map50']}",
            f"best_map5095: {summary['best_map5095']}",
            f"best_precision: {summary['best_precision']}",
            f"best_recall: {summary['best_recall']}",
            f"last_epoch: {summary['last_epoch']}",
            f"last_map50: {summary['last_map50']}",
            f"last_map5095: {summary['last_map5095']}",
            f"last_precision: {summary['last_precision']}",
            f"last_recall: {summary['last_recall']}",
            f"weights_path: {summary['weights_path']}",
            f"weights_alias: {summary['weights_alias']}",
        ]),
        encoding="utf-8",
    )

    plot_training_curves(history, run_dir)

    run_info = {
        "started_at": run_started_at,
        "ended_at": summary["ended_at"],
        "total_images_train": len(train_ds),
        "total_images_valid": len(valid_ds),
        "run_dir": str(run_dir),
    }
    save_json(run_dir / "run_info.json", run_info)

    # Percentil simple para tiempos (sin numpy)
    def _percentile(xs, p):
        if not xs:
            return None
        xs_sorted = sorted(xs)
        k = int(round((p / 100.0) * (len(xs_sorted) - 1)))
        return float(xs_sorted[k])

    perf = {
        "total_time_s": float(total_time),
        "avg_epoch_time_s": float(sum(h["epoch_time_s"] for h in history) / len(history)) if history else None,
        "avg_batch_time_s": float(sum(batch_times_s) / len(batch_times_s)) if batch_times_s else None,
        "p95_batch_time_s": _percentile(batch_times_s, 95) if batch_times_s else None,
        "gpu": _gpu_snapshot(),
    }
    save_json(run_dir / "perf.json", perf)

    logger.info("Results saved to: %s", run_dir.resolve())


if __name__ == "__main__":
    main()
