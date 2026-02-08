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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.ops import box_iou

from transformers import DetrForObjectDetection


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger("train_detr")


logger = setup_logging()


# ============================================================================
# TRAIN DETR (1-stage, end-to-end) + VOC XML + 4 canales (RGBA / RGB+IR)
#
# Estructura equivalente a tu train_frcnn_voc_eval.py:
#  1) Config
#  2) Dataset (VOC XML)
#  3) Collate
#  4) Model (load + patch 4ch)
#  5) Train loop
#  6) Valid loop + métricas
#  7) Save best
#
# CAMBIOS vs Faster R-CNN (2-stages):
# - Faster R-CNN: Backbone + RPN + RoIHeads, targets: boxes XYXY + labels.
# - DETR: Backbone + Transformer + queries. No RPN, no RoIHeads.
#         targets: class_labels + boxes NORMALIZADAS en formato cxcywh.
# - HuggingFace DetrImageProcessor NO soporta bien 4 canales en preprocess;
#   por eso hacemos el preprocess manual (normalize + padding + pixel_mask).
# ============================================================================


# -----------------------------
# Dataset VOC (imagenes + XML)
# -----------------------------
class VOCFolderDataset(Dataset):
    def __init__(self, folder: str, class_name: str = "Person", annotations_dir: str = None):
        self.folder = Path(folder)
        self.class_name = class_name.strip().lower()
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        self.images = sorted([p for p in self.folder.iterdir() if p.suffix.lower() in exts])

    def __len__(self):
        return len(self.images)

    def _parse_xml_xyxy(self, xml_path: Path):
        boxes = []
        if not xml_path.exists():
            return torch.zeros((0, 4), dtype=torch.float32)

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

        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32)

        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        if self.annotations_dir:
            xml_path = self.annotations_dir / f"{img_path.stem}.xml"
        else:
            xml_path = img_path.with_suffix(".xml")

        # 4 canales (RGBA)
        img = Image.open(img_path).convert("RGBA")
        w, h = img.size

        boxes_xyxy = self._parse_xml_xyxy(xml_path)

        # En DETR (HF) el id de clase será 0 si solo tienes "person"
        labels = torch.zeros((boxes_xyxy.shape[0],), dtype=torch.int64)

        target = {
            "boxes_xyxy": boxes_xyxy,              # Para métricas (XYXY en píxeles)
            "labels": labels,                      # 0=person
            "orig_size": torch.tensor([h, w]),     # (H,W)
            "image_id": torch.tensor([idx]),
        }
        return img, target


# -----------------------------
# Patch primera conv 3->4 canales
# -----------------------------
def patch_first_conv_to_4ch(model: torch.nn.Module) -> bool:
    """
    Similar a tu patch en Faster R-CNN, pero aquí el backbone está dentro de transformers.
    Busca la primera Conv2d con in_channels=3 y la reemplaza por in_channels=4.
    """
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Conv2d) and child.in_channels == 3:
                new_conv = torch.nn.Conv2d(
                    4,
                    child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode,
                )
                with torch.no_grad():
                    new_conv.weight[:, :3] = child.weight
                    new_conv.weight[:, 3:4] = child.weight.mean(dim=1, keepdim=True)
                    if child.bias is not None:
                        new_conv.bias.copy_(child.bias)
                setattr(module, name, new_conv)
                return True
    return False


# -----------------------------
# Conversión cajas: XYXY pix -> cxcywh normalizado [0..1]
# (DETR espera esto en labels)
# -----------------------------
def xyxy_pix_to_cxcywh_norm(boxes_xyxy: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return torch.stack([cx, cy, bw, bh], dim=1).clamp(0, 1)


# -----------------------------
# Preprocess manual para DETR con 4ch
# CAMBIO vs Faster:
# - Faster: transform de torchvision + model(images, targets) directo.
# - DETR: necesita pixel_values + pixel_mask + labels (class_labels, boxes norm).
# Además, HuggingFace processor falla con 4ch => manual.
# -----------------------------
def make_collate_fn_4ch(image_mean, image_std):
    mean = torch.tensor(image_mean, dtype=torch.float32).view(4, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(4, 1, 1)

    def collate_fn(batch):
        images, targets = zip(*batch)

        # Convertir a tensores 4xHxW float en [0,1]
        tensors = []
        sizes = []
        for im in images:
            arr = np.array(im, dtype=np.float32) / 255.0  # HWC, 4
            ten = torch.from_numpy(arr).permute(2, 0, 1)  # CHW, 4
            tensors.append(ten)
            sizes.append((ten.shape[1], ten.shape[2]))  # (H,W)

        max_h = max(h for h, w in sizes)
        max_w = max(w for h, w in sizes)

        pixel_values = []
        pixel_mask = []
        labels = []
        targets_out = []

        for ten, t, (h, w) in zip(tensors, targets, sizes):
            # Normalización 4 canales
            ten = (ten - mean) / std

            # Padding a tamaño máximo del batch
            pad_h = max_h - h
            pad_w = max_w - w
            ten_padded = F.pad(ten, (0, pad_w, 0, pad_h), value=0.0)  # pad W then H
            mask = torch.zeros((max_h, max_w), dtype=torch.bool)
            mask[:h, :w] = True  # True = píxel válido (no padding)

            pixel_values.append(ten_padded)
            pixel_mask.append(mask)

            # Labels en formato DETR:
            #  - class_labels: (N,)
            #  - boxes: (N,4) en cxcywh normalizado
            boxes_xyxy = t["boxes_xyxy"]
            class_labels = t["labels"]
            boxes_norm = xyxy_pix_to_cxcywh_norm(boxes_xyxy, h=h, w=w)

            labels.append({
                "class_labels": class_labels,
                "boxes": boxes_norm,
            })

            # guardamos target original para métricas (orig_size, boxes_xyxy)
            targets_out.append(t)

        batch_encoding = {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "pixel_mask": torch.stack(pixel_mask, dim=0),
            "labels": labels,
        }
        return batch_encoding, targets_out

    return collate_fn


# -----------------------------
# Postprocess DETR: outputs -> boxes XYXY en píxeles + scores
# CAMBIO vs Faster:
# - Faster devuelve cajas finales directamente.
# - DETR devuelve pred_boxes (cxcywh norm) y logits; hay que convertir.
# -----------------------------
@torch.no_grad()
def detr_postprocess(outputs, target_sizes_hw, score_thresh=0.0):
    """
    target_sizes_hw: Tensor (B,2) con (H,W) original por imagen.
    Devuelve lista de dicts con boxes XYXY en píxeles.
    """
    # outputs.logits: (B, num_queries, num_classes+1)
    # outputs.pred_boxes: (B, num_queries, 4) cxcywh norm
    probs = outputs.logits.softmax(-1)
    scores_all, labels_all = probs[..., :-1].max(-1)  # quitamos "no-object"

    results = []
    for scores, labels, boxes, (h, w) in zip(scores_all, labels_all, outputs.pred_boxes, target_sizes_hw.tolist()):
        keep = scores >= score_thresh
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        # cxcywh -> xyxy (norm)
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = cx - 0.5 * bw
        y1 = cy - 0.5 * bh
        x2 = cx + 0.5 * bw
        y2 = cy + 0.5 * bh
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)

        scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes_xyxy.device)
        boxes_xyxy = boxes_xyxy * scale

        results.append({
            "scores": scores.detach().cpu(),
            "labels": labels.detach().cpu(),
            "boxes": boxes_xyxy.detach().cpu(),
        })
    return results


# -----------------------------
# Métricas (igual que en tus scripts)
# -----------------------------
@torch.no_grad()
def precision_recall_at(preds, gts, conf_th=0.25, iou_th=0.5):
    tp = 0
    fp = 0
    total_gt = sum(int(gt["boxes_xyxy"].shape[0]) for gt in gts)

    for p, gt in zip(preds, gts):
        p_boxes = p["boxes"]
        p_scores = p["scores"]
        gt_boxes = gt["boxes_xyxy"]

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
    return float(precision), float(recall)


def ap_at_iou(preds, gts, iou_th=0.5, min_score_for_map=0.05):
    records = []
    total_gt = sum(int(gt["boxes_xyxy"].shape[0]) for gt in gts)

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
        gt_boxes = gts[img_id]["boxes_xyxy"]

        if gt_boxes.numel() == 0:
            tps.append(0); fps.append(1)
            continue

        ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_idx = torch.max(ious, dim=0)
        best_idx = int(best_idx.item())

        if best_iou.item() >= iou_th and best_idx not in matched[img_id]:
            tps.append(1); fps.append(0)
            matched[img_id].add(best_idx)
        else:
            tps.append(0); fps.append(1)

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
    return float(ap / 101.0)


@torch.no_grad()
def evaluate_detection(model, dataloader, device, conf_th=0.25, min_score_for_map=0.05):
    model.eval()
    preds = []
    gts = []

    for batch, targets in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in lab.items()} for lab in batch["labels"]]
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in lab.items()} for lab in batch["labels"]]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)  # (B,2) (H,W)
        processed = detr_postprocess(outputs, target_sizes, score_thresh=0.0)

        for det, tgt in zip(processed, targets):
            # Solo 1 clase => filtramos label 0 (person)
            keep = det["labels"] == 0
            preds.append({"boxes": det["boxes"][keep], "scores": det["scores"][keep]})
            gts.append(tgt)

    prec, rec = precision_recall_at(preds, gts, conf_th=conf_th, iou_th=0.5)
    ap50 = ap_at_iou(preds, gts, iou_th=0.5, min_score_for_map=min_score_for_map)

    iou_ths = [0.50 + 0.05 * i for i in range(10)]
    map5095 = sum(ap_at_iou(preds, gts, iou_th=t, min_score_for_map=min_score_for_map) for t in iou_ths) / len(iou_ths)

    return {"precision": prec, "recall": rec, "map50": ap50, "map5095": float(map5095)}


def make_run_dir(root: str, prefix: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{prefix}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _safe_cuda_sync(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


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


def main():
    # -----------------------------
    # CONFIG (igual que tu script)
    # -----------------------------
    train_dir = "fused/train"
    valid_dir = "fused/test"
    annotations_dir = "fused/Annotations"

    epochs = 10
    batch_size = 2
    lr = 1e-5
    device = "cuda"
    num_workers = 0
    out = "detr_personas.pth"
    save_best = True
    conf_th = 0.9
    min_score_for_map = 0.05

    base_model = "facebook/detr-resnet-50"

    # Tus mean/std 4ch (los que usabas en Faster)
    image_mean = [0.19630877966050161, 0.1903079616632919, 0.12459238259621531, 0.2914455310556421]
    image_std  = [0.19535953989083024, 0.19516226791592045, 0.18131376178315223, 0.19002960530299567]
    results_root = "results"
    run_dir = make_run_dir(results_root, "train_detr")
    out_path = Path(out)
    best_weights_path = run_dir / out_path.name
    run_started_at = datetime.now().isoformat()

    # -----------------------------
    # Device
    # -----------------------------
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA no disponible, usando CPU.")

    # -----------------------------
    # Dataset / DataLoader
    # -----------------------------
    train_ds = VOCFolderDataset(train_dir, class_name="Person", annotations_dir=annotations_dir)
    valid_ds = VOCFolderDataset(valid_dir, class_name="Person", annotations_dir=annotations_dir)

    collate_fn = make_collate_fn_4ch(image_mean, image_std)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

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
        "base_model": base_model,
        "image_mean": image_mean,
        "image_std": image_std,
        "run_dir": str(run_dir),
        "train_images": len(train_ds),
        "valid_images": len(valid_ds),
        "started_at": run_started_at,
    }
    save_json(run_dir / "config.json", config)

    # -----------------------------
    # Model (DETR 1-stage)
    # CAMBIO vs Faster: aquí no hay roi_heads.box_predictor, sino head interno.
    # -----------------------------
    model = DetrForObjectDetection.from_pretrained(
        base_model,
        num_labels=1,                 # 1 clase (person)
        ignore_mismatched_sizes=True, # classifier reinit
        low_cpu_mem_usage=False,      # opcional: quita parte de warnings "meta"
    )

    patched = patch_first_conv_to_4ch(model)
    if not patched:
        print("WARNING: no se pudo parchear conv 3->4 automáticamente. Revisa el backbone.")

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    logger.info("Train images: %s | Valid images: %s | device: %s", len(train_ds), len(valid_ds), device)
    logger.info("epochs=%s batch_size=%s lr=%s conf_th=%s", epochs, batch_size, lr, conf_th)

    best_map50 = -1.0
    t0_total = time.time()
    best_epoch = -1
    history = []
    batch_times_s = []

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # -----------------------------
    # Train loop
    # CAMBIO vs Faster: outputs.loss y outputs.loss_dict vienen del set-matching interno.
    # -----------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for step, (batch, _) in enumerate(train_dl, start=1):
            _safe_cuda_sync(device)
            t_batch0 = time.perf_counter()
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in lab.items()} for lab in batch["labels"]]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            _safe_cuda_sync(device)
            t_batch1 = time.perf_counter()
            batch_times_s.append(t_batch1 - t_batch0)

            if step == 1 or step % 20 == 0:
                parts = {k: float(v.detach().cpu()) for k, v in outputs.loss_dict.items()}
                logger.info(
                    "[Epoch %s/%s] step %s/%s avg_loss=%.4f parts=%s",
                    epoch, epochs, step, len(train_dl), running_loss / step, parts
                )

        avg_train_loss = running_loss / max(1, len(train_dl))

        # -----------------------------
        # Validación + métricas
        # -----------------------------
        val_metrics = evaluate_detection(model, valid_dl, device=device, conf_th=conf_th, min_score_for_map=min_score_for_map)

        epoch_time = time.time() - t0
        logger.info(
            "== Epoch %s done | time=%.1fs | train_loss=%.4f | val: P=%.3f R=%.3f mAP50=%.3f mAP50-95=%.3f",
            epoch, epoch_time, avg_train_loss,
            val_metrics["precision"], val_metrics["recall"],
            val_metrics["map50"], val_metrics["map5095"]
        )

        row = {
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "precision": float(val_metrics["precision"]),
            "recall": float(val_metrics["recall"]),
            "map50": float(val_metrics["map50"]),
            "map5095": float(val_metrics["map5095"]),
            "epoch_time_s": float(epoch_time),
        }
        history.append(row)

        if save_best and val_metrics["map50"] > best_map50:
            best_map50 = val_metrics["map50"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_weights_path)
            if out_path != best_weights_path:
                torch.save(model.state_dict(), out_path)
            logger.info("  -> NEW BEST (mAP50=%.3f) saved to %s", best_map50, best_weights_path)
        logger.info("Best epoch so far: %s with mAP50=%s", best_epoch, best_map50)
    total_time = time.time() - t0_total

    if not save_best:
        torch.save(model.state_dict(), best_weights_path)
        if out_path != best_weights_path:
            torch.save(model.state_dict(), out_path)

    logger.info("Done. Total time: %.2f min | saved: %s | best_epoch: %s", total_time / 60.0, best_weights_path, best_epoch)

    best = max(history, key=lambda x: x["map50"]) if history else None
    last = history[-1] if history else None
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
