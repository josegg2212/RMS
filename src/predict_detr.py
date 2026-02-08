from pathlib import Path
import json
import csv
import time
import subprocess
import logging
from datetime import datetime
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.ops import box_iou

from transformers import DetrForObjectDetection


# -----------------------------
# Logging
# -----------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger("predict_detr")


logger = setup_logging()


# ============================================================================
# PREDICT / EVAL folder con DETR (1-stage) + 4 canales
#
# CAMBIOS vs predict_folder_eval.py (Faster):
# - Faster: model([img]) => boxes/scores/labels directos.
# - DETR: model(pixel_values, pixel_mask) => logits (puntajes) + pred_boxes (cxcywh: centro x/y, ancho/alto).
#         Hay que postprocesar (softmax + cxcywh->xyxy + escalar a pixeles).
# - No usamos DetrImageProcessor para preprocess porque 4ch rompe inferencia.
# ============================================================================


# Parchea la primera Conv2d del backbone (extractor) para aceptar 4 canales
def patch_first_conv_to_4ch(model: torch.nn.Module) -> bool:
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


# Lee anotaciones VOC y devuelve cajas XYXY en pixeles
def parse_voc_xml(xml_path: Path, class_name="Person"):
    class_name = class_name.strip().lower()
    boxes = []

    if not xml_path.exists():
        return torch.zeros((0, 4), dtype=torch.float32)

    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name != class_name:
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


# Postprocesa salidas DETR a cajas XYXY y scores en pixeles
@torch.no_grad()
def detr_postprocess(outputs, h: int, w: int, score_thresh=0.0):
    probs = outputs.logits.softmax(-1)
    scores, labels = probs[..., :-1].max(-1)  # quitamos no-object

    scores = scores[0]  # batch=1
    labels = labels[0]
    boxes = outputs.pred_boxes[0]  # (num_queries,4) cxcywh norm

    keep = scores >= score_thresh
    scores = scores[keep].cpu()
    labels = labels[keep].cpu()
    boxes = boxes[keep]

    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = cx - 0.5 * bw
    y1 = cy - 0.5 * bh
    x2 = cx + 0.5 * bw
    y2 = cy + 0.5 * bh
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)

    scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes_xyxy.device)
    boxes_xyxy = (boxes_xyxy * scale).cpu()

    return {"boxes": boxes_xyxy, "scores": scores, "labels": labels}


# Precision/recall para 1 clase con umbral de confianza
@torch.no_grad()
def precision_recall_at(preds, gts, conf_th=0.25, iou_th=0.5):
    tp = 0
    fp = 0
    total_gt = sum(int(gt.shape[0]) for gt in gts)

    for p, gt_boxes in zip(preds, gts):
        p_boxes = p["boxes"]
        p_scores = p["scores"]

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


# AP@IoU con interpolacion (101 puntos). IoU = solape entre cajas.
def ap_at_iou(preds, gts, iou_th=0.5, min_score_for_map=0.05):
    records = []
    total_gt = sum(int(gt.shape[0]) for gt in gts)

    for img_id, (p, gt_boxes) in enumerate(zip(preds, gts)):
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
        gt_boxes = gts[img_id]
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
    return float(ap / 101.0)


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


# Histograma de scores para diagnostico
def plot_score_histogram(scores_list, run_dir: Path):
    if not scores_list:
        return
    scores_list = [s for s in scores_list if s.numel() > 0]
    if not scores_list:
        return

    all_scores = torch.cat(scores_list, dim=0).cpu().numpy()
    plt.figure()
    plt.hist(all_scores, bins=30, range=(0, 1))
    plt.title("Score Histogram")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "score_histogram.png", dpi=160)
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


# Ejecucion principal: inferencia y metricas
def main():
    # -----------------------------
    # CONFIG principal
    # -----------------------------
    weights = "detr_personas.pth"
    input_dir = "fused/test"
    annotations_dir = "fused/Annotations"

    threshold = 0.9
    device = "cuda"
    class_name = "Person"
    min_score_for_map = 0.05

    base_model = "facebook/detr-resnet-50"

    image_mean = [0.19630877966050161, 0.1903079616632919, 0.12459238259621531, 0.2914455310556421]
    image_std = [0.19535953989083024, 0.19516226791592045, 0.18131376178315223, 0.19002960530299567]
    results_root = "results"
    run_dir = make_run_dir(results_root, "predict_detr")
    output_dir = run_dir / "images"
    run_started_at = datetime.now().isoformat()
    mean = torch.tensor(image_mean, dtype=torch.float32).view(1, 4, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(1, 4, 1, 1)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA no disponible, usando CPU.")

    # -----------------------------
    # Model load
    # -----------------------------
    model = DetrForObjectDetection.from_pretrained(
        base_model,
        num_labels=1,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    patch_first_conv_to_4ch(model)

    sd = torch.load(weights, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    # -----------------------------
    # Paths
    # -----------------------------
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_dir = Path(annotations_dir) if annotations_dir else None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])

    config = {
        "weights": weights,
        "input_dir": input_dir,
        "annotations_dir": annotations_dir,
        "output_dir": str(out_dir),
        "threshold": threshold,
        "device": device,
        "class_name": class_name,
        "min_score_for_map": min_score_for_map,
        "base_model": base_model,
        "image_mean": image_mean,
        "image_std": image_std,
        "run_dir": str(run_dir),
        "num_images": len(image_files),
        "started_at": run_started_at,
    }
    save_json(run_dir / "config.json", config)

    preds_all = []
    gts_all = []
    has_any_xml = False

    logger.info("Images found: %s", len(image_files))
    scores_for_hist = []
    image_times_s = []
    infer_times_s = []

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    preds_csv_path = run_dir / "predictions.csv"
    preds_jsonl_path = run_dir / "predictions.jsonl"

    with preds_csv_path.open("w", newline="", encoding="utf-8") as csv_f, \
         preds_jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        writer = csv.DictWriter(
            csv_f,
            fieldnames=[
                "image",
                "num_preds",
                "num_drawn",
                "gt_count",
                "scores",
                "boxes",
            ],
        )
        writer.writeheader()

        for i, img_path in enumerate(image_files, start=1):
            t_img0 = time.perf_counter()
            # 1) Leer RGBA
            img_pil = Image.open(img_path).convert("RGBA")
            img_np = np.array(img_pil, dtype=np.float32) / 255.0  # HWC
            h, w = img_np.shape[0], img_np.shape[1]

            # 2) A tensor BCHW + normalize (4ch)
            ten = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # 1,4,H,W
            ten = (ten - mean) / std
            pixel_values = ten.to(device)

            # 3) pixel_mask (True = valido)
            pixel_mask = torch.ones((1, h, w), dtype=torch.bool, device=device)

            # 4) Inferencia
            _safe_cuda_sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            _safe_cuda_sync(device)
            t1 = time.perf_counter()

            det = detr_postprocess(outputs, h=h, w=w, score_thresh=0.0)

            # Solo clase person=0
            keep = det["labels"] == 0
            boxes = det["boxes"][keep]
            scores = det["scores"][keep]

            preds_all.append({"boxes": boxes, "scores": scores})
            scores_for_hist.append(scores)

            # GT
            if ann_dir is not None:
                xml_path = ann_dir / f"{img_path.stem}.xml"
            else:
                xml_path = img_path.with_suffix(".xml")
            gt_boxes = parse_voc_xml(xml_path, class_name=class_name)
            if xml_path.exists():
                has_any_xml = True
            gts_all.append(gt_boxes)

            # Dibujo
            img_cv = cv2.cvtColor((img_np * 255.0).astype(np.uint8), cv2.COLOR_RGBA2BGR)
            draw_keep = scores >= threshold
            draw_boxes = boxes[draw_keep]
            draw_scores = scores[draw_keep]

            for box, sc in zip(draw_boxes, draw_scores):
                x1, y1, x2, y2 = box.numpy().astype(int)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_cv, f"Person {sc:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imwrite(str(out_dir / img_path.name), img_cv)

            record = {
                "image": img_path.name,
                "num_preds": int(boxes.shape[0]),
                "num_drawn": int(draw_boxes.shape[0]),
                "gt_count": int(gt_boxes.shape[0]),
                "scores": json.dumps([float(s) for s in scores.tolist()]),
                "boxes": json.dumps([[float(v) for v in b.tolist()] for b in boxes]),
            }
            writer.writerow(record)
            jsonl_f.write(json.dumps({
                "image": img_path.name,
                "scores": [float(s) for s in scores.tolist()],
                "boxes": [[float(v) for v in b.tolist()] for b in boxes],
                "threshold": threshold,
            }) + "\n")

            t_img1 = time.perf_counter()
            image_times_s.append(t_img1 - t_img0)
            infer_times_s.append(t1 - t0)

            logger.info("[%s/%s] %s -> det(draw)=%s", i, len(image_files), img_path.name, int(draw_boxes.shape[0]))

    logger.info("Saved annotated images to: %s", out_dir.resolve())

    if has_any_xml:
        prec, rec = precision_recall_at(preds_all, gts_all, conf_th=threshold, iou_th=0.5)
        ap50 = ap_at_iou(preds_all, gts_all, iou_th=0.5, min_score_for_map=min_score_for_map)
        iou_ths = [0.50 + 0.05 * i for i in range(10)]
        map5095 = sum(ap_at_iou(preds_all, gts_all, iou_th=t, min_score_for_map=min_score_for_map) for t in iou_ths) / len(iou_ths)

        logger.info("================= TEST METRICS =================")
        logger.info("Precision (conf=%s, IoU=0.5): %.3f", threshold, prec)
        logger.info("Recall    (conf=%s, IoU=0.5): %.3f", threshold, rec)
        logger.info("mAP@0.5: %.3f", ap50)
        logger.info("mAP@0.5:0.95: %.3f", map5095)
        logger.info("================================================")
        metrics = {
            "precision": float(prec),
            "recall": float(rec),
            "map50": float(ap50),
            "map5095": float(map5095),
            "conf_th": threshold,
        }
    else:
        logger.warning("No se encontraron .xml en test, asi que no puedo calcular metricas (solo guarde predicciones).")
        metrics = {
            "precision": None,
            "recall": None,
            "map50": None,
            "map5095": None,
            "conf_th": threshold,
        }

    save_json(run_dir / "metrics.json", metrics)
    (run_dir / "metrics.txt").write_text(
        "\n".join([
            "PREDICT METRICS",
            f"precision: {metrics['precision']}",
            f"recall: {metrics['recall']}",
            f"map50: {metrics['map50']}",
            f"map5095: {metrics['map5095']}",
            f"conf_th: {metrics['conf_th']}",
        ]),
        encoding="utf-8",
    )

    plot_score_histogram(scores_for_hist, run_dir)

    run_info = {
        "started_at": run_started_at,
        "ended_at": datetime.now().isoformat(),
        "num_images": len(image_files),
        "run_dir": str(run_dir),
        "images_dir": str(out_dir),
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
        "num_images": len(image_files),
        "image_time_avg_s": float(sum(image_times_s) / len(image_times_s)) if image_times_s else None,
        "image_time_p50_s": _percentile(image_times_s, 50) if image_times_s else None,
        "image_time_p95_s": _percentile(image_times_s, 95) if image_times_s else None,
        "infer_time_avg_s": float(sum(infer_times_s) / len(infer_times_s)) if infer_times_s else None,
        "infer_time_p50_s": _percentile(infer_times_s, 50) if infer_times_s else None,
        "infer_time_p95_s": _percentile(infer_times_s, 95) if infer_times_s else None,
        "fps_infer_avg": float(len(infer_times_s) / sum(infer_times_s)) if infer_times_s and sum(infer_times_s) > 0 else None,
        "gpu": _gpu_snapshot(),
    }
    save_json(run_dir / "perf.json", perf)
    logger.info("Results saved to: %s", run_dir.resolve())


if __name__ == "__main__":
    main()
