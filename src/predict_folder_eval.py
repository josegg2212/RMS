import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.ops import box_iou

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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


def get_model(num_classes: int = 2):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    _patch_first_conv_for_4ch(model)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


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
    ap /= 101.0
    return float(ap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="frcnn_personas.pth")
    parser.add_argument("--input_dir", type=str, default="dataset_personas/test")
    parser.add_argument("--output_dir", type=str, default="out_test")
    parser.add_argument("--threshold", type=float, default=0.25)  # para dibujar + PR
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--class_name", type=str, default="Person")
    parser.add_argument("--min_score_for_map", type=float, default=0.05)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA no disponible, usando CPU.")

    model = get_model(num_classes=2)
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])

    preds_all = []
    gts_all = []
    has_any_xml = False

    print(f"Images found: {len(image_files)}")
    for i, img_path in enumerate(image_files, start=1):
        img_pil = Image.open(img_path).convert("RGBA")
        img_tensor = TF.to_tensor(img_pil).to(device)

        with torch.no_grad():
            out = model([img_tensor])[0]

        # filtra Person (label==1)
        labels = out["labels"].detach().cpu()
        boxes = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()

        keep = labels == 1
        boxes = boxes[keep]
        scores = scores[keep]

        preds_all.append({"boxes": boxes, "scores": scores})

        # GT si existe xml
        xml_path = img_path.with_suffix(".xml")
        gt_boxes = parse_voc_xml(xml_path, class_name=args.class_name)
        if xml_path.exists():
            has_any_xml = True
        gts_all.append(gt_boxes)

        # Dibujar detecciones por threshold
        draw_keep = scores >= args.threshold
        draw_boxes = boxes[draw_keep]
        draw_scores = scores[draw_keep]

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        for box, sc in zip(draw_boxes, draw_scores):
            x1, y1, x2, y2 = box.numpy().astype(int)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"Person {sc:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img_cv)
        print(f"[{i}/{len(image_files)}] {img_path.name} -> det(draw)={int(draw_boxes.shape[0])}")

    print(f"\nSaved annotated images to: {out_dir.resolve()}")

    # Métricas finales (si hay XML)
    if has_any_xml:
        prec, rec = precision_recall_at(preds_all, gts_all, conf_th=args.threshold, iou_th=0.5)
        ap50 = ap_at_iou(preds_all, gts_all, iou_th=0.5, min_score_for_map=args.min_score_for_map)
        iou_ths = [0.50 + 0.05 * i for i in range(10)]
        map5095 = sum(ap_at_iou(preds_all, gts_all, iou_th=t, min_score_for_map=args.min_score_for_map) for t in iou_ths) / len(iou_ths)

        print("\n================= TEST METRICS =================")
        print(f"Precision (conf={args.threshold}, IoU=0.5): {prec:.3f}")
        print(f"Recall    (conf={args.threshold}, IoU=0.5): {rec:.3f}")
        print(f"mAP@0.5: {ap50:.3f}")
        print(f"mAP@0.5:0.95: {map5095:.3f}")
        print("================================================\n")
    else:
        print("\nNo se encontraron .xml en test, así que no puedo calcular métricas (solo guardé predicciones).")


if __name__ == "__main__":
    main()
