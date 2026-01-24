import time
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.ops import box_iou

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# -----------------------------
# Dataset VOC (images + .xml en la misma carpeta)
# -----------------------------
class VOCFolderDataset(Dataset):
    def __init__(self, folder: str, class_name: str = "Person"):
        self.folder = Path(folder)
        self.class_name = class_name.strip().lower()

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.images = sorted([p for p in self.folder.iterdir() if p.suffix.lower() in exts])

    def __len__(self):
        return len(self.images)

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
                labels.append(1)  # 1 = Person (0 = background)

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
        xml_path = img_path.with_suffix(".xml")

        img = Image.open(img_path).convert("RGBA")
        image = TF.to_tensor(img)

        boxes, labels = self._parse_xml(xml_path)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# Modelo
# -----------------------------
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

    # Reemplazamos cabeza a 2 clases: background + Person
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# -----------------------------
# Métricas (1 clase): Precision/Recall @ (conf, IoU=0.5) + mAP50 + mAP50-95
# Implementación simple estilo COCO (101-point interpolation)
# -----------------------------
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

        # ordenar por score (greedy)
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


def ap_at_iou(preds, gts, iou_th=0.5, min_score_for_map=0.05):
    # Flatten predicciones (score, img_id, box)
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

    # COCO: 101-point interpolation
    ap = 0.0
    for r in torch.linspace(0, 1, 101):
        mask = recall >= r
        p = torch.max(precision[mask]).item() if torch.any(mask) else 0.0
        ap += p
    ap /= 101.0
    return ap


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
            # filtra label==1 por seguridad (Person)
            labels = out["labels"].detach().cpu()
            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()

            keep = labels == 1
            boxes = boxes[keep]
            scores = scores[keep]

            preds.append({"boxes": boxes, "scores": scores})
            gts.append({"boxes": tgt["boxes"].detach().cpu()})

    # Precision/Recall @ conf_th, IoU=0.5
    prec, rec = precision_recall_at(preds, gts, conf_th=conf_th, iou_th=0.5)

    # mAP@0.5 y mAP@0.5:0.95
    ap50 = ap_at_iou(preds, gts, iou_th=0.5, min_score_for_map=min_score_for_map)

    iou_ths = [0.50 + 0.05 * i for i in range(10)]  # 0.50..0.95
    aps = [ap_at_iou(preds, gts, iou_th=t, min_score_for_map=min_score_for_map) for t in iou_ths]
    map5095 = float(sum(aps) / len(aps))

    return {
        "precision": float(prec),
        "recall": float(rec),
        "map50": float(ap50),
        "map5095": float(map5095),
    }


# -----------------------------
# Train loop + valid por época + resumen
# -----------------------------
def main():
    train_dir = "dataset_personas/train"
    valid_dir = "dataset_personas/valid"
    epochs = 10
    batch_size = 2
    lr = 0.005
    device = "cuda"
    num_workers = 2
    out = "frcnn_personas.pth"
    save_best = False
    conf_th = 0.25
    min_score_for_map = 0.05

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA no disponible, usando CPU.")

    train_ds = VOCFolderDataset(train_dir, class_name="Person")
    valid_ds = VOCFolderDataset(valid_dir, class_name="Person")

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )

    model = get_model(num_classes=2).to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    print(f"Train images: {len(train_ds)} | Valid images: {len(valid_ds)} | device: {device}")
    print(f"epochs={epochs} batch_size={batch_size} lr={lr} conf_th={conf_th}")

    history = []
    best_map50 = -1.0
    best_epoch = -1

    t0_total = time.time()

    for epoch in range(1, epochs + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        running_loss = 0.0

        for step, (images, targets) in enumerate(train_dl, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step == 1 or step % 20 == 0:
                loss_parts = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
                avg_loss = running_loss / step
                print(f"[Epoch {epoch}/{epochs}] step {step}/{len(train_dl)} "
                      f"avg_loss={avg_loss:.4f} parts={loss_parts}")

        avg_train_loss = running_loss / max(1, len(train_dl))

        # Validación
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

        print(
            f"== Epoch {epoch} done | time={epoch_time:.1f}s | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val: P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} "
            f"mAP50={val_metrics['map50']:.3f} mAP50-95={val_metrics['map5095']:.3f}"
        )

        # Guardar best
        if save_best and val_metrics["map50"] > best_map50:
            best_map50 = val_metrics["map50"]
            best_epoch = epoch
            torch.save(model.state_dict(), out)
            print(f"  -> NEW BEST (mAP50={best_map50:.3f}) saved to {out}")

    total_time = time.time() - t0_total

    # Si no save_best, guarda al final
    if not save_best:
        torch.save(model.state_dict(), out)

    # Resumen final
    best = max(history, key=lambda x: x["map50"]) if history else None
    print("\n================= TRAIN SUMMARY =================")
    print(f"Total time: {total_time/60:.2f} min | saved: {out}")
    if best is not None:
        print(f"Best epoch (by mAP50): {best['epoch']} | "
              f"P={best['precision']:.3f} R={best['recall']:.3f} "
              f"mAP50={best['map50']:.3f} mAP50-95={best['map5095']:.3f} "
              f"train_loss={best['train_loss']:.4f} time={best['epoch_time_s']:.1f}s")
    last = history[-1] if history else None
    if last is not None:
        print(f"Last epoch: {last['epoch']} | "
              f"P={last['precision']:.3f} R={last['recall']:.3f} "
              f"mAP50={last['map50']:.3f} mAP50-95={last['map5095']:.3f} "
              f"train_loss={last['train_loss']:.4f} time={last['epoch_time_s']:.1f}s")
    print("================================================\n")


if __name__ == "__main__":
    main()
