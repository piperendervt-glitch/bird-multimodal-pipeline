"""Phase 4a 段階A: YOLOv8 で CUB-200 画像から鳥を検出する。

- 検出結果と CUB-200 同梱の正解 bbox で IoU を計算
- 結果を JSON に保存（段階B の切り抜きで使用）
- DINOv2 や MLP はインポートしない
"""

import json
import os
import time

from ultralytics import YOLO


PHASE1_DIR = os.path.join("..", "results", "bird_phase1")
PHASE4A_DIR = os.path.join("..", "results", "bird_phase4a")
CUB_DIR = os.path.join("..", "data", "cub200", "CUB_200_2011")

# COCO の bird クラス ID = 14
BIRD_CLASS_ID = 14


def load_cub_bboxes(cub_dir):
    """CUB-200 の正解 bbox を読み込む。img_id → {x,y,w,h}"""
    bboxes = {}
    with open(os.path.join(cub_dir, "bounding_boxes.txt")) as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            x, y, w, h = (float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4]))
            bboxes[img_id] = {"x": x, "y": y, "w": w, "h": h}
    return bboxes


def compute_iou(box1, box2):
    """IoU を計算。box は (x1, y1, x2, y2) 形式"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-9)


def main():
    print("=== Phase 4a-A: YOLO 鳥検出 ===")

    # メタデータ
    with open(os.path.join(PHASE1_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    # 画像パス
    images = {}
    with open(os.path.join(CUB_DIR, "images.txt")) as f:
        for line in f:
            img_id, path = line.strip().split()
            images[int(img_id)] = os.path.join(CUB_DIR, "images", path)

    # 正解 bbox
    gt_bboxes = load_cub_bboxes(CUB_DIR)

    # YOLO 読み込み
    print("\nYOLOv8n を読み込み中...")
    model = YOLO("yolov8n.pt")
    print(f"COCO bird クラス ID: {BIRD_CLASS_ID}")

    all_ids = meta["train_ids"] + meta["test_ids"]
    print(f"\n全 {len(all_ids)} 枚を処理中...")

    results_list = []
    detected_count = 0
    total_iou = 0.0
    iou_count = 0

    start = time.time()
    for i, img_id in enumerate(all_ids):
        img_path = images[img_id]

        # YOLO 推論（conf=0.1 で鳥を広めに検出）
        result = model(img_path, verbose=False, conf=0.1)[0]

        bird_detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != BIRD_CLASS_ID:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bird_detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": conf,
            })

        gt = gt_bboxes.get(img_id)
        gt_box = None
        if gt:
            gt_box = (gt["x"], gt["y"], gt["x"] + gt["w"], gt["y"] + gt["h"])

        best_iou = 0.0
        best_detection = None
        if bird_detections and gt_box:
            for det in bird_detections:
                det_box = (det["x1"], det["y1"], det["x2"], det["y2"])
                iou = compute_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_detection = det
        elif bird_detections:
            # gt_box が無い場合は最高信頼度の検出を採用
            best_detection = max(bird_detections, key=lambda d: d["confidence"])

        detected = len(bird_detections) > 0
        if detected:
            detected_count += 1
        if best_iou > 0.0:
            total_iou += best_iou
            iou_count += 1

        results_list.append({
            "img_id": img_id,
            "detected": detected,
            "n_detections": len(bird_detections),
            "best_iou": best_iou,
            "best_detection": best_detection,
            "gt_bbox": gt if gt else None,
        })

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(all_ids)} ({rate:.1f} 枚/秒) "
                  f"検出率: {detected_count/(i+1)*100:.1f}%")

    elapsed = time.time() - start

    detection_rate = detected_count / len(all_ids)
    mean_iou = total_iou / (iou_count + 1e-9)
    iou_gt_05 = sum(1 for r in results_list if r["best_iou"] > 0.5)
    iou_gt_07 = sum(1 for r in results_list if r["best_iou"] > 0.7)

    print(f"\n=== 検出結果サマリー ===")
    print(f"処理枚数: {len(all_ids)}")
    print(f"処理時間: {elapsed:.1f} 秒 ({len(all_ids)/elapsed:.2f} 枚/秒)")
    print(f"鳥検出率: {detected_count}/{len(all_ids)} = {detection_rate*100:.2f}%")
    print(f"平均 IoU (検出できた画像のみ): {mean_iou:.4f}")
    print(f"IoU > 0.5 の割合: {iou_gt_05}/{len(all_ids)} "
          f"({iou_gt_05/len(all_ids)*100:.2f}%)")
    print(f"IoU > 0.7 の割合: {iou_gt_07}/{len(all_ids)} "
          f"({iou_gt_07/len(all_ids)*100:.2f}%)")

    not_detected = [r for r in results_list if not r["detected"]]
    print(f"\n未検出画像: {len(not_detected)} 枚")
    if not_detected:
        print("  例（先頭 5 枚）:")
        for r in not_detected[:5]:
            print(f"    img_id={r['img_id']}: {images[r['img_id']]}")

    os.makedirs(PHASE4A_DIR, exist_ok=True)
    out_path = os.path.join(PHASE4A_DIR, "detection_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "detection_rate": detection_rate,
            "mean_iou": mean_iou,
            "total_images": len(all_ids),
            "detected_count": detected_count,
            "iou_gt_05": iou_gt_05,
            "iou_gt_07": iou_gt_07,
            "inference_time_sec": elapsed,
            "images_per_sec": len(all_ids) / elapsed,
            "results": results_list,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")

    print(f"\n--- 失敗条件の判定 ---")
    if detection_rate < 0.80:
        print(f"  NG: 検出率 {detection_rate*100:.2f}% < 80%")
    else:
        print(f"  OK: 検出率 {detection_rate*100:.2f}% >= 80%")


if __name__ == "__main__":
    main()
