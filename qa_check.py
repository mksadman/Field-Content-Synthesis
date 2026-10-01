"""QA checklist for synthetic/degraded/, per Section 7 of the task doc
('Quality check before hand-over'). Covers every item marked 'Script check'
there:
  - every image has exactly 6 boxes, one per class
  - no box touches or goes past the image edge
  - no box is tiny (< ~1% of image area or < ~10px tall)
  - no two boxes overlap heavily (IoU > 0.3)
  - no source card from val/test appears in any synthetic file name

The other Section 7 items (visual check, OCR check, balance check, privacy
check) aren't things a script can fully verify on its own -- this script
does what it can toward them (writes 30 random box-overlay samples for the
visual check into preview/qa_overlays/) and says plainly what still needs a
human at the end.

Note on layout/level: this pipeline doesn't tag card layout (old/smart) or
degradation level (moderate/heavy) anywhere, per an explicit supervisor
call to skip that. So the doc's "balance check: count per layout and per
level" genuinely can't be run here -- that's not a gap in this script, it's
a consequence of that earlier decision.

Doesn't touch generate_degraded.py or its output -- read-only checks plus a
report and preview images.
"""
import cv2
import csv
import random
from collections import Counter
from pathlib import Path

CLASS_NAMES = ["name_bn", "name_en", "guardian", "mother", "dob", "nid"]

DEG_IMAGES = Path("synthetic/degraded/images")
DEG_LABELS = Path("synthetic/degraded/labels")
MANIFEST_PATH = Path("synthetic/degraded/manifest.csv")

# These don't exist yet (only dataset/real/train/ has been built so far) --
# the leakage check below just skips itself gracefully until they do.
VAL_IMAGES = Path("dataset/real/val/images")
TEST_IMAGES = Path("dataset/real/test/images")

PREVIEW_DIR = Path("preview/qa_overlays")
REPORT_PATH = Path("synthetic/degraded/qa_report.csv")

TINY_AREA_FRACTION = 0.01   # 1% of image area
TINY_HEIGHT_PX = 10
EDGE_EPS = 1e-4             # within this of 0.0/1.0 counts as "touching" the edge
IOU_THRESHOLD = 0.3
N_PREVIEW_SAMPLES = 30


def load_yolo_labels(txt_path):
    labels = []
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            labels.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return labels


def to_corners(box):
    cls, xc, yc, w, h = box
    return cls, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2


def iou(box_a, box_b):
    _, ax0, ay0, ax1, ay1 = to_corners(box_a)
    _, bx0, by0, bx1, by1 = to_corners(box_b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def class_name(cls):
    return CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else f"class{cls}"


def check_image(name, img_path, label_path, violations):
    img = cv2.imread(str(img_path))
    if img is None:
        violations.append((name, "unreadable_image", "cv2 could not open the file"))
        return
    h, w = img.shape[:2]
    labels = load_yolo_labels(label_path)

    classes = sorted(c for c, *_ in labels)
    if classes != list(range(6)):
        violations.append((name, "box_count", f"expected classes 0-5 exactly once, got {classes}"))

    for box in labels:
        cls, x0, y0, x1, y1 = to_corners(box)
        cname = class_name(cls)

        if x0 <= EDGE_EPS or y0 <= EDGE_EPS or x1 >= 1 - EDGE_EPS or y1 >= 1 - EDGE_EPS:
            violations.append((name, "touches_edge", f"{cname} box touches/exceeds image edge"))

        box_h_px = (y1 - y0) * h
        area_frac = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area_frac < TINY_AREA_FRACTION or box_h_px < TINY_HEIGHT_PX:
            violations.append((name, "tiny_box", f"{cname} box is {area_frac * 100:.2f}% of image area, {box_h_px:.1f}px tall"))

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            v = iou(labels[i], labels[j])
            if v > IOU_THRESHOLD:
                cni, cnj = class_name(labels[i][0]), class_name(labels[j][0])
                violations.append((name, "box_overlap", f"{cni} and {cnj} overlap with IoU={v:.2f}"))


def check_val_test_leakage(violations):
    val_test_ids = set()
    for split_dir in (VAL_IMAGES, TEST_IMAGES):
        if split_dir.exists():
            for p in split_dir.glob("*"):
                val_test_ids.add(p.stem)
    if not val_test_ids:
        print("  (no dataset/real/val or dataset/real/test folder yet -- skipping leakage check)")
        return
    if not MANIFEST_PATH.exists():
        print("  manifest.csv not found -- can't check leakage")
        return
    with open(MANIFEST_PATH) as f:
        for row in csv.DictReader(f):
            if row["source_card"] in val_test_ids:
                violations.append((row["output_file"], "val_test_leakage", f"source card '{row['source_card']}' belongs to val/test"))


def draw_boxes(img, labels):
    vis = img.copy()
    h, w = vis.shape[:2]
    for cls, xc, yc, bw, bh in labels:
        x0, y0 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x1, y1 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(vis, class_name(cls), (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return vis


def save_preview_sample(image_paths):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sample = random.sample(image_paths, min(N_PREVIEW_SAMPLES, len(image_paths)))
    written = 0
    for img_path in sample:
        label_path = DEG_LABELS / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        vis = draw_boxes(img, load_yolo_labels(label_path))
        cv2.imwrite(str(PREVIEW_DIR / img_path.name), vis)
        written += 1
    print(f"  wrote {written} box-overlay samples to {PREVIEW_DIR}/ for the visual check "
          f"(open these and confirm each box covers the full value, not the label)")


if __name__ == "__main__":
    image_paths = sorted(DEG_IMAGES.glob("*.jpg"))
    print(f"Checking {len(image_paths)} images in {DEG_IMAGES}...")

    violations = []
    for img_path in image_paths:
        label_path = DEG_LABELS / f"{img_path.stem}.txt"
        if not label_path.exists():
            violations.append((img_path.name, "missing_label", "no matching .txt label file"))
            continue
        check_image(img_path.name, img_path, label_path, violations)

    check_val_test_leakage(violations)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "check", "detail"])
        writer.writerows(violations)

    counts = Counter(v[1] for v in violations)
    n_files_flagged = len(set(v[0] for v in violations))
    print(f"\n{len(violations)} total violations across {n_files_flagged} files:")
    for check, n in counts.most_common():
        print(f"  {check}: {n}")
    print(f"Full detail written to {REPORT_PATH}")

    save_preview_sample(image_paths)

    print(
        "\nStill needs a human, per Section 7: open the preview/qa_overlays samples "
        "(visual check -- every box covers the full value, no label), run your existing "
        "OCR on ~50 random crops (OCR check), and confirm no real data left the project "
        "drive (privacy check). The doc's 'balance check: count per layout and per level' "
        "can't be run -- this pipeline doesn't tag layout or level anywhere, per your "
        "supervisor's call to skip that."
    )
