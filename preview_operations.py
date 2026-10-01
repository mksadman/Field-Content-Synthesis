import cv2
import albumentations as A
import matplotlib.pyplot as plt
from pathlib import Path

# Pulls the op definitions straight from generate_degraded.py -- only one
# place values live. Edit a range there or in photocopy_glare.py and this
# preview picks it up automatically.
from generate_degraded import (
    ALBUMENTATIONS_OPS, GEOMETRIC_OPS, CUSTOM_OPS, BBOX_PARAMS, load_yolo_labels,
)
from photocopy_glare import framing_margin, framing_overcrop

CLASS_NAMES = ["name_bn", "name_en", "guardian", "mother", "dob", "nid"]
TRAIN_IMAGES = Path("dataset/real/train/images")
TRAIN_LABELS = Path("dataset/real/train/labels")
OUT_DIR = Path("preview/operation_grids")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# framing has no severity split (and isn't used by generate_degraded.py yet)
CUSTOM_FRAMING_OPS = {
    "framing_extra_margin": lambda img, labels: framing_margin(img, labels),
    "framing_overcrop": lambda img, labels: framing_overcrop(img, labels),
}


def draw_boxes(img, labels):
    h, w = img.shape[:2]
    vis = img.copy()
    for cls, xc, yc, bw, bh in labels:
        x0, y0 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x1, y1 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(vis, CLASS_NAMES[cls], (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return vis


def apply_photometric(img, op_name):
    transform = ALBUMENTATIONS_OPS[op_name]()
    ops = transform.transforms if isinstance(transform, A.Compose) else [transform]
    return A.Compose(ops)(image=img)["image"]


def apply_geometric(img, labels, op_name):
    bboxes = [(xc, yc, w, h) for _, xc, yc, w, h in labels]
    classes = [c for c, *_ in labels]
    compose = A.Compose([GEOMETRIC_OPS[op_name]()], bbox_params=BBOX_PARAMS)
    result = compose(image=img, bboxes=bboxes, cls=classes)
    # Albumentations hands class ids back as floats after a geometric
    # transform -- cast back to int or CLASS_NAMES[cls] / YOLO label writing
    # breaks.
    return result["image"], [(int(c), *b) for c, b in zip(result["cls"], result["bboxes"])]


def make_grid(image_path, label_path):
    """One random draw per operation. Since every op is now a range instead
    of a fixed moderate/heavy value, rerunning this script will show a
    different (but still valid) sample each time -- rerun a few times if you
    want to see how much a value actually swings."""
    img = cv2.imread(str(image_path))
    labels = load_yolo_labels(label_path)
    panels = [("original", img, labels)]

    for name in ALBUMENTATIONS_OPS:
        panels.append((name, apply_photometric(img, name), labels))

    for name in GEOMETRIC_OPS:
        try:
            out_img, out_labels = apply_geometric(img, labels, name)
            tag = name if len(out_labels) == 6 else f"{name} (DROPPED a box!)"
            panels.append((tag, out_img, out_labels))
        except Exception as e:
            print(f"  {name} failed: {e}")

    for name in CUSTOM_OPS:
        panels.append((name, CUSTOM_OPS[name](img), labels))

    for name, func in CUSTOM_FRAMING_OPS.items():
        out_img, out_labels = func(img, labels)
        tag = name if len(out_labels) == 6 else f"{name} (DROPPED a box!)"
        panels.append((tag, out_img, out_labels))

    cols = 4
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    for ax, (name, out_img, out_labels) in zip(axes, panels):
        vis = draw_boxes(out_img, out_labels)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / f"{image_path.stem}_grid.png"
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(TRAIN_IMAGES.glob(ext))
    for img_path in sorted(images):
        label_path = TRAIN_LABELS / f"{img_path.stem}.txt"
        if label_path.exists():
            make_grid(img_path, label_path)
