import cv2
import random
import csv
import shutil
import numpy as np
import albumentations as A
from pathlib import Path
from photocopy_glare import (
    photocopy_effect, salt_pepper_noise, low_light_effect, white_balance_tint,
    soft_shadow, finger_phone_shadow, glossy_streak, hologram_patch,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = ["name_bn", "name_en", "guardian", "mother", "dob", "nid"]

TRAIN_IMAGES = Path("dataset/real/train/images")
TRAIN_LABELS = Path("dataset/real/train/labels")
OUT_IMAGES = Path("synthetic/degraded/images")
OUT_LABELS = Path("synthetic/degraded/labels")
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = Path("synthetic/degraded/manifest.csv")

SINGLE_OP_COUNT = 5     # variants with exactly 1 operation applied
CHAIN_VARIANT_COUNT = 5  # variants with a random 2-4 op combo applied
VARIANTS_PER_CARD = SINGLE_OP_COUNT + CHAIN_VARIANT_COUNT
# min_visibility=1.0 (exact 100.000...%) was dropping EVERY box on EVERY
# geometric op, even a safe 2-degree rotation nowhere near the card edge --
# because any rotation/perspective warp involves floating-point trig and
# pixel-grid rounding, so "how much of the box survived" almost never comes
# out to a perfectly clean 1.0, even when nothing visible was actually cut
# off. 0.98 still rejects boxes that are genuinely, meaningfully clipped
# (the thing Rule 5 actually cares about), while letting boxes through that
# are for all practical purposes fully intact.
BBOX_PARAMS = A.BboxParams(format="yolo", label_fields=["cls"], min_visibility=0.98)

# No moderate/heavy tiers anymore -- each op has ONE range, and Albumentations
# (or our own random.uniform/randint calls inside photocopy_glare.py) samples
# a fresh value from it every time the op runs. So strength still varies
# naturally per image, it's just not sorted into two named levels, and the
# level is no longer written into filenames or the manifest.
# These ranges are still placeholders -- tune them directly with your
# supervisor using preview_operations.py.
ALBUMENTATIONS_OPS = {
    # These 4 ranges got tested directly against the actual sample card
    # (515x325px) with plain cv2 blur/downscale at each strength, because the
    # card images are small enough that "moderate-sounding" numbers were
    # actually wiping out the Bengali text entirely. Narrowed to whatever
    # stayed legible in that test:
    # Tightened again after checking a zoomed crop of the BENGALI text
    # specifically (not just the English line) -- Bengali glyphs are finer
    # and blur out faster than Latin letters at the same kernel size, so the
    # earlier "readable" ranges were actually still wrecking the Bengali
    # name/field values while looking fine on the English ones.
    "blur_gaussian": lambda: A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    "blur_motion": lambda: A.MotionBlur(blur_limit=(3, 5), p=1.0),
    "blur_defocus": lambda: A.Defocus(radius=(2, 3), alias_blur=(0.1, 0.2), p=1.0),
    "resolution_downscale": lambda: A.Downscale(scale_range=(0.45, 0.75), p=1.0),
    "compression_jpeg": lambda: A.ImageCompression(quality_range=(15, 60), p=1.0),
    "compression_jpeg_twice": lambda: A.Compose([
        A.ImageCompression(quality_range=(15, 60)),
        A.ImageCompression(quality_range=(15, 60)),
    ]),
    # std_range is a FRACTION of 255, not raw pixel units -- tested directly:
    # 0.02 was nearly invisible at normal viewing size, only clearly visible
    # from ~0.05 up. Floor raised so a single draw is never a no-op-looking
    # result.
    "noise_gaussian": lambda: A.GaussNoise(std_range=(0.05, 0.15), p=1.0),
    "noise_iso": lambda: A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
    "lighting_brightness_contrast": lambda: A.RandomBrightnessContrast(0.4, 0.4, p=1.0),
    "lighting_gamma": lambda: A.RandomGamma(gamma_limit=(60, 150), p=1.0),
    "lighting_overexposure": lambda: A.RandomBrightnessContrast(brightness_limit=(0.2, 0.5), contrast_limit=(-0.3, -0.1), p=1.0),
    "lighting_underexposure": lambda: A.RandomBrightnessContrast(brightness_limit=(-0.35, -0.1), contrast_limit=(-0.15, -0.05), p=1.0),
    # deg_06 on nid_0004: src_radius up to 140 with the default 6-10
    # overlapping flare circles was blanketing a third of the card in
    # near-opaque white blobs (the pastel pink/green look is just that white
    # overlay showing the skin/background color through it) -- it swallowed
    # the whole face photo and part of the name row, not just "some" text.
    # Circles are drawn stacked/overlapping, so opacity builds up with both
    # radius AND count -- shrunk both so a draw is still visibly a sun flare
    # but can no longer cover that much of the card at once.
    "glare_sun_flare": lambda: A.RandomSunFlare(
        src_radius=random.randint(35, 65), num_flare_circles_range=(2, 4), p=1.0,
    ),
    "colour_hue_saturation": lambda: A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=35, p=1.0),
}

GEOMETRIC_OPS = {
    "geometry_rotation": lambda: A.Affine(rotate=(-4, 4), p=1.0),
    "geometry_scale_translate": lambda: A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=1.0),
    "geometry_perspective": lambda: A.Perspective(scale=(0.01, 0.03), p=1.0),
}

CUSTOM_OPS = {
    "noise_salt_pepper": lambda img: salt_pepper_noise(img),
    "lighting_low_light": lambda img: low_light_effect(img),
    "colour_white_balance_warm": lambda img: white_balance_tint(img, warm=True),
    "colour_white_balance_cool": lambda img: white_balance_tint(img, warm=False),
    "shadow_soft_polygon": lambda img: soft_shadow(img),
    "shadow_finger": lambda img: finger_phone_shadow(img, shape="finger"),
    "shadow_phone": lambda img: finger_phone_shadow(img, shape="phone"),
    "fading_photocopy": lambda img: photocopy_effect(img),
    "glare_glossy_streak": lambda img: glossy_streak(img),
    "glare_hologram": lambda img: hologram_patch(img),
}

ALL_OP_NAMES = list(ALBUMENTATIONS_OPS) + list(GEOMETRIC_OPS) + list(CUSTOM_OPS)

# Some ops compound DESTRUCTIVELY when chained together -- e.g. two different
# blur types stacked doesn't look "a bit blurrier," it wipes out every field
# completely; three different darkening effects stacked turns a whole region
# solid black. That's a genuinely useless training image (no information
# left to learn from), not just "a very bad photo." So a combined-variant
# chain is only allowed to pick AT MOST ONE operation from each family below
# -- still lets multiple different KINDS of degradation stack (e.g. one blur
# + one darkening + one noise + one glare is fine), just not two of the same
# destructive kind piled on top of each other.
CONFLICT_FAMILIES = [
    {"blur_gaussian", "blur_motion", "blur_defocus", "resolution_downscale"},
    {"lighting_underexposure", "lighting_low_light", "shadow_soft_polygon", "shadow_finger", "shadow_phone"},
    # Same reasoning: two bright/white-blob glare effects stacked (e.g. sun
    # flare + glossy streak) would double up the exact same "big pale blob
    # covers everything" failure mode as deg_06, just from a different pair
    # of ops. Cap combined chains to one glare effect.
    {"glare_sun_flare", "glare_glossy_streak", "glare_hologram"},
]


def make_picks(n_ops):
    """Random ops for a chain, but never more than one per CONFLICT_FAMILIES
    group -- avoids the "two blurs" / "three darkenings" stacking problem."""
    pool = ALL_OP_NAMES.copy()
    random.shuffle(pool)
    chosen = []
    used_families = set()
    for op in pool:
        if len(chosen) >= n_ops:
            break
        fam_idx = next((i for i, fam in enumerate(CONFLICT_FAMILIES) if op in fam), None)
        if fam_idx is not None and fam_idx in used_families:
            continue
        chosen.append(op)
        if fam_idx is not None:
            used_families.add(fam_idx)
    return chosen


def load_yolo_labels(txt_path):
    """A handful of real label files have a box drawn a hair past the card's
    edge (seen so far: the 'nid' field's y_max landing at 1.002-1.011
    instead of <=1.0 -- an annotation-tool slip, not something this pipeline
    did). Albumentations refuses to run ANY geometric op on an out-of-[0,1]
    bbox and raises immediately, which was killing the ENTIRE card (all 10
    variants, including the blur/noise/lighting ones that never touch
    bboxes at all) over one bad field. Clamping to the image edges here
    fixes it at the source for every op, not just geometric ones, and only
    touches boxes that were already invalid -- a well-formed box round-trips
    through this unchanged."""
    labels = []
    with open(txt_path) as f:
        for line in f:
            cls, xc, yc, w, h = line.split()
            cls, xc, yc, w, h = int(cls), float(xc), float(yc), float(w), float(h)
            x0, y0 = xc - w / 2, yc - h / 2
            x1, y1 = xc + w / 2, yc + h / 2
            cx0, cy0 = min(max(x0, 0.0), 1.0), min(max(y0, 0.0), 1.0)
            cx1, cy1 = min(max(x1, 0.0), 1.0), min(max(y1, 0.0), 1.0)
            if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
                print(f"  [{txt_path.stem}] clamped out-of-bounds box (class {cls}) to image edges")
            labels.append((cls, (cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0))
    return labels


def save_yolo_labels(txt_path, labels):
    with open(txt_path, "w") as f:
        for c, xc, yc, w, h in labels:
            f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# Section 7 ("Quality check before hand-over") script-check thresholds --
# enforced HERE at generation time, not just checked afterward by
# qa_check.py. A variant failing any of these never gets saved in the first
# place, so a fresh run is compliant by construction instead of needing a
# separate fix-and-regenerate pass. Same numbers qa_check.py uses, so the
# two stay in agreement.
TINY_AREA_FRACTION = 0.01   # 1% of image area
TINY_HEIGHT_PX = 10
EDGE_EPS = 1e-4             # within this of 0.0/1.0 counts as "touching" the edge
IOU_THRESHOLD = 0.3


def _to_corners(box):
    cls, xc, yc, w, h = box
    return cls, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2


def _iou(box_a, box_b):
    _, ax0, ay0, ax1, ay1 = _to_corners(box_a)
    _, bx0, by0, bx1, by1 = _to_corners(box_b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def validate_labels(labels, img_shape):
    """All 4 of Section 7's per-image 'Script check' items that can be
    checked here (the 5th, val/test leakage, is guaranteed structurally --
    this function only ever sees labels loaded from TRAIN_LABELS). Returns
    (True, None) if the variant is clean, else (False, reason) so the
    caller can log exactly which rule killed it -- same detail level as
    qa_check.py's report."""
    h, w = img_shape[:2]

    classes = sorted(c for c, *_ in labels)
    if classes != list(range(6)):
        return False, f"expected 6 boxes (classes 0-5), got {classes}"

    for box in labels:
        cls, x0, y0, x1, y1 = _to_corners(box)
        if x0 <= EDGE_EPS or y0 <= EDGE_EPS or x1 >= 1 - EDGE_EPS or y1 >= 1 - EDGE_EPS:
            return False, f"{CLASS_NAMES[cls]} box touches/exceeds image edge"
        box_h_px = (y1 - y0) * h
        area_frac = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area_frac < TINY_AREA_FRACTION or box_h_px < TINY_HEIGHT_PX:
            return False, f"{CLASS_NAMES[cls]} box too tiny ({area_frac * 100:.2f}% area, {box_h_px:.1f}px tall)"

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            v = _iou(labels[i], labels[j])
            if v > IOU_THRESHOLD:
                return False, f"{CLASS_NAMES[labels[i][0]]}/{CLASS_NAMES[labels[j][0]]} overlap IoU={v:.2f}"

    return True, None


def apply_single_op(op_name, img, labels):
    if op_name in ALBUMENTATIONS_OPS:
        return ALBUMENTATIONS_OPS[op_name]()(image=img)["image"], labels
    if op_name in GEOMETRIC_OPS:
        bboxes = [(xc, yc, w, h) for _, xc, yc, w, h in labels]
        classes = [c for c, *_ in labels]
        compose = A.Compose([GEOMETRIC_OPS[op_name]()], bbox_params=BBOX_PARAMS)
        result = compose(image=img, bboxes=bboxes, cls=classes)
        # Albumentations hands class ids back as floats after a geometric
        # transform -- cast back to int or the saved YOLO label lines end up
        # malformed (e.g. "0.0 0.44..." instead of "0 0.44...").
        return result["image"], [(int(c), *b) for c, b in zip(result["cls"], result["bboxes"])]
    return CUSTOM_OPS[op_name](img), labels


def apply_chain(op_names, img, labels):
    for op_name in op_names:
        img, labels = apply_single_op(op_name, img, labels)
        if len(labels) < 6:
            return img, labels
    return img, labels


def generate_variants(image_path, label_path):
    source_id = image_path.stem
    img = cv2.imread(str(image_path))
    labels = load_yolo_labels(label_path)

    rows = []
    variant_idx = 1
    attempts = 0
    max_attempts = VARIANTS_PER_CARD * 10

    # Phase 1: single-op variants. Sampled WITHOUT replacement, so the
    # SINGLE_OP_COUNT slots are guaranteed to be different operations --
    # this is what fixes "one op might never get picked": every card now
    # deliberately shows a distinct spread of ops instead of leaving it to
    # chance which ones show up.
    single_op_pool = random.sample(ALL_OP_NAMES, min(SINGLE_OP_COUNT, len(ALL_OP_NAMES)))
    for op_name in single_op_pool:
        attempts += 1
        out_img, out_labels = apply_chain([op_name], img.copy(), labels)
        ok, reason = validate_labels(out_labels, out_img.shape) if len(out_labels) == 6 else (False, "clipped a box (fewer than 6 survived)")
        if not ok:
            print(f"  [{source_id}] dropped [{op_name}]: {reason}")
            continue
        name = f"{source_id}_deg_{variant_idx:02d}"
        cv2.imwrite(str(OUT_IMAGES / f"{name}.jpg"), out_img)
        save_yolo_labels(OUT_LABELS / f"{name}.txt", out_labels)
        rows.append([name, source_id, op_name, SEED])
        variant_idx += 1

    # Phase 2: combined variants -- a random 2-4 op chain, still randomly
    # picked (repeats across chains are fine/expected here).
    n_chain_done = 0
    while n_chain_done < CHAIN_VARIANT_COUNT and attempts < max_attempts:
        attempts += 1
        op_names = make_picks(random.randint(2, 3))
        out_img, out_labels = apply_chain(op_names, img.copy(), labels)
        ok, reason = validate_labels(out_labels, out_img.shape) if len(out_labels) == 6 else (False, "clipped a box (fewer than 6 survived)")
        if not ok:
            print(f"  [{source_id}] dropped {op_names}: {reason}")
            continue
        name = f"{source_id}_deg_{variant_idx:02d}"
        cv2.imwrite(str(OUT_IMAGES / f"{name}.jpg"), out_img)
        save_yolo_labels(OUT_LABELS / f"{name}.txt", out_labels)
        rows.append([name, source_id, "+".join(op_names), SEED])
        variant_idx += 1
        n_chain_done += 1

    if len(rows) < VARIANTS_PER_CARD:
        print(f"  [{source_id}] WARNING: only generated {len(rows)}/{VARIANTS_PER_CARD} (some got dropped -- see reasons above)")

    return rows


if __name__ == "__main__":
    # Wipe previous run's output first so reruns while tuning values don't
    # leave stale images/rows lying around next to the new ones.
    if OUT_IMAGES.exists():
        shutil.rmtree(OUT_IMAGES)
    if OUT_LABELS.exists():
        shutil.rmtree(OUT_LABELS)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    all_rows = []
    images = sorted(list(TRAIN_IMAGES.glob("*.png")) + list(TRAIN_IMAGES.glob("*.jpg")))
    for img_path in images:
        label_path = TRAIN_LABELS / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"Skipping {img_path.name}: no label file")
            continue
        try:
            labels = load_yolo_labels(label_path)
        except ValueError as e:
            print(f"Skipping {img_path.name}: malformed label file ({label_path.name}) -- {e}")
            continue
        if len(labels) != 6:
            print(f"Skipping {img_path.name}: expected 6 label lines, found {len(labels)} in {label_path.name}")
            continue
        print(f"Generating variants for {img_path.name}...")
        try:
            all_rows.extend(generate_variants(img_path, label_path))
        except Exception as e:
            print(f"Skipping {img_path.name}: failed while generating variants -- {e}")
            continue

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["output_file", "source_card", "operations", "seed"])
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} variants written to {OUT_IMAGES} and {OUT_LABELS}.")
