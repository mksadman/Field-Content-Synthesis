"""Task A's degradation pipeline, run over Task B's rendered cards
(synthetic/rendered/) instead of the real train cards.

Per the doc:
  - Section 4.4: "Run the same pipeline over the Task B rendered cards,
    2-3 variants per rendered card."
  - Section 5.5: "Rendered cards get fewer degraded copies so synthetic
    text does not outnumber real print."
  - Section 6: this output's method tag is ren-deg. Per your supervisor,
    it goes in its OWN folder -- synthetic/degraded_rendered/ -- not
    synthetic/degraded/, which stays real-card output only.

This imports the operations, ranges, conflict families, and Section 7
validation straight from generate_degraded.py instead of copy-pasting them,
so it's actually "the same pipeline" at the code level, not just
conceptually similar -- the two can never quietly drift apart. Only what's
genuinely different for rendered cards lives here: input/output paths, the
ren-deg naming (with layout carried through from Task B's own manifest,
since Task B already knows it and Task A's real-card side doesn't track
layout at all), and 2-3 variants per card instead of 5-10.

Does not touch generate_degraded.py, synthetic/degraded/, or anything
Task A has already produced from the real train cards.
"""
import cv2
import csv
import random
import shutil
from pathlib import Path

from generate_degraded import (
    SEED, make_picks, apply_chain, load_yolo_labels, save_yolo_labels,
    validate_labels,
)

RENDERED_IMAGES = Path("synthetic/rendered/images")
RENDERED_LABELS = Path("synthetic/rendered/labels")
RENDERED_MANIFEST = Path("synthetic/rendered/manifest.csv")

OUT_IMAGES = Path("synthetic/degraded_rendered/images")
OUT_LABELS = Path("synthetic/degraded_rendered/labels")
OUT_MANIFEST = Path("synthetic/degraded_rendered/manifest.csv")

VARIANTS_PER_CARD_RANGE = (2, 3)   # doc Section 4.4
MAX_ATTEMPTS_PER_VARIANT = 15


def load_source_lookup():
    """Task B's own manifest already records each rendered file's real
    source_id and layout -- read it from there instead of re-parsing the
    filename, since that's the authoritative source B computed it from."""
    lookup = {}
    with open(RENDERED_MANIFEST, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lookup[row["file"]] = (row["source_id"], row["layout"])
    return lookup


def generate_variants_for_card(image_path, label_path, source_id, layout):
    stem = image_path.stem
    img = cv2.imread(str(image_path))
    labels = load_yolo_labels(label_path)

    rows = []
    n_variants = random.randint(*VARIANTS_PER_CARD_RANGE)
    variant_idx = 1
    attempts = 0
    max_attempts = n_variants * MAX_ATTEMPTS_PER_VARIANT

    while variant_idx <= n_variants and attempts < max_attempts:
        attempts += 1
        op_names = make_picks(random.randint(2, 3))
        out_img, out_labels = apply_chain(op_names, img.copy(), labels)
        ok, reason = validate_labels(out_labels, out_img.shape) if len(out_labels) == 6 else (False, "clipped a box (fewer than 6 survived)")
        if not ok:
            print(f"  [{stem}] dropped {op_names}: {reason}")
            continue
        name = f"{source_id}_{layout}_ren-deg_{variant_idx:02d}"
        cv2.imwrite(str(OUT_IMAGES / f"{name}.jpg"), out_img)
        save_yolo_labels(OUT_LABELS / f"{name}.txt", out_labels)
        rows.append([name, source_id, layout, "+".join(op_names), SEED])
        variant_idx += 1

    if len(rows) < n_variants:
        print(f"  [{stem}] WARNING: only generated {len(rows)}/{n_variants} (some got dropped -- see reasons above)")

    return rows


if __name__ == "__main__":
    if OUT_IMAGES.exists():
        shutil.rmtree(OUT_IMAGES)
    if OUT_LABELS.exists():
        shutil.rmtree(OUT_LABELS)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    if OUT_MANIFEST.exists():
        OUT_MANIFEST.unlink()

    if not RENDERED_MANIFEST.exists():
        raise SystemExit(f"Can't find {RENDERED_MANIFEST} -- needed to look up each rendered card's source_id/layout.")
    lookup = load_source_lookup()

    all_rows = []
    images = sorted(RENDERED_IMAGES.glob("*.png")) + sorted(RENDERED_IMAGES.glob("*.jpg"))
    for img_path in images:
        label_path = RENDERED_LABELS / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"Skipping {img_path.name}: no label file")
            continue
        if img_path.name not in lookup:
            print(f"Skipping {img_path.name}: not found in {RENDERED_MANIFEST} (can't trace source_id/layout)")
            continue
        source_id, layout = lookup[img_path.name]
        try:
            labels = load_yolo_labels(label_path)
        except ValueError as e:
            print(f"Skipping {img_path.name}: malformed label file -- {e}")
            continue
        if len(labels) != 6:
            print(f"Skipping {img_path.name}: expected 6 label lines, found {len(labels)}")
            continue
        print(f"Generating variants for {img_path.name}...")
        try:
            all_rows.extend(generate_variants_for_card(img_path, label_path, source_id, layout))
        except Exception as e:
            print(f"Skipping {img_path.name}: failed while generating variants -- {e}")
            continue

    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["output_file", "source_card", "layout", "operations", "seed"])
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} variants written to {OUT_IMAGES} and {OUT_LABELS}.")
