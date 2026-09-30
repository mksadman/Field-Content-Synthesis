#!/usr/bin/env python3
"""
synthesize_batch.py -- Task B (field-content synthesis) for every card in images/.

For each source card this runs the same steps as synthesize_one.py, whose
functions it imports (blank template, font matching, fake values, rendering,
labelling, checks):

  1. Skip (and log) the card if its label file is missing, does not hold
     exactly 6 boxes (one per class), or its folder gives no known card type.
  2. Build the blank template and its measurements.
  3. Make `samples_per_card` synthetic cards with fresh made-up values;
     `hard_per_card` of them use the longest name that still fits (or a
     wrapped name, if the card type allows wrapping).
  4. Check every sample; a failing sample is retried with new values up to
     `max_retries` times, then dropped and logged.
  5. An error on one card is logged and the batch moves on.

Card type comes from the image's folder (images/images_old -> old,
images/images_smart -> smart). Seeds are derived from the global seed and the
source id, so a re-run gives identical output and one card can be regenerated
alone. Cards whose outputs all exist are skipped unless --overwrite is given.

Usage
  python synthesize_batch.py --config config.yaml [--limit N] [--overwrite]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

import synthesize_one as s1
from synthesize_one import CLASS_NAMES, SynthError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MANIFEST_FIELDS = (["file", "label_file", "source_id", "layout", "seed", "card_seed", "sample_seed",
                    "slot", "retries", "hard_case", "wrapped"]
                   + [f"{n}_{k}" for n in CLASS_NAMES
                      for k in ("value", "font", "size", "dx", "dy", "darkness", "blur")])
SKIPPED_FIELDS = ["source_id", "layout", "file", "kind", "reason"]


# =============================================================================
# Discovery and seeds
# =============================================================================
def discover_cards(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Find source images -> (cards, skipped).

    Cards are ordered round-robin across card types (old, smart, old, ...) so
    that --limit N gives a mix of both. Images outside a known card-type
    folder are returned as skipped.
    """
    b = cfg["batch"]
    root = s1.cfg_path(cfg, b["images_dir"])
    if not root.is_dir():
        raise SynthError(f"Images folder not found: {root}")
    per_layout, skipped = {}, []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        folder = p.parent.name if p.parent.parent == root else None
        layout = b["layout_folders"].get(folder) if folder else None
        if layout is None:
            where = p.parent.relative_to(root.parent)
            skipped.append({"source_id": p.stem, "layout": "", "file": p.name, "kind": "card_skipped",
                            "reason": f"image outside a card-type folder ({where})"})
        elif layout not in cfg["layouts"]:
            skipped.append({"source_id": p.stem, "layout": layout, "file": p.name, "kind": "card_skipped",
                            "reason": f"unknown card type '{layout}' (no block under layouts:)"})
        else:
            per_layout.setdefault(layout, []).append({"path": p, "source_id": p.stem, "layout": layout})
    cards = [c for group in zip_longest(*per_layout.values()) for c in group if c is not None]

    ids = Counter(c["source_id"] for c in cards)
    dupes = {i for i, n in ids.items() if n > 1}
    if dupes:  # one label folder serves all images, so a repeated name is ambiguous
        for c in [c for c in cards if c["source_id"] in dupes]:
            skipped.append({"source_id": c["source_id"], "layout": c["layout"], "file": c["path"].name,
                            "kind": "card_skipped", "reason": "image name used more than once"})
        cards = [c for c in cards if c["source_id"] not in dupes]
    return cards, skipped


def derive_seed(*parts) -> int:
    """Stable 63-bit seed from the global seed plus ids (same on every machine/run)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & (2 ** 63 - 1)


# =============================================================================
# One card
# =============================================================================
class Outputs:
    """Output folders and file names."""

    def __init__(self, cfg: dict):
        out = s1.cfg_path(cfg, cfg["paths"]["out_root"])
        self.templates = out / "templates"
        self.images = out / "rendered" / "images"
        self.labels = out / "rendered" / "labels"
        self.manifest = out / "rendered" / "manifest.csv"
        self.skipped = out / "rendered" / "skipped.csv"
        self.preview = s1.cfg_path(cfg, cfg["batch"]["preview_dir"])
        self.degraded = [out / "degraded" / "images", out / "degraded" / "labels"]  # Task A, later
        for d in (self.templates, self.images, self.labels, self.preview, *self.degraded):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def stem(source_id, layout, slot):
        return f"{source_id}_{layout}_ren_clean_{slot:02d}"

    def card_files(self, source_id, layout, n):
        return [self.images / f"{self.stem(source_id, layout, k)}.png" for k in range(1, n + 1)]

    def remove_card(self, source_id, layout):
        """Delete earlier rendered outputs of one card (before regenerating it)."""
        pattern = f"{source_id}_{layout}_ren_clean_*"
        for d, ext in ((self.images, ".png"), (self.labels, ".txt"), (self.preview, ".png")):
            for f in d.glob(pattern + ext):
                f.unlink()


def process_card(card: dict, cfg: dict, names: dict, out: Outputs, log: list) -> list[dict]:
    """Template + samples for one source card. Returns manifest rows.

    Skips and dropped samples are appended to `log`; SynthError means the
    card itself cannot be used (logged by the caller).
    """
    b, sid, layout = cfg["batch"], card["source_id"], card["layout"]
    card_cfg, layout_cfg = s1.layout_settings(cfg, layout)

    label = s1.cfg_path(cfg, b["labels_dir"]) / f"{sid}.txt"
    if not label.is_file():
        raise SynthError("label file missing")
    img = cv2.imread(str(card["path"]), cv2.IMREAD_COLOR)
    if img is None:
        raise SynthError("image could not be read")
    H, W = img.shape[:2]
    boxes_orig = s1.read_yolo_labels(str(label), W, H)  # raises unless exactly 6 boxes, one per class

    card_seed = derive_seed(cfg["generation"]["seed"], sid)
    card_cfg = s1.scale_pixel_settings(card_cfg, W)
    card_cfg = {**card_cfg, "generation": {**card_cfg["generation"], "seed": card_seed}}

    # ---- Step 1 + 2: template, measurements, fonts -----------------------------
    blank, fields, forbidden = s1.make_blank_template(img, boxes_orig, card_cfg, layout_cfg)
    s1.assign_fonts(fields, card_cfg, verbose=False, layout_cfg=layout_cfg)
    prefix = f"{sid}_{layout}"
    cv2.imwrite(str(out.templates / f"{prefix}_template.png"), blank)
    meta = {"source_image": card["path"].name, "source_id": sid, "layout": layout,
            "image_size": [W, H], "card_seed": card_seed, "class_names": list(CLASS_NAMES),
            "fields": fields}
    (out.templates / f"{prefix}_template.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Steps 3 + 4: samples ---------------------------------------------------
    rend = s1.FieldRenderer(fields, card_cfg, layout_cfg)
    n = b["samples_per_card"]
    card_rng = np.random.default_rng(card_seed)
    hard_slots = set((card_rng.choice(n, size=min(b["hard_per_card"], n), replace=False) + 1).tolist())
    hard_mode = "wrap" if layout_cfg["names_may_wrap"] else "hard"

    rows = []
    for slot in range(1, n + 1):
        hard = slot in hard_slots
        stem = out.stem(sid, layout, slot)
        reasons = []
        for retry in range(b["max_retries"] + 1):
            sample_seed = derive_seed(cfg["generation"]["seed"], sid, slot, retry)
            rng = np.random.default_rng(sample_seed)
            try:
                gen = s1.FakeValues(names, layout_cfg, card_cfg["generation"], rng)
                values = s1.choose_card_values(gen, rend, hard_mode if hard else "normal",
                                               card_cfg, layout_cfg, rng)
            except SynthError as e:
                reasons = [f"no value fits: {e}"]
                continue
            img_out, info = s1.render_card(blank, rend, values, card_cfg, rng)
            boxes, visible = s1.boxes_from_changes(img_out, blank, info, fields, forbidden, card_cfg)
            reasons = s1.check_sample(boxes, visible, info, forbidden, card_cfg)
            if reasons:
                continue

            cv2.imwrite(str(out.images / f"{stem}.png"), img_out)
            (out.labels / f"{stem}.txt").write_text(s1.yolo_lines(boxes, W, H))
            cv2.imwrite(str(out.preview / f"{stem}.png"), s1.draw_boxes(img_out, boxes))
            row = {"file": f"{stem}.png", "label_file": f"{stem}.txt", "source_id": sid, "layout": layout,
                   "seed": cfg["generation"]["seed"], "card_seed": card_seed, "sample_seed": sample_seed,
                   "slot": slot, "retries": retry, "hard_case": hard,
                   "wrapped": any(len(v) > 1 for v in values.values())}
            for name in CLASS_NAMES:
                i, fnt = info[name], fields[name]["font"]
                row.update({f"{name}_value": " ".join(values[name]),
                            f"{name}_font": Path(fnt["path"]).name, f"{name}_size": fnt["size"],
                            f"{name}_dx": i["dx"], f"{name}_dy": i["dy"],
                            f"{name}_darkness": i["darkness"], f"{name}_blur": i["blur"]})
            rows.append(row)
            break
        else:
            log.append({"source_id": sid, "layout": layout, "file": f"{stem}.png", "kind": "sample_dropped",
                        "reason": "; ".join(reasons)})
    return rows


# =============================================================================
# CSV bookkeeping (merged with earlier runs, so resumed cards keep their rows)
# =============================================================================
def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    # utf-8-sig so Excel shows the Bangla values correctly.
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def merge_rows(old: list[dict], new: list[dict], handled: set, key: str) -> list[dict]:
    """Old rows of cards not handled in this run + all new rows, sorted."""
    kept = [r for r in old if r.get("source_id") not in handled]
    return sorted(kept + new, key=lambda r: (r.get("source_id", ""), r.get(key, "")))


# =============================================================================
# Final checks over the whole batch
# =============================================================================
def final_checks(out: Outputs, cfg: dict) -> tuple[Counter, list[str]]:
    """Re-read every rendered label and check it. Returns (count per layout, problems)."""
    c = cfg["checks"]
    per_layout, problems = Counter(), []
    for img_path in sorted(out.images.glob("*_ren_clean_*.png")):
        layout = next((lay for lay in cfg["layouts"] if f"_{lay}_ren_clean_" in img_path.name), "?")
        per_layout[layout] += 1
        lbl = out.labels / f"{img_path.stem}.txt"
        if not lbl.is_file():
            problems.append(f"{img_path.name}: no label file")
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        lines = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()]
        ids = [int(ln[0]) for ln in lines]
        if len(lines) != len(CLASS_NAMES) or sorted(ids) != list(range(len(CLASS_NAMES))):
            problems.append(f"{lbl.name}: not exactly 6 lines, one per class")
            continue
        boxes = []
        for ln in lines:
            cx, cy, bw, bh = (float(v) for v in ln[1:])
            # Back to whole pixels (labels are written with 6 decimals).
            x1, y1, x2, y2 = (round(v) for v in ((cx - bw / 2) * W, (cy - bh / 2) * H,
                                                 (cx + bw / 2) * W, (cy + bh / 2) * H))
            name = CLASS_NAMES[int(ln[0])]
            if x1 <= 0 or y1 <= 0 or x2 >= W or y2 >= H:
                problems.append(f"{lbl.name}: {name} box touches the image edge")
            if y2 - y1 < c["min_box_height_px"]:
                problems.append(f"{lbl.name}: {name} box under {c['min_box_height_px']} px tall")
            boxes.append((name, (x1, y1, x2, y2)))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if s1.iou(boxes[i][1], boxes[j][1]) > c["max_iou"]:
                    problems.append(f"{lbl.name}: {boxes[i][0]}/{boxes[j][0]} IoU > {c['max_iou']}")
    return per_layout, problems


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Task B field-content synthesis for every card in images/.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N cards (taken alternately from each card type)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate cards whose outputs already exist")
    args = ap.parse_args(argv)

    cfg = s1.load_config(args.config)
    if "batch" not in cfg:
        raise SynthError("config.yaml is missing the 'batch' section")
    b = cfg["batch"]
    for layout in set(b["layout_folders"].values()) & set(cfg["layouts"]):
        s1.layout_settings(cfg, layout)  # fail fast on a bad layout block or missing font
    s1.check_raqm()
    names = yaml.safe_load(s1.cfg_path(cfg, cfg["paths"]["names_file"]).read_text(encoding="utf-8"))
    out = Outputs(cfg)

    cards, log = discover_cards(cfg)
    for entry in log:
        print(f"skip  {entry['file']}: {entry['reason']}")
    if args.limit is not None:
        cards = cards[:max(0, args.limit)]
    n = b["samples_per_card"]

    new_rows, handled = [], set(e["source_id"] for e in log)
    stats = Counter()
    t0 = time.time()
    try:
        for k, card in enumerate(cards, 1):
            sid, layout = card["source_id"], card["layout"]
            tag = f"[{k}/{len(cards)}] {sid} ({layout})"
            if not args.overwrite and all(f.is_file() for f in out.card_files(sid, layout, n)):
                stats["resumed"] += 1
                print(f"{tag}: already rendered, skipped (use --overwrite to redo)")
                continue

            handled.add(sid)
            out.remove_card(sid, layout)
            card_log = []
            try:
                rows = process_card(card, cfg, names, out, card_log)
            except SynthError as e:
                stats["skipped"] += 1
                log.append({"source_id": sid, "layout": layout, "file": card["path"].name,
                            "kind": "card_skipped", "reason": str(e).splitlines()[0]})
                print(f"{tag}: SKIPPED - {str(e).splitlines()[0]}")
                continue
            except Exception as e:  # never stop the batch because of one card
                out.remove_card(sid, layout)  # no half-finished card without manifest rows
                stats["errors"] += 1
                reason = f"error: {type(e).__name__}: {e}"
                log.append({"source_id": sid, "layout": layout, "file": card["path"].name,
                            "kind": "card_error", "reason": reason})
                print(f"{tag}: ERROR - {reason}")
                continue

            stats["processed"] += 1
            new_rows += rows
            log += card_log
            retries = sum(r["retries"] for r in rows)
            print(f"{tag}: {len(rows)}/{n} made"
                  + (f", {retries} retries" if retries else "")
                  + (f", {len(card_log)} dropped" if card_log else ""))
    finally:
        # Written even on Ctrl+C, so finished cards are recorded.
        manifest = merge_rows(read_csv(out.manifest), new_rows, handled, "file")
        write_csv(out.manifest, MANIFEST_FIELDS, manifest)
        skipped = merge_rows(read_csv(out.skipped), log, handled, "file")
        write_csv(out.skipped, SKIPPED_FIELDS, skipped)

    # ---- Summary ------------------------------------------------------------------
    dropped = [e for e in log if e["kind"] == "sample_dropped"]
    card_skips = [e for e in log if e["kind"] == "card_skipped"]
    made = Counter(r["layout"] for r in new_rows)
    print(f"\nSummary ({time.time() - t0:.0f} s)")
    print(f"  cards processed : {stats['processed']}")
    print(f"  cards skipped   : {len(card_skips)}"
          + (f"  (+{stats['errors']} errors)" if stats["errors"] else "")
          + (f"  (+{stats['resumed']} already done)" if stats["resumed"] else ""))
    for reason, count in Counter(e["reason"] for e in card_skips).most_common(8):
        print(f"      {count:3d}  {reason}")
    print(f"  images made     : {len(new_rows)}  ("
          + ", ".join(f"{lay}: {made[lay]}" for lay in sorted(set(b['layout_folders'].values()))) + ")")
    print(f"  hard cases      : {sum(r['hard_case'] for r in new_rows)}")
    print(f"  samples dropped : {len(dropped)}")
    # One sample can fail several checks; count each reason once per sample.
    top = Counter(r.strip() for e in dropped for r in set(e["reason"].split("; ")))
    for reason, count in top.most_common(8):
        print(f"      {count:3d}  {reason}")

    # ---- Final checks over everything in rendered/ --------------------------------
    per_layout, problems = final_checks(out, cfg)
    print("\nFinal checks (all rendered images)")
    print("  images per layout: " + ", ".join(f"{k}: {v}" for k, v in sorted(per_layout.items())))
    print(f"  problems: {len(problems)}")
    for p in problems[:20]:
        print(f"    - {p}")
    counts = [per_layout.get(lay, 0) for lay in sorted(set(b["layout_folders"].values()))]
    if min(counts) and max(counts) > b["layout_balance_warn"] * min(counts):
        print(f"  WARNING: layouts are unbalanced ({counts}); one has more than "
              f"{b['layout_balance_warn']}x the other")
    elif not min(counts) and max(counts):
        print("  WARNING: one layout has no rendered images at all")
    print(f"\n  manifest : {out.manifest}\n  skipped  : {out.skipped}\n  previews : {out.preview}")
    return 1 if problems else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SynthError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
