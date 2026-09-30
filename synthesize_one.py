#!/usr/bin/env python3
"""
synthesize_one.py -- Task B (field-content synthesis) for ONE NID card image.

Pipeline
  Step 1  Blank template: erase only the ink inside the 6 value boxes (labels,
          photo, background pattern and borders stay untouched) and measure
          each field: start point, height, colour, max width, box margins.
  Step 2  Fonts: require Pillow + libraqm, render a Bangla shaping test, pick
          the font size per field whose ink height matches the measured one,
          and save an original-vs-fake comparison.
  Step 3  Fake values: made-up names, dates and NID numbers only. The real
          values on the card are never read, copied or printed.
  Step 4  Render + label: draw values with small imperfections, derive each
          box from the pixels that actually changed, write YOLO labels.
  Checks  Every output is validated; samples that fail are dropped.

All settings live in config.yaml; made-up name parts live in fake_names.yaml.

Usage
  python synthesize_one.py --image nid_0006_front.png --label nid_0006_front.txt --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, features

# Fixed class order -- never change: the index is the YOLO class id.
CLASS_NAMES = ("name_bn", "name_en", "guardian", "mother", "dob", "nid")
MONTHS_SHORT = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MONTHS_LONG = ("January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December")
# Box colours (BGR) for the preview images, one per class.
PREVIEW_COLORS = ((0, 0, 255), (0, 160, 0), (255, 0, 0),
                  (0, 140, 255), (200, 0, 200), (160, 160, 0))

RAQM_HELP = """\
Pillow was built without libraqm (complex text layout), so Bangla conjuncts and
vowel signs would be drawn wrongly. This script does not fall back to basic
rendering. Install raqm support, then run again:

  Windows : pip install --upgrade --force-reinstall pillow
            (the official PyPI wheels bundle raqm, FriBiDi and HarfBuzz)
  Linux   : sudo apt install libraqm0 libfribidi0 libharfbuzz0b
            pip install --upgrade --force-reinstall pillow
  macOS   : brew install libraqm
            pip install --upgrade --force-reinstall pillow

Check with:  python -c "from PIL import features; print(features.check('raqm'))"
"""


class SynthError(RuntimeError):
    """A fatal, user-facing problem. Printed without a traceback."""


# =============================================================================
# Config and inputs
# =============================================================================
def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SynthError(f"Config file not found: {p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["_dir"] = p.resolve().parent

    for section in ("class_names", "layout", "layouts", "fields", "fonts", "template",
                    "paths", "generation", "augment", "checks"):
        if section not in cfg:
            raise SynthError(f"config.yaml is missing the '{section}' section")
    if tuple(cfg["class_names"]) != CLASS_NAMES:
        raise SynthError(f"class_names must be exactly {list(CLASS_NAMES)} (fixed order)")
    if cfg["layout"] not in cfg["layouts"]:
        raise SynthError(f"layout '{cfg['layout']}' has no block under 'layouts:'")
    for name in CLASS_NAMES:
        if name not in cfg["fields"]:
            raise SynthError(f"config.yaml 'fields' has no entry for '{name}'")
        font = Path(cfg["fields"][name]["font"])
        if not font.is_file():
            raise SynthError(f"Font for field '{name}' not found: {font}")
    return cfg


def cfg_path(cfg: dict, rel: str) -> Path:
    """Resolve a path from the config relative to the config file's folder."""
    p = Path(rel)
    return p if p.is_absolute() else cfg["_dir"] / p


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def layout_settings(cfg: dict, layout: str) -> tuple[dict, dict]:
    """(cfg, layout_cfg) for one card type.

    A layout block may carry `overrides:` with any top-level section (fields,
    fonts, generation, template, augment, checks); they are merged over the
    global settings, so e.g. smart cards can use their own fonts and birth years.
    """
    if layout not in cfg["layouts"]:
        raise SynthError(f"layout '{layout}' has no block under 'layouts:'")
    lay = cfg["layouts"][layout]
    if sum(lay["nid_groups"]) != lay["nid_digits"]:
        raise SynthError(f"layout '{layout}': nid_groups must add up to nid_digits")
    merged = _deep_merge(cfg, lay.get("overrides", {}))
    for name in CLASS_NAMES:
        font = Path(merged["fields"][name]["font"])
        if not font.is_file():
            raise SynthError(f"layout '{layout}': font for field '{name}' not found: {font}")
    return merged, lay


# Template settings measured in pixels (tuned at template.reference_width_px)
# and whether they scale with length or with area.
_PIXEL_SETTINGS = {"bg_kernel_px": 1, "halo_px": 1, "erase_dilate_px": 1, "bg_ring_px": 1,
                   "inpaint_radius": 1, "inpaint_extra_dilate_px": 1, "safety_gap_px": 1,
                   "obstacle_min_px": 1, "min_component_px": 2, "thin_line_open_px": 1,
                   "pattern_line_min_length_px": 1, "pattern_line_max_width_px": 1,
                   "obstacle_min_area_px": 2, "erase_close_px": 1, "attach_px": 1}
# May scale down to 0 (e.g. no line opening on a tiny card, where it would eat the text).
_PIXEL_SETTINGS_MIN0 = {"thin_line_open_px"}


def scale_pixel_settings(cfg: dict, image_width: int) -> dict:
    """Copy of cfg with the pixel-sized template settings scaled to this image.

    Cards range from ~300 to ~1500 px wide; a stroke-removal kernel or safety
    gap tuned on a 1062 px card is wrong at other sizes. Jitter and the check
    limits stay in absolute pixels on purpose (they are rules, not tuning).
    """
    tc = dict(cfg["template"])
    s = image_width / tc["reference_width_px"]
    for key, power in _PIXEL_SETTINGS.items():



        tc[key] = max(0 if key in _PIXEL_SETTINGS_MIN0 else 1, int(round(tc[key] * s ** power)))
    tc["bg_kernel_px"] = max(3, tc["bg_kernel_px"] | 1)  # odd, >= 3
    return {**cfg, "template": tc}


def _format_samples(name: str, layout_cfg: dict) -> list[str]:
    """Values with the same character layout as the card's DOB / NID."""
    if name == "nid":
        digits = "8" * layout_cfg["nid_digits"]
        parts, i = [], 0
        for g in layout_cfg["nid_groups"]:
            parts.append(digits[i:i + g])
            i += g
        return [layout_cfg["nid_separator"].join(parts)]
    return [format_date(dt.date(1975, m, 15), layout_cfg["date_format"]) for m in range(1, 13)]


def width_capped_size(fc: dict, name: str, size: int, orig_width: int, layout_cfg: dict, cfg: dict):
    """Cap a height-matched size for fixed-format fields (DOB, NID).

    Blur thickens every stroke, which inflates a small measured height a lot
    (a 45 px digit can measure 59 px) but a long width hardly at all. The
    fake DOB/NID has the same format as the original, so a same-format value
    must not come out clearly wider than the original did. Returns the size
    to use and whether the width decided it.
    """
    samples = _format_samples(name, layout_cfg)
    lang = lang_of(fc["script"])

    def width(sz):
        font = load_font(fc["font"], fc.get("font_index", 0), sz)
        ws = []
        for t in samples:
            mask, _ = render_mask([t], font, lang)
            cols = np.nonzero((mask > cfg["fonts"]["ink_alpha_threshold"]).any(axis=0))[0]
            ws.append(cols[-1] - cols[0] + 1)
        return float(np.median(ws))

    limit = orig_width * (1 + cfg["fonts"]["width_cap_tolerance"])
    if width(size) <= limit:
        return size, False
    lo = cfg["fonts"]["size_range"][0]
    while size > lo and width(size) > orig_width:
        size -= 1
    return size, True


def assign_fonts(fields: dict, cfg: dict, verbose: bool = True, layout_cfg: dict | None = None):
    """Step 2.3/2.5: pick the font size per field and store font info in `fields`.

    Fields printed at one size (e.g. the three Bangla names) share the median
    of their measured heights, so one noisy measurement cannot skew a field.
    With `layout_cfg`, DOB and NID sizes are also capped by the original's
    width (see width_capped_size).
    """
    target_h = {n: f["ref_height"] for n, f in fields.items()}
    for group in cfg["fonts"].get("shared_size_groups", []):
        shared = int(round(np.median([fields[n]["ref_height"] for n in group])))
        target_h.update({n: shared for n in group})
    for name, f in fields.items():
        fc = cfg["fields"][name]
        size, offset, got = pick_font_size(fc, target_h[name], cfg)
        by_width = False
        if layout_cfg is not None and name in ("dob", "nid"):
            capped, by_width = width_capped_size(fc, name, size, f["orig_ink_width"], layout_cfg, cfg)
            if by_width:
                size = capped
                # Re-measure the anchor offset at the new size.
                font = load_font(fc["font"], fc.get("font_index", 0), size)
                mask, (_, ay) = render_mask([cfg["fonts"]["reference_text"][fc["script"]]], font,
                                            lang_of(fc["script"]))
                m = text_metrics(mask > 127, fc["script"])
                offset, got = m["anchor_y"] - ay, m["ref_height"]
        f["font"] = {"path": fc["font"], "index": fc.get("font_index", 0), "size": size,
                     "draw_baseline_y": f["anchor_y"] - offset, "rendered_ref_height": got,
                     "size_source": "width" if by_width else "height"}
        if verbose:
            print(f"    {name:9s} {Path(fc['font']).name:14s} size={size:3d} "
                  f"(ref height measured {f['ref_height']}, target {target_h[name]}, rendered {got}"
                  + (", capped by original width" if by_width else "") + ")")


def read_yolo_labels(path: str, width: int, height: int) -> dict[int, tuple]:
    """Read the YOLO label file -> {class_id: (x1, y1, x2, y2)} in pixels (x2/y2 exclusive)."""
    p = Path(path)
    if not p.is_file():
        raise SynthError(f"Label file not found: {p}")
    rows = [ln.split() for ln in p.read_text().splitlines() if ln.strip()]
    if len(rows) != len(CLASS_NAMES):
        raise SynthError(f"{p.name}: expected exactly {len(CLASS_NAMES)} boxes, found {len(rows)}")

    boxes = {}
    for r in rows:
        if len(r) != 5:
            raise SynthError(f"{p.name}: malformed YOLO line: {' '.join(r)}")
        cid = int(r[0])
        cx, cy, bw, bh = map(float, r[1:])
        if cid not in range(len(CLASS_NAMES)) or cid in boxes:
            raise SynthError(f"{p.name}: class ids must be 0-5, each exactly once (bad id {cid})")
        x1 = max(0, round((cx - bw / 2) * width))
        y1 = max(0, round((cy - bh / 2) * height))
        x2 = min(width, round((cx + bw / 2) * width))
        y2 = min(height, round((cy + bh / 2) * height))
        boxes[cid] = (x1, y1, x2, y2)
    return boxes


# =============================================================================
# Small image helpers
# =============================================================================
def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def ellipse(radius: int) -> np.ndarray:
    k = 2 * max(0, int(radius)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), ellipse(radius)).astype(bool)


def darkness_map(gray: np.ndarray, kernel_px: int) -> np.ndarray:
    """How much darker each pixel is than its local background.

    A grayscale closing removes dark strokes thinner than the kernel, which
    leaves an estimate of the background (including faint patterns); the
    difference is high on text and near zero on paper and watermark.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    return cv2.subtract(background, gray)


def mask_bbox(mask: np.ndarray):
    """(x1, y1, x2, y2) of the True pixels, x2/y2 inclusive; None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def text_metrics(mask: np.ndarray, script: str) -> dict:
    """Measure a text ink mask in a way that does not depend on its content.

    - Baseline: median bottom of the ink columns that reach the lower part of
      the text (ignores Bangla matra-only columns, T-bars and the minority of
      descender columns such as g, y or the Bangla u-kar).
    - Reference top: the matra (head-line) for Bangla -- it spans almost every
      column, so the median column top sits on it while vowel signs above it
      (i-kar, e-kar ...) are a minority; the cap/digit top (5th percentile of
      column tops) for Latin.
    - Anchor row used for placement: matra for Bangla, baseline for Latin.
    Used identically on the real ink and on rendered reference strings, so the
    two heights are directly comparable.
    """
    bb = mask_bbox(mask)
    if bb is None:
        raise SynthError("text_metrics: empty ink mask")
    x1, y1, x2, y2 = bb
    m = mask[y1:y2 + 1, x1:x2 + 1]
    has = m.any(axis=0)
    col_top = np.argmax(m, axis=0)[has]
    col_bot = (m.shape[0] - 1 - np.argmax(m[::-1], axis=0))[has]

    if script == "bn":
        ref_top = int(round(np.median(col_top)))
    else:
        ref_top = int(round(np.percentile(col_top, 5)))
    tall = col_bot - ref_top > 0.35 * (m.shape[0] - 1 - ref_top)
    baseline = int(round(np.median(col_bot[tall] if tall.any() else col_bot)))

    return {
        "left": x1, "top": y1, "right": x2, "bottom": y2,
        "ink_height": y2 - y1 + 1,
        "ref_top": y1 + ref_top,
        "baseline": y1 + baseline,
        "ref_height": baseline - ref_top + 1,
        "anchor_y": y1 + (ref_top if script == "bn" else baseline),
        "anchor_kind": "matra" if script == "bn" else "baseline",
    }


# =============================================================================
# Step 1: blank template
# =============================================================================
def _value_components(d: np.ndarray, inside: np.ndarray, min_contrast: int, tc: dict):
    """Split the ink around one box (darkness map `d`, padded crop) into value
    ink and foreign ink. `inside` marks the real box within the crop.

    - Thin background lines (guilloche, 1-2 px) are opened away before
      grouping pixels into components, so they cannot glue letters to each
      other or to the box edge; the letters' own thin parts are added back.
    - Foreign = components mostly outside the box (a label, border or the
      neighbouring value reaching in) and components lying wholly above/below
      the value's text line with a clear gap (a small label printed inside
      the box, specks). Diacritics that touch the line stay with the value.
    Returns (keep, attached, foreign, thr), all masks over the padded crop (so
    value ink poking out of a tight box is still erased). `keep` is the value
    ink used for measuring; `attached` is every dark pixel connected to it
    (thin stroke ends the opening cut off, and any line glued to a letter),
    which must be erased too.
    """
    otsu, _ = cv2.threshold(d[inside].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(min_contrast, otsu)
    strong = d > thr
    r = tc["thin_line_open_px"]
    core = cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_OPEN, ellipse(r)) if r > 0 else strong.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    in_px = np.bincount(lab[inside], minlength=n)

    inner, foreign_ids = [], []
    for i in range(1, n):
        area = stats[i][4]
        if area < tc["min_component_px"] or in_px[i] == 0:
            continue
        (inner if in_px[i] >= tc["min_inside_fraction"] * area else foreign_ids).append(i)

    if inner:
        # The text line = vertical extent of the large components.
        amax = max(stats[i][4] for i in inner)
        big = [i for i in inner if stats[i][4] >= tc["main_line_area_ratio"] * amax]
        top = min(stats[i][1] for i in big)
        bottom = max(stats[i][1] + stats[i][3] for i in big)
        max_gap = tc["main_line_gap_ratio"] * (bottom - top)
        keep_ids = []
        for i in inner:
            y, h = stats[i][1], stats[i][3]
            gap = top - (y + h) if y + h <= top else (y - bottom if y >= bottom else 0)
            (keep_ids if gap <= max_gap else foreign_ids).append(i)
        # Real boxes are sometimes drawn tighter than the value, so its last
        # letters fall mostly outside the box. Labels are only ever left of or
        # above/below a value, never on its line to the right, so a component
        # centred on the text line at/after the value's start is value ink.
        if keep_ids:
            left = min(stats[i][0] for i in keep_ids)
            for i in list(foreign_ids):
                x, y, w, h = stats[i][:4]
                if x >= left and top <= y + h / 2 <= bottom and stats[i][4] >= tc["min_component_px"]:
                    foreign_ids.remove(i)
                    keep_ids.append(i)
    else:
        keep_ids = []

    keep = strong & dilate(np.isin(lab, keep_ids), r + 1)
    foreign = strong & dilate(np.isin(lab, foreign_ids), r + 1) & ~keep
    # Only near the value's own strokes: a label whose thin strokes vanished in
    # the opening can be glued to the value by a pattern line, and must survive.
    _, lab_s = cv2.connectedComponents(strong.astype(np.uint8), connectivity=8)
    touching = np.unique(lab_s[keep])
    attached = np.isin(lab_s, touching[touching > 0]) & ~foreign & dilate(keep, tc["attach_px"])
    return keep, attached, foreign, thr


def detect_value_ink(dark: np.ndarray, dark_color: np.ndarray, box, tc: dict):
    """Find the value's ink pixels inside one box.

    `dark` is luminance darkness (finds the solid text); `dark_color` is the
    per-channel maximum darkness, which also catches coloured halos (e.g. the
    pink fringe around red text) and faded coloured print (pale orange
    digits) that is almost as bright as the paper. The colour map is only
    used for the text itself when the luminance pass finds (almost) nothing.

    Returns (ink, erase, foreign): full-size bool masks. `ink` is the solid
    text used for measuring, `erase` adds its anti-aliased / coloured halo,
    `foreign` holds ink that is not part of the value (see
    _value_components) -- it is never erased.
    """
    H, W = dark.shape
    x1, y1, x2, y2 = box
    pad = max(3, int(round(tc["box_pad_ratio"] * (y2 - y1))))
    X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    d, dc = dark[Y1:Y2, X1:X2], dark_color[Y1:Y2, X1:X2]
    inside = np.zeros(d.shape, bool)
    inside[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = True

    keep, attached, foreign, thr = _value_components(d, inside, tc["min_ink_contrast"], tc)
    keep_c, attached_c, foreign_c, thr_c = _value_components(dc, inside, tc["min_ink_contrast_color"], tc)
    extra = np.zeros(d.shape, bool)
    if keep.sum() < tc["fallback_ink_fraction"] * inside.sum():
        if keep_c.sum() > keep.sum():
            keep, attached, foreign, thr, d = keep_c, attached_c, foreign_c, thr_c, dc
    elif keep.any():
        # Unevenly lit print: part of a coloured value (e.g. the last NID
        # digits) can be too faint for the luminance pass. Colour-pass ink on
        # the value's own line, from its start rightwards, is erased as well.
        bx1, by1, bx2, by2 = mask_bbox(keep)
        extra = keep_c & ~dilate(keep, tc["halo_px"])
        extra[:by1] = extra[by2 + 1:] = False
        extra[:, :bx1] = False

    # Halo: weaker (or merely coloured) pixels, only right next to the text so
    # the watermark further away is left alone.
    weak = (d > thr * tc["weak_ink_ratio"]) | (dc > tc["halo_min_contrast"])
    erase = keep | attached | (weak & dilate(keep | attached, tc["halo_px"])) | dilate(extra, tc["halo_px"])
    # Heavy, blurry strokes can be wider than the background kernel, leaving
    # their cores undetected; close small gaps and fill enclosed holes so no
    # dark core survives the erase (erasing a letter's counter is harmless).
    r = tc["erase_close_px"]
    if r > 0:
        erase = cv2.morphologyEx(erase.astype(np.uint8), cv2.MORPH_CLOSE, ellipse(r)).astype(bool)
    reach = np.pad(erase, 1).astype(np.uint8)
    cv2.floodFill(reach, None, (0, 0), 1)
    erase |= reach[1:-1, 1:-1] == 0
    erase &= ~dilate(foreign, 1)

    def full(local):
        out = np.zeros((H, W), bool)
        out[Y1:Y2, X1:X2] = local
        return out

    return full(keep), full(erase), full(foreign)


def erase_ink(work: np.ndarray, erase: np.ndarray, tc: dict, rng) -> tuple[str, float]:
    """Remove the ink in `erase` from `work` (in place). Returns (method, spread)."""
    mask = dilate(erase, tc["erase_dilate_px"])
    ring = dilate(mask, tc["bg_ring_px"]) & ~mask
    smooth = cv2.medianBlur(to_gray(work), 5)[ring]
    spread = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))

    if spread <= tc["plain_bg_max_spread"]:
        # Plain background: fill with the sampled background colour plus the
        # paper's own grain so the patch does not look flat.
        colors = work[ring].reshape(-1, 3).astype(np.float32)
        median, std = np.median(colors, axis=0), colors.std(axis=0)
        noise = rng.normal(0.0, 1.0, (int(mask.sum()), 3)) * std * tc["plain_fill_noise_scale"]
        work[mask] = np.clip(median + noise, 0, 255).astype(np.uint8)
        return "plain_fill", spread

    # Patterned background (watermark, guilloche): let inpainting continue it.
    inpaint_mask = dilate(mask, tc["inpaint_extra_dilate_px"]).astype(np.uint8) * 255
    work[:] = cv2.inpaint(work, inpaint_mask, tc["inpaint_radius"], cv2.INPAINT_TELEA)
    return "inpaint", spread


def obstacle_mask(img: np.ndarray, cfg: dict, layout_cfg: dict) -> np.ndarray:
    """Non-background elements: labels, borders, printed text, photo, signature, logo."""
    tc = cfg["template"]
    H, W = img.shape[:2]
    gray = to_gray(img)
    dark = darkness_map(gray, tc["bg_kernel_px"])
    obs = (dark > tc["obstacle_min_contrast"]) | (gray < tc["obstacle_abs_dark"])
    obs = cv2.morphologyEx(obs.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Drop background pattern: long thin curves (guilloche) that cross the
    # value lines on smart cards, and mesh fragments too small to be a label
    # letter. Labels, photo and borders are compact, larger shapes.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(obs, connectivity=8)
    length = np.maximum(stats[:, 2], stats[:, 3])
    thin = (length >= tc["pattern_line_min_length_px"]) & \
           (stats[:, 4] <= tc["pattern_line_max_width_px"] * length)
    pattern = thin | (stats[:, 4] < tc["obstacle_min_area_px"])
    pattern[0] = False
    obs = obs.astype(bool) & ~pattern[lab]

    for nx1, ny1, nx2, ny2 in layout_cfg.get("forbidden_regions", {}).values():
        obs[int(ny1 * H):int(ny2 * H), int(nx1 * W):int(nx2 * W)] = True
    return obs


def make_blank_template(img, boxes, cfg, layout_cfg):
    """Step 1: erase the 6 values and measure each field. Returns (blank, fields, forbidden)."""
    tc = cfg["template"]
    H, W = img.shape[:2]
    dark = darkness_map(to_gray(img), tc["bg_kernel_px"])
    dark_color = np.max([darkness_map(np.ascontiguousarray(img[..., c]), tc["bg_kernel_px"])
                         for c in range(3)], axis=0)
    blank = img.copy()
    rng = np.random.default_rng(cfg["generation"]["seed"])  # only for the fill grain
    jitter_max = max(cfg["augment"]["jitter_px"])
    fields, inks = {}, {}
    foreign_all = np.zeros((H, W), bool)   # labels / specks next to the values
    erased_all = np.zeros((H, W), bool)

    for cid, name in enumerate(CLASS_NAMES):
        box = boxes[cid]
        script = cfg["fields"][name]["script"]
        ink, erase, foreign = detect_value_ink(dark, dark_color, box, tc)
        if not ink.any():
            raise SynthError(f"No text ink found inside the '{name}' box")
        foreign_in_box = int(foreign[box[1]:box[3], box[0]:box[2]].sum())
        if foreign_in_box:
            print(f"  note: '{name}': ink outside the value line was left untouched "
                  f"({foreign_in_box} px, probably a label, border or speck)")

        m = text_metrics(ink, script)
        ink_px = img[ink].reshape(-1, 3)
        lum = to_gray(ink_px.reshape(-1, 1, 3)).ravel()
        darkest = ink_px[lum <= np.quantile(lum, tc["darkest_ink_fraction"])]
        b, g, r = np.median(darkest, axis=0)

        method, spread = erase_ink(blank, erase, tc, rng)
        inks[name] = ink
        foreign_all |= foreign
        erased_all |= erase
        fields[name] = {
            "class_id": cid,
            "script": script,
            "orig_box": list(box),
            "start_x": m["left"],
            "start_y": m["top"],
            "anchor_kind": m["anchor_kind"],
            "anchor_y": m["anchor_y"],
            "text_height": m["ink_height"],
            "ref_height": m["ref_height"],
            "orig_ink_width": m["right"] - m["left"] + 1,
            "color_rgb": [int(r), int(g), int(b)],
            # A real box drawn tighter than the ink gives a negative margin;
            # the synthetic box must always hold the whole text, so clamp.
            "margins": {
                "left": max(tc["min_margin_px"], m["left"] - box[0]),
                "top": max(tc["min_margin_px"], m["top"] - box[1]),
                "right": max(tc["min_margin_px"], (box[2] - 1) - m["right"]),
                "bottom": max(tc["min_margin_px"], (box[3] - 1) - m["bottom"]),
            },
            "erase_method": method,
            "bg_spread": round(spread, 1),
        }

    # Max width: from the text start to the next non-background element on
    # the same line (label, photo, border, card edge), minus a safety gap.
    obstacles = obstacle_mask(blank, cfg, layout_cfg)
    for name, f in fields.items():
        x1, y1, x2, y2 = f["orig_box"]
        leftover = int((obstacles[y1:y2, x1:x2] & dilate(inks[name], 2)[y1:y2, x1:x2]).sum())
        if leftover:
            print(f"  note: '{name}': {leftover} dark px remain where the value was erased")
    # Where the real value was printed there can be no label or photo: anything
    # dark left there is pattern or erase residue. Clear it so it neither ends
    # the line nor makes the checks reject a box.
    g = tc["erase_dilate_px"]
    for f in fields.values():
        obstacles[max(0, f["start_y"] - g):f["start_y"] + f["text_height"] + g,
                  max(0, f["start_x"] - g):f["start_x"] + f["orig_ink_width"] + g] = False
    for name, f in fields.items():
        # Scan only the rows of the text line itself, so a small label printed
        # just above or below the value (smart cards) does not end the line.
        sx, ty1, ty2 = f["start_x"], f["start_y"], f["start_y"] + f["text_height"]
        counts = obstacles[ty1:ty2, sx:].sum(axis=0)
        hits = np.nonzero(counts >= tc["obstacle_min_px"])[0]
        obstacle_x = sx + int(hits[0]) if hits.size else W
        f["obstacle_x"] = obstacle_x
        f["max_width"] = obstacle_x - tc["safety_gap_px"] - sx
        # What the ink itself may use: the jitter must also fit. The box's right
        # margin is not subtracted -- fit_margins() trims it where the next
        # element is closer, so a loosely drawn real box cannot shrink the value space.
        f["max_ink_width"] = f["max_width"] - jitter_max
        if f["max_ink_width"] < 0.5 * f["orig_ink_width"]:
            raise SynthError(f"'{name}': measured max width ({f['max_ink_width']} px) is far below the "
                             f"original text width; check the erase step / obstacle settings")

    # Regions a box must never cover: the strong obstacles above, plus the
    # labels found right next to each value in the ink step (small smart-card
    # labels are too faint for a global threshold on dim photos, but they are
    # exactly the pixels a box could reach). Ink that was erased is not a label.
    forbidden = obstacles | dilate(foreign_all & ~dilate(erased_all, 1), 1)
    return blank, fields, forbidden


# =============================================================================
# Step 2: fonts
# =============================================================================
def check_raqm():
    if not features.check("raqm"):
        raise SynthError(RAQM_HELP)


@lru_cache(maxsize=None)
def load_font(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=index, layout_engine=ImageFont.Layout.RAQM)


def lang_of(script: str) -> str:
    return "bn" if script == "bn" else "en"


def render_mask(lines: list[str], font, lang: str, line_advance: int = 0):
    """Render text (one or more lines) as an 8-bit alpha mask on a tight canvas.

    Returns (mask, (ax, ay)): (ax, ay) is where the first line's left/baseline
    anchor sits in the canvas.
    """
    boxes = [font.getbbox(t, anchor="ls", language=lang) for t in lines]
    left = min(b[0] for b in boxes)
    top = min(b[1] + i * line_advance for i, b in enumerate(boxes))
    right = max(b[2] for b in boxes)
    bottom = max(b[3] + i * line_advance for i, b in enumerate(boxes))
    pad = 4
    canvas = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 0)
    draw = ImageDraw.Draw(canvas)
    ax, ay = pad - left, pad - top
    for i, t in enumerate(lines):
        draw.text((ax, ay + i * line_advance), t, font=font, fill=255, anchor="ls", language=lang)
    return np.array(canvas), (ax, ay)


def pick_font_size(fcfg: dict, target_ref_h: int, cfg: dict):
    """Font size whose rendered reference height best matches the measured one.

    Returns (size, anchor_offset, rendered_ref_height); anchor_offset is the
    distance from the font baseline to the measured anchor row (matra/baseline).
    """
    script = fcfg["script"]
    ref = cfg["fonts"]["reference_text"][script]
    lo, hi = cfg["fonts"]["size_range"]
    best = None
    for size in range(lo, hi + 1):
        font = load_font(fcfg["font"], fcfg.get("font_index", 0), size)
        mask, (_, ay) = render_mask([ref], font, lang_of(script))
        m = text_metrics(mask > 127, script)
        err = abs(m["ref_height"] - target_ref_h)
        if best is None or err < best[0]:
            best = (err, size, m["anchor_y"] - ay, m["ref_height"])
        if m["ref_height"] > target_ref_h + 2:
            break
    return best[1], best[2], best[3]


def caption_font(cfg, size=20):
    return ImageFont.truetype(cfg["fonts"]["caption_font"], size)


def bangla_render_test(cfg: dict, out_path: Path):
    """Render the shaping test string with every Bangla font, one row each."""
    specs = [(cfg["fields"][n]["font"], cfg["fields"][n].get("font_index", 0))
             for n in CLASS_NAMES if cfg["fields"][n]["script"] == "bn"]
    specs += [(s["font"], s.get("font_index", 0)) for s in cfg["fonts"].get("bangla_preview_fonts", [])]
    specs = list(dict.fromkeys(specs))  # unique, keep order

    text, size = cfg["fonts"]["bangla_test_string"], cfg["fonts"]["bangla_test_size"]
    rows = []
    for path, index in specs:
        if not Path(path).is_file():
            print(f"  note: preview font not found, skipped: {path}")
            continue
        mask, _ = render_mask([text], load_font(path, index, size), "bn")
        rows.append((f"{Path(path).name} (index {index})", 255 - mask))

    cap = caption_font(cfg)
    width = max(r[1].shape[1] for r in rows) + 40
    height = sum(r[1].shape[0] + 50 for r in rows) + 20
    sheet = Image.new("L", (width, height), 255)
    draw, y = ImageDraw.Draw(sheet), 10
    for caption, img in rows:
        draw.text((20, y), caption, font=cap, fill=0)
        sheet.paste(Image.fromarray(img), (20, y + 30))
        y += img.shape[0] + 50
    sheet.save(out_path)


# =============================================================================
# Step 3: fake values
# =============================================================================
def format_date(d: dt.date, fmt: str) -> str:
    return (fmt.replace("%d", f"{d.day:02d}").replace("%m", f"{d.month:02d}")
               .replace("%B", MONTHS_LONG[d.month - 1]).replace("%b", MONTHS_SHORT[d.month - 1])
               .replace("%Y", f"{d.year:04d}"))


class FakeValues:
    """Made-up values built from the name parts in fake_names.yaml."""

    def __init__(self, names: dict, layout_cfg: dict, gen_cfg: dict, rng):
        self.names, self.lay, self.gen, self.rng = names, layout_cfg, gen_cfg, rng

    def _pick(self, items, k=1):
        idx = self.rng.choice(len(items), size=min(k, len(items)), replace=False)
        return [items[i] for i in idx]

    def community(self) -> str:
        w = self.names["community_weights"]
        keys = list(w)
        p = np.array([w[k] for k in keys], float)
        return keys[self.rng.choice(len(keys), p=p / p.sum())]

    def person(self, gender: str, n_words: int, community: str) -> tuple[str, str]:
        """A name of exactly n_words words -> (Bangla, English spelling)."""
        pool = self.names[gender][community]
        if n_words == 1:
            words = self._pick(pool["given"])
        else:
            slots = n_words - 1  # last word is the surname
            head, middle = [], []
            if slots >= 2 and pool.get("prefix") and self.rng.random() < 0.6:
                head, slots = self._pick(pool["prefix"]), slots - 1
            if slots >= 2 and pool.get("middle") and self.rng.random() < 0.5:
                middle, slots = self._pick(pool["middle"]), slots - 1
            words = head + self._pick(pool["given"], slots) + middle + self._pick(pool["surname"])
        bn = " ".join(w[0] for w in words)
        en = " ".join(w[1] for w in words)
        return bn, en.upper() if self.lay["english_name_all_caps"] else en

    def dob(self) -> str:
        y0, y1 = self.gen["dob_years"]
        start, end = dt.date(y0, 1, 1), dt.date(y1, 12, 31)
        d = start + dt.timedelta(days=int(self.rng.integers(0, (end - start).days + 1)))
        return format_date(d, self.lay["date_format"])

    def nid(self) -> str:
        digits = "".join(str(v) for v in self.rng.integers(0, 10, self.lay["nid_digits"]))
        parts, i = [], 0
        for g in self.lay["nid_groups"]:
            parts.append(digits[i:i + g])
            i += g
        return self.lay["nid_separator"].join(parts)


# =============================================================================
# Rendering (shared by Steps 2-4)
# =============================================================================
class FieldRenderer:
    """Renders values for each field with its chosen font, and checks widths."""

    def __init__(self, fields: dict, cfg: dict, layout_cfg: dict):
        self.f, self.cfg, self.lay = fields, cfg, layout_cfg
        self.alpha_thr = cfg["fonts"]["ink_alpha_threshold"]
        self._width_cache = {}

    def font(self, name):
        fnt = self.f[name]["font"]
        return load_font(fnt["path"], fnt["index"], fnt["size"])

    def mask(self, name, lines):
        adv = round(self.lay["wrap_line_spacing"] * self.f[name]["font"]["size"])
        return render_mask(lines, self.font(name), lang_of(self.f[name]["script"]), adv)

    def ink_width(self, name, text) -> int:
        key = (name, text)
        if key not in self._width_cache:
            mask, _ = self.mask(name, [text])
            cols = np.nonzero((mask > self.alpha_thr).any(axis=0))[0]
            self._width_cache[key] = int(cols[-1] - cols[0] + 1) if cols.size else 0
        return self._width_cache[key]

    def fit(self, name, text, allow_wrap):
        """Lines for `text` that fit the field's max width, or None.

        Never shrinks the font: a value either fits at the measured size, or
        (if wrapping is allowed) splits into two lines that each fit.
        """
        maxw = self.f[name]["max_ink_width"]
        if self.ink_width(name, text) <= maxw:
            return [text]
        if not (allow_wrap and name in self.lay["wrap_fields"]):
            return None
        words, best = text.split(), None
        for k in range(1, len(words)):
            a, b = " ".join(words[:k]), " ".join(words[k:])
            wa, wb = self.ink_width(name, a), self.ink_width(name, b)
            if wa <= maxw and wb <= maxw and (best is None or max(wa, wb) < best[0]):
                best = (max(wa, wb), [a, b])
        return best[1] if best else None


def pick_fitting(rend, fields, make, mode, gen_cfg, rng, allow_wrap):
    """Choose a value for one or more linked fields (e.g. name_bn + name_en).

    make(n_words) -> tuple of texts aligned with `fields`.
    mode "hard": the longest candidate that still fits on one line.
    mode "wrap": a candidate that needs (and fits) a second line.
    mode "normal": random length; if too wide, a shorter value is drawn.
    Returns {field: lines}.
    """
    lo, hi = gen_cfg["name_words"]
    if mode in ("wrap", "hard"):
        best = None
        for _ in range(gen_cfg["hard_candidates"]):
            texts = make(int(rng.integers(max(lo, 2), hi + 1)))
            if mode == "wrap":
                lays = [rend.fit(f, t, allow_wrap=True) for f, t in zip(fields, texts)]
                if all(lays) and any(len(l) == 2 for l in lays):
                    return dict(zip(fields, lays))
            else:
                if all(rend.fit(f, t, allow_wrap=False) for f, t in zip(fields, texts)):
                    score = max(rend.ink_width(f, t) / rend.f[f]["max_ink_width"]
                                for f, t in zip(fields, texts))
                    if best is None or score > best[0]:
                        best = (score, texts)
        if best:
            return {f: [t] for f, t in zip(fields, best[1])}
        if mode == "wrap":
            return pick_fitting(rend, fields, make, "hard", gen_cfg, rng, allow_wrap)
        # nothing long fits -> fall through to normal

    n = int(rng.integers(lo, hi + 1))
    for _ in range(gen_cfg["max_value_tries"]):
        texts = make(n)
        lays = [rend.fit(f, t, allow_wrap=False) for f, t in zip(fields, texts)]
        if all(lays):
            return dict(zip(fields, lays))
        n = max(lo, n - 1)  # too wide: pick a shorter value, never a smaller font
    raise SynthError(f"No value for {fields} fits the measured max width")


def choose_card_values(gen: FakeValues, rend: FieldRenderer, mode: str, cfg, layout_cfg, rng):
    """All 6 fake values for one card -> {field: [lines]}."""
    g = cfg["generation"]
    wrap_ok = bool(layout_cfg["names_may_wrap"])
    holder = "male" if rng.random() < 0.5 else "female"
    comm = gen.community()

    values = {}
    values.update(pick_fitting(rend, ("name_bn", "name_en"),
                               lambda n: gen.person(holder, n, comm), mode, g, rng, wrap_ok))
    values.update(pick_fitting(rend, ("guardian",),  # father or husband
                               lambda n: (gen.person("male", n, comm)[0],), mode, g, rng, wrap_ok))
    values.update(pick_fitting(rend, ("mother",),
                               lambda n: (gen.person("female", n, comm)[0],), mode, g, rng, wrap_ok))
    for name, make in (("dob", gen.dob), ("nid", gen.nid)):
        text = make()
        if not rend.fit(name, text, allow_wrap=False):
            raise SynthError(f"'{name}' value does not fit the measured max width")
        values[name] = [text]
    return values


# =============================================================================
# Step 4: render, box, label
# =============================================================================
def jitter(rng, lo, hi) -> int:
    return int(rng.integers(lo, hi + 1)) * (1 if rng.random() < 0.5 else -1)


def render_card(blank, rend: FieldRenderer, values: dict, cfg, rng, augment=True):
    """Draw all values onto a copy of the blank template.

    Returns (image, info): info[field] holds the imperfection values used and
    `area`, the full-size mask of pixels this field may have changed.
    """
    a = cfg["augment"]
    H, W = blank.shape[:2]
    out = blank.astype(np.float32)
    info = {}
    for name in CLASS_NAMES:
        f = rend.f[name]
        mask, (ax, ay) = rend.mask(name, values[name])
        cols = np.nonzero((mask > rend.alpha_thr).any(axis=0))[0]

        dx = jitter(rng, *a["jitter_px"]) if augment else 0
        dy = jitter(rng, *a["jitter_px"]) if augment else 0
        dark = float(rng.uniform(*a["darkness"])) if augment else 1.0
        sigma = float(rng.uniform(*a["blur_sigma"])) if augment else 0.0

        # Place so the ink starts at start_x and the font baseline sits where
        # the measured matra/baseline puts it.
        X0 = f["start_x"] + dx - int(cols[0])
        Y0 = f["font"]["draw_baseline_y"] + dy - ay
        h, w = mask.shape
        cx1, cy1, cx2, cy2 = max(0, X0), max(0, Y0), min(W, X0 + w), min(H, Y0 + h)
        kept = mask[cy1 - Y0:cy2 - Y0, cx1 - X0:cx2 - X0]
        clipped = int(kept.astype(np.int64).sum()) < int(mask.astype(np.int64).sum())

        color = np.clip(255 - (255 - np.array(f["color_rgb"][::-1], np.float32)) * dark, 0, 255)
        alpha = (kept.astype(np.float32) / 255.0)[..., None]
        region = out[cy1:cy2, cx1:cx2]
        region[:] = region * (1 - alpha) + color * alpha

        layer = np.zeros((H, W), bool)
        layer[cy1:cy2, cx1:cx2] = kept > 0
        pad = int(np.ceil(3 * sigma)) + 1
        area = dilate(layer, pad)
        if sigma > 0:
            # Very light blur, applied to the text area only.
            bx1, by1, bx2, by2 = mask_bbox(area)
            patch = out[by1:by2 + 1, bx1:bx2 + 1]
            blurred = cv2.GaussianBlur(patch, (0, 0), sigma)
            sel = area[by1:by2 + 1, bx1:bx2 + 1]
            patch[sel] = blurred[sel]

        info[name] = {"dx": dx, "dy": dy, "darkness": round(dark, 3),
                      "blur": round(sigma, 3), "clipped": clipped, "area": area}
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), info


def fit_margins(ink, margins, forbidden, gap: int = 1):
    """Box = ink bbox + margins, with any side's margin cut back where a label,
    photo or border is closer than that margin. The ink itself is never cut;
    if it touches a forbidden pixel, check_sample() drops the sample."""
    H, W = forbidden.shape
    x1, y1, x2, y2 = ink  # inclusive
    # The image border counts as forbidden too: keep at least 1 px inside it.
    L, T = min(margins["left"], x1 - 1), min(margins["top"], y1 - 1)
    R, B = min(margins["right"], W - 2 - x2), min(margins["bottom"], H - 2 - y2)
    L, T, R, B = max(0, L), max(0, T), max(0, R), max(0, B)
    def shrink(strip_hits, margin):
        hits = np.nonzero(strip_hits)[0]
        return margin if hits.size == 0 else max(0, int(hits[0]) - gap)

    # Each strip is ordered moving away from the ink. Left/right first, over
    # the ink's own rows; then top/bottom over the trimmed width only, so an
    # element beside the text (e.g. the card border) cannot cut the margin
    # below it.
    R = shrink(forbidden[y1:y2 + 1, x2 + 1:min(W, x2 + 1 + R)].any(axis=0), R)
    L = shrink(forbidden[y1:y2 + 1, max(0, x1 - L):x1].any(axis=0)[::-1], L)
    xa, xb = x1 - L, x2 + R + 1
    B = shrink(forbidden[y2 + 1:min(H, y2 + 1 + B), xa:xb].any(axis=1), B)
    T = shrink(forbidden[max(0, y1 - T):y1, xa:xb].any(axis=1)[::-1], T)
    return x1 - L, y1 - T, x2 + R + 1, y2 + B + 1


def boxes_from_changes(img, blank, info, fields, forbidden, cfg):
    """Boxes from the pixels that actually changed, plus the Step 1 margins."""
    c = cfg["checks"]
    diff = np.abs(img.astype(np.int16) - blank.astype(np.int16)).max(axis=2)
    boxes, visible = {}, {}
    for name in CLASS_NAMES:
        area = info[name]["area"]
        ink = mask_bbox((diff > c["ink_change_threshold"]) & area)
        visible[name] = mask_bbox((diff > c["any_change_threshold"]) & area)
        if ink is None:
            boxes[name] = None
            continue
        boxes[name] = fit_margins(ink, fields[name]["margins"], forbidden)
    return boxes, visible


def check_sample(boxes, visible, info, forbidden, cfg) -> list[str]:
    """All checks for one output. Returns a list of failure reasons (empty = pass)."""
    c = cfg["checks"]
    H, W = forbidden.shape
    reasons = []
    if len(boxes) != len(CLASS_NAMES) or any(b is None for b in boxes.values()):
        return ["not exactly 6 boxes"]
    for name, (x1, y1, x2, y2) in boxes.items():
        if info[name]["clipped"]:
            reasons.append(f"value clipped at image edge ({name})")
        if x1 <= 0 or y1 <= 0 or x2 >= W or y2 >= H:
            reasons.append(f"box touches image edge ({name})")
        if y2 - y1 < c["min_box_height_px"]:
            reasons.append(f"box too small ({name})")
        region = forbidden[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        if int(region.sum()) > c["forbidden_px_tolerance"]:
            reasons.append(f"box overlaps label/photo ({name})")
        v = visible[name]
        if v is None or v[0] < x1 or v[1] < y1 or v[2] >= x2 or v[3] >= y2:
            reasons.append(f"text not fully inside box ({name})")
    names = list(boxes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if iou(boxes[names[i]], boxes[names[j]]) > c["max_iou"]:
                reasons.append(f"boxes overlap ({names[i]}/{names[j]})")
    return reasons


def yolo_lines(boxes, W, H) -> str:
    out = []
    for cid, name in enumerate(CLASS_NAMES):
        x1, y1, x2, y2 = boxes[name]
        out.append(f"{cid} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                   f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return "\n".join(out) + "\n"


def draw_boxes(img, boxes):
    vis = img.copy()
    for cid, name in enumerate(CLASS_NAMES):
        x1, y1, x2, y2 = boxes[name]
        color = PREVIEW_COLORS[cid]
        cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        cv2.putText(vis, f"{cid} {name}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def font_compare(img, blank, rend, values, cfg, out_path: Path):
    """Original value crop next to a fake value rendered with the chosen font/size."""
    fake, info = render_card(blank, rend, values, cfg, np.random.default_rng(0), augment=False)
    H, W = img.shape[:2]
    cap = caption_font(cfg, 18)
    rows = []
    for name in CLASS_NAMES:
        f = rend.f[name]
        x1, y1, x2, y2 = f["orig_box"]
        ya, yb = max(0, y1 - 10), min(H, y2 + 10)
        fx2 = max(x2, mask_bbox(info[name]["area"])[2] + 10)
        left = cv2.cvtColor(img[ya:yb, x1:x2], cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(fake[ya:yb, x1:min(W, fx2)], cv2.COLOR_BGR2RGB)
        fnt = f["font"]
        rows.append((f"{name}: original  |  fake ({Path(fnt['path']).name}, {fnt['size']} px)", left, right))

    width = max(l.shape[1] + r.shape[1] for _, l, r in rows) + 60
    height = sum(l.shape[0] + 40 for _, l, _ in rows) + 10
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw, y = ImageDraw.Draw(sheet), 5
    for caption, left, right in rows:
        draw.text((10, y), caption, font=cap, fill=(0, 0, 0))
        sheet.paste(Image.fromarray(left), (10, y + 26))
        sheet.paste(Image.fromarray(right), (10 + left.shape[1] + 30, y + 26))
        y += left.shape[0] + 40
    sheet.save(out_path)


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Task B field-content synthesis for one NID card.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    layout = cfg["layout"]
    cfg, layout_cfg = layout_settings(cfg, layout)
    gen_cfg = cfg["generation"]

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SynthError(f"Could not read image: {args.image}")
    H, W = img.shape[:2]
    cfg = scale_pixel_settings(cfg, W)
    boxes_orig = read_yolo_labels(args.label, W, H)
    source = re.sub(r"_(front|back)$", "", Path(args.image).stem)  # nid_0006_front -> nid_0006
    prefix = f"{source}_{layout}"

    root = cfg_path(cfg, cfg["paths"]["out_root"])
    d_tmpl, d_prev = root / "templates", root / "preview"
    d_img, d_lbl = root / "rendered" / "images", root / "rendered" / "labels"
    for d in (d_tmpl, d_prev, d_img, d_lbl):
        d.mkdir(parents=True, exist_ok=True)

    # Checked first so a missing libraqm fails fast (it is Step 2.1).
    check_raqm()

    # ---- Step 1: blank template ------------------------------------------------
    print(f"[1] Blank template for {Path(args.image).name} (layout '{layout}')")
    blank, fields, forbidden = make_blank_template(img, boxes_orig, cfg, layout_cfg)
    tmpl_png, tmpl_json = d_tmpl / f"{prefix}_template.png", d_tmpl / f"{prefix}_template.json"
    cv2.imwrite(str(tmpl_png), blank)
    for name, f in fields.items():
        print(f"    {name:9s} {f['erase_method']:10s} start=({f['start_x']},{f['start_y']}) "
              f"h={f['text_height']} max_w={f['max_width']}")

    # ---- Step 2: fonts ---------------------------------------------------------
    print("[2] Fonts")
    bangla_render_test(cfg, d_prev / "bangla_render_test.png")
    assign_fonts(fields, cfg, layout_cfg=layout_cfg)

    meta = {"source_image": Path(args.image).name, "source_card": source, "layout": layout,
            "image_size": [W, H], "class_names": list(CLASS_NAMES), "fields": fields}
    tmpl_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    names = yaml.safe_load(cfg_path(cfg, cfg["paths"]["names_file"]).read_text(encoding="utf-8"))
    rend = FieldRenderer(fields, cfg, layout_cfg)
    preview_rng = np.random.default_rng(gen_cfg["seed"] + 1)
    preview_values = choose_card_values(FakeValues(names, layout_cfg, gen_cfg, preview_rng),
                                        rend, "normal", cfg, layout_cfg, preview_rng)
    font_compare(img, blank, rend, preview_values, cfg, d_prev / "font_compare.png")

    # ---- Steps 3 + 4: fake values, render, label, check ------------------------
    print("[3-4] Rendering samples")
    for d, pattern in ((d_img, "*.png"), (d_lbl, "*.txt"), (d_prev, "*_boxes.png")):
        for old in d.glob(f"{prefix}_ren_clean_{pattern}"):
            old.unlink()

    n_target = gen_cfg["num_samples"]
    max_attempts = n_target * (gen_cfg["refill_max_factor"] if gen_cfg["refill_dropped"] else 1)
    main_rng = np.random.default_rng(gen_cfg["seed"])
    n_hard = round(gen_cfg["hard_case_fraction"] * n_target)
    hard_set = set(main_rng.choice(n_target, size=n_hard, replace=False).tolist())
    wrap_on = bool(layout_cfg["names_may_wrap"])

    manifest, drops, kept, attempt = [], Counter(), 0, 0
    while attempt < max_attempts and kept < n_target:
        sample_seed = gen_cfg["seed"] * 1_000_003 + attempt
        rng = np.random.default_rng(sample_seed)
        if attempt < n_target:
            hard = attempt in hard_set
        else:
            hard = rng.random() < gen_cfg["hard_case_fraction"]
        mode = "hard" if hard else ("wrap" if wrap_on and rng.random() < gen_cfg["wrap_fraction"] else "normal")
        attempt += 1

        values = choose_card_values(FakeValues(names, layout_cfg, gen_cfg, rng),
                                    rend, mode, cfg, layout_cfg, rng)
        out, info = render_card(blank, rend, values, cfg, rng)
        boxes, visible = boxes_from_changes(out, blank, info, fields, forbidden, cfg)
        reasons = check_sample(boxes, visible, info, forbidden, cfg)
        if reasons:
            drops.update(reasons)
            continue

        kept += 1
        stem = f"{prefix}_ren_clean_{kept:02d}"
        cv2.imwrite(str(d_img / f"{stem}.png"), out)
        (d_lbl / f"{stem}.txt").write_text(yolo_lines(boxes, W, H))
        cv2.imwrite(str(d_prev / f"{stem}_boxes.png"), draw_boxes(out, boxes))

        row = {"file": f"{stem}.png", "label_file": f"{stem}.txt", "source_card": source,
               "layout": layout, "seed": gen_cfg["seed"], "sample_seed": sample_seed,
               "attempt": attempt, "hard_case": hard, "wrapped": any(len(v) > 1 for v in values.values())}
        for name in CLASS_NAMES:
            i, fnt = info[name], fields[name]["font"]
            row.update({f"{name}_value": " ".join(values[name]),
                        f"{name}_font": Path(fnt["path"]).name, f"{name}_size": fnt["size"],
                        f"{name}_dx": i["dx"], f"{name}_dy": i["dy"],
                        f"{name}_darkness": i["darkness"], f"{name}_blur": i["blur"]})
        manifest.append(row)

    manifest_path = root / "rendered" / "manifest.csv"
    if manifest:
        # utf-8-sig so Excel shows the Bangla values correctly.
        with open(manifest_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
            w.writeheader()
            w.writerows(manifest)

    # ---- Summary ---------------------------------------------------------------
    dropped = attempt - kept
    print("\nSummary")
    print(f"  attempted : {attempt}")
    print(f"  made      : {kept}  ({sum(r['hard_case'] for r in manifest)} hard cases, "
          f"{sum(r['wrapped'] for r in manifest)} wrapped)")
    print(f"  dropped   : {dropped}")
    for reason, count in drops.most_common():
        print(f"    - {reason}: {count}")
    print(f"  template  : {tmpl_png}")
    print(f"  outputs   : {d_img}")
    print(f"  manifest  : {manifest_path if manifest else '(none, nothing passed)'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SynthError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
