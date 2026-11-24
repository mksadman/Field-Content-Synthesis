#!/usr/bin/env python3
"""
synthesize_back.py -- synthetic address values on the BACK side of NID cards.

For every back-side card image:
  1. Read the address box (x0 y0 x1 y1) from the coords folder and rotate the
     card upright if the coords say it was scanned upside down.
  2. Split the ink inside the box into text lines and measure them: matra-to-
     baseline height, line pitch, start x, continuation-line indent, colour.
  3. Erase the whole address block ("ঠিকানা: ..."), filling plain paper with
     its own grain and inpainting patterned (smart-card) backgrounds.
  4. Draw made-up addresses (fake_addresses.yaml) with the same label, size,
     spacing and wrapping rules, plus small imperfections (jitter, darkness,
     blur), and write the outputs below.

The real address on the card is never read, copied or printed.

Outputs (under back.out_dir, default synthetic/back):
  images/<id>.png       full card (original orientation)
  strips/<id>.png       address crop, upright (same layout as the source "strips")
  labels/<id>.gt.txt    address text, same header format as the source data/*.gt.txt
  labels/<id>.txt       YOLO box, class 0 = address
  templates/<card>.png  the erased card (only with --save-blank)
  preview/<id>.png      box overlay (only with --preview)
  manifest.csv          one row per sample; skipped.csv lists every dropped sample
                        (sample = "all" when the whole card failed) and why

The script writes images/, strips/ and labels/ flat, and leaves manifest `split`
empty for new samples. synthetic/back was split by source card into
{images,strips,labels}/{train,test}/ afterwards; split.csv (id, split, source_card)
records the assignment, and --append keeps the `split` of existing rows.

Usage
  python "image synthesis/synthesize_back.py" --config config.yaml -limit 5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthesize_one import (  # noqa: E402  (reuse the front-side building blocks)
    SynthError, cfg_path, check_raqm, darkness_map, detect_value_ink, dilate,
    ellipse, load_config, load_font, mask_bbox, pick_font_size, render_mask,
    scale_pixel_settings, text_metrics, to_gray,
)

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MANIFEST_FIELDS = ["id", "split", "source_card", "nid_version", "font", "font_size", "lines",
                   "rotated_180", "skew_deg", "x1", "y1", "x2", "y2", "image_w", "image_h",
                   "dx", "dy", "darkness", "blur", "erase_method", "text", "text_lines"]


# =============================================================================
# Inputs
# =============================================================================
def imread(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SynthError(f"Cannot read image: {path}")
    return img


def imwrite(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise SynthError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def read_coords(path: Path) -> dict:
    """Parse '<key>: <value>' lines of a coords file."""
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    for k in ("x0", "y0", "x1", "y1"):
        if k not in out:
            raise SynthError(f"{path.name}: missing '{k}'")
    return out


def read_version(gt_dir: Path | None, stem: str) -> str:
    """nid_version (smart/regular/unknown) from the card's gt header only."""
    if gt_dir is None:
        return "unknown"
    p = gt_dir / f"{stem}.gt.txt"
    if not p.is_file():
        return "unknown"
    m = re.search(r"^#\s*nid_version:\s*(\w+)", p.read_text(encoding="utf-8-sig"), re.M)
    return m.group(1) if m else "unknown"


def discover(cards_dir: Path, coords_dir: Path) -> tuple[list[tuple[Path, Path]], list[dict]]:
    cards, skipped = [], []
    for img in sorted(p for p in cards_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        c = coords_dir / f"{img.stem}.txt"
        if c.is_file():
            cards.append((img, c))
        else:
            skipped.append({"card": img.stem, "sample": "all", "reason": "no coords file"})
    return cards, skipped


# =============================================================================
# Fake addresses
# =============================================================================
class FakeAddress:
    def __init__(self, data: dict, rng: np.random.Generator):
        self.d, self.rng = data, rng
        self.p = data["probabilities"]

    def pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def chance(self, key: str) -> bool:
        return self.rng.random() < self.p[key]

    def num(self, lo: int, hi: int) -> str:
        return str(int(self.rng.integers(lo, hi + 1))).translate(BN_DIGITS)

    def house(self) -> str:
        out = self.pick(self.d["house_formats"])
        while "{L}" in out:
            out = out.replace("{L}", self.pick(self.d["house_letters"]), 1)
        while "{k}" in out:
            out = out.replace("{k}", self.num(1, 9), 1)
        return out.replace("{n}", self.num(1, 999)).replace("{m}", self.num(1, 30))

    def village(self) -> str:
        r = self.rng.random()
        base = self.pick(self.d["village_base"])
        if r < self.d["road_fraction"]:
            return self.pick(self.d["road_formats"]).replace("{n}", self.num(1, 30)).replace("{name}", base)
        if r < self.d["road_fraction"] + self.d["bari_fraction"]:
            return f"{self.pick(self.d['bari_families'])} বাড়ী"
        return f"{self.pick(self.d['village_prefix'])}{base}{self.pick(self.d['village_suffix'])}"

    def make(self) -> str:
        dist = self.pick(self.d["districts"])
        upazila = self.pick(dist["upazilas"])
        parts = []
        if self.chance("house"):
            sep = " " if self.chance("house_no_colon") else ": "
            parts.append(f"বাসা/হোল্ডিং{sep}{self.house()}")
        if self.chance("village") or not parts:
            parts.append(f"গ্রাম/রাস্তা: {self.village()}")
        po = self.pick([upazila.replace(" সদর", ""), self.pick(self.d["village_base"])])
        if self.chance("post_code"):
            po += " - " + self.num(*dist["post_code"])
        parts.append(f"ডাকঘর: {po}")
        parts.append(upazila)
        if dist.get("city_corp") and self.chance("city_corp"):
            parts.append(self.pick(dist["city_corp"]))
        elif self.chance("pourashava"):
            parts.append(f"{upazila.replace(' সদর', '')} পৌরসভা")
        parts.append(dist["name"])
        return f"{self.d['label']} " + ", ".join(parts)


# =============================================================================
# Measuring the original block
# =============================================================================
def rule_lines(strong: np.ndarray, box, version: str) -> np.ndarray:
    """Printed rules in and around the box that must survive the erase.

    - Horizontal table lines (old cards): rows holding a run far longer than
      any word's matra; every ink pixel on those rows is protected, so a rule
      broken into pieces is kept whole.
    - Vertical borders (old cards, often dashed): columns near the box's left/
      right edge whose ink continues above and below the box.
    `strong` is a full-size ink mask. Returns a full-size mask.
    """
    H, W = strong.shape
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    out = np.zeros((H, W), bool)

    sub = strong[y0:y1, x0:x1].astype(np.uint8)
    hor = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.45 * w)), 1)))
    rows = np.nonzero(hor.any(axis=1))[0]
    for r in rows:
        r0, r1 = max(0, y0 + r - 1), min(H, y0 + r + 2)
        out[r0:r1, x0:x1] |= strong[r0:r1, x0:x1]

    if version == "regular":
        band = max(3, int(0.03 * W))
        span = max(10, 2 * h)
        outside = np.r_[max(0, y0 - span):y0, y1:min(H, y1 + span)]
        if outside.size:
            for xa, xb in ((max(0, x0 - band), min(W, x0 + band)), (max(0, x1 - band), min(W, x1 + band))):
                cov = strong[outside, xa:xb].mean(axis=0)
                for c in np.nonzero(cov >= 0.35)[0]:
                    cx = xa + c
                    out[y0:y1, max(0, cx - 1):cx + 2] |= strong[y0:y1, max(0, cx - 1):cx + 2]
    return dilate(out, 1)


def split_lines(ink: np.ndarray, box, max_valley: float = 0.4) -> list[tuple[int, int]]:
    """Row bands (y_top, y_bottom) of the text lines inside the box.

    Wrapped Bangla lines can touch (vowel signs above the matra reach into the
    line above), so lines are split at deep valleys of the row profile rather
    than at empty rows: a valley is a split when it is at most `max_valley` of
    the lower of the two peaks either side, and both sides are line-sized.
    """
    x0, y0, x1, y1 = box
    prof = ink[y0:y1, x0:x1].sum(axis=1).astype(np.float32)
    if prof.max() <= 0:
        return []
    # Light smoothing on big scans only: on small cards a line is ~10 rows and
    # smoothing would fill the valleys between lines.
    k = int((y1 - y0) * 0.006)
    if k > 0:
        prof = np.convolve(prof, np.ones(2 * k + 1) / (2 * k + 1), mode="same")

    def trim(a, b):
        rows = np.nonzero(prof[a:b + 1] > 0.08 * prof[a:b + 1].max())[0]
        return (a + int(rows[0]), a + int(rows[-1])) if rows.size else None

    def split(a, b):
        seg = trim(a, b)
        if seg is None:
            return []
        a, b = seg
        best = None
        for v in range(a + 2, b - 1):
            if not (prof[v] <= prof[v - 1] and prof[v] <= prof[v + 1]):
                continue
            ratio = prof[v] / max(1e-6, min(prof[a:v].max(), prof[v + 1:b + 1].max()))
            if ratio <= max_valley and (best is None or ratio < best[0]):
                left, right = trim(a, v), trim(v + 1, b)
                if left and right:
                    hl, hr = left[1] - left[0] + 1, right[1] - right[0] + 1
                    if min(hl, hr) >= max(4, 0.4 * max(hl, hr)):
                        best = (ratio, v)
        if best is None:
            return [(a, b)]
        return split(a, best[1]) + split(best[1] + 1, b)

    return [(y0 + a, y0 + b) for a, b in split(0, len(prof) - 1)]


def measure_block(img, box, tc, version, region=None):
    """Detect and measure the address ink.

    `region` (optional bool mask) is the exact address area when it is not the
    axis-aligned `box` (the coords box after deskewing); `box` must enclose it.
    Returns (lines, erase_mask, color_bgr, rules): `lines` holds one
    text_metrics dict per text line, top to bottom.
    """
    H, W = img.shape[:2]
    dark = darkness_map(to_gray(img), tc["bg_kernel_px"])
    dark_color = np.max([darkness_map(np.ascontiguousarray(img[..., c]), tc["bg_kernel_px"])
                         for c in range(3)], axis=0)
    x0, y0, x1, y1 = box
    if region is None:
        region = np.zeros((H, W), bool)
        region[y0:y1, x0:x1] = True
    thr = ink_threshold(dark, region, tc)
    rough = (dark > thr) & region
    rules = rule_lines(dark > thr, box, version)
    bands = split_lines(rough & ~rules, box)
    if not bands:
        raise SynthError("no ink inside the address box")

    # Rules must survive: hide them from the value-ink detection and the erase.
    dark_t, dark_color_t = dark.copy(), dark_color.copy()
    dark_t[rules] = 0
    dark_color_t[rules] = 0
    lines, erase_all, keep_all = [], np.zeros((H, W), bool), np.zeros((H, W), bool)
    for top, bot in bands:
        lbox = (x0, max(y0, top), x1, min(y1, bot + 1))
        if lbox[3] - lbox[1] < 3:
            continue
        keep, erase, _ = detect_value_ink(dark_t, dark_color_t, lbox, tc)
        # Measure this line only: on small cards the detection pad reaches the
        # neighbouring lines, which would inflate the measured height.
        own = np.zeros((H, W), bool)
        own[max(0, top - 1):bot + 2] = True
        keep &= ~rules & own
        if keep.sum() < tc["min_component_px"] * 3:
            continue
        lines.append(text_metrics(keep, "bn"))
        erase_all |= erase & ~rules
        keep_all |= keep
    if not lines:
        raise SynthError("address ink too faint to measure")
    lines.sort(key=lambda m: m["anchor_y"])

    px = img[keep_all].reshape(-1, 3)
    lum = to_gray(px.reshape(-1, 1, 3)).ravel()
    darkest = px[lum <= np.quantile(lum, tc["darkest_ink_fraction"])]

    # Erase mask for the whole block. The per-line detection above is built
    # for one-line values and leaves vowel signs that touch a neighbouring
    # line or a rule; here every dark component lying mostly inside the box
    # is value ink (the box holds nothing but the address), plus its halo.
    ref_h = float(np.median([m["ref_height"] for m in lines]))
    pad = max(2, int(round(0.6 * ref_h)))
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    strong = np.zeros((H, W), np.uint8)
    strong[Y0:Y1, X0:X1] = (dark[Y0:Y1, X0:X1] > thr) & ~rules[Y0:Y1, X0:X1]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
    in_px = np.bincount(lab[region], minlength=n)
    ok = (in_px >= 0.5 * stats[:, 4]) & (stats[:, 4] >= 2)
    ok[0] = False
    value = ok[lab] | keep_all
    halo = (dark_color > tc["halo_min_contrast"]) & dilate(value, tc["halo_px"])
    erase = value | halo | erase_all
    r = tc["erase_close_px"]
    if r > 0:
        erase = cv2.morphologyEx(erase.astype(np.uint8), cv2.MORPH_CLOSE, ellipse(r)).astype(bool)
    erase &= ~rules
    return lines, erase, np.median(darkest, axis=0), rules


def ink_threshold(dark: np.ndarray, region: np.ndarray, tc: dict) -> float:
    vals = dark[region].reshape(-1, 1)
    otsu, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(tc["min_ink_contrast"] * 0.5, float(otsu))


def estimate_skew(ink: np.ndarray, box, max_deg: float, step: float) -> float:
    """Angle (degrees, cv2 sign convention) that makes the text lines in the
    box horizontal: the rotation whose row profile is sharpest."""
    x0, y0, x1, y1 = box
    py = int(0.15 * (y1 - y0)) + 2
    sub = ink[max(0, y0 - py):y1 + py, x0:x1].astype(np.float32)
    if sub.sum() == 0:
        return 0.0
    h, w = sub.shape
    best = (-1.0, 0.0)
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        r = cv2.warpAffine(sub, cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0), (w, h),
                           flags=cv2.INTER_LINEAR)
        score = float((r.sum(axis=1) ** 2).sum())
        if score > best[0]:
            best = (score, float(a))
    return best[1]


class Frame:
    """Small rotation that straightens the address block (identity if none).
    Measuring and drawing happen in the straight frame; only masks and the
    drawn text layer are mapped back, so card pixels are never resampled."""

    def __init__(self, angle: float, center, shape):
        self.angle, self.H, self.W = angle, shape[0], shape[1]
        self.M = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.Minv = cv2.getRotationMatrix2D(center, -angle, 1.0)

    def _warp(self, a, M, interp, border):
        if self.angle == 0.0:
            return a.copy()
        return cv2.warpAffine(a, M, (self.W, self.H), flags=interp, borderMode=border)

    def to_straight(self, img):
        return self._warp(img, self.M, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)

    def mask_to_straight(self, m):
        return self._warp(m.astype(np.uint8), self.M, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT) > 0

    def mask_back(self, m):
        return self._warp(m.astype(np.uint8), self.Minv, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT) > 0

    def layer_back(self, layer):
        return self._warp(layer, self.Minv, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)


def fill_erased(img: np.ndarray, erase: np.ndarray, rules: np.ndarray, tc: dict, rng) -> str:
    """Replace the `erase` pixels (in place) with the card's own background.

    The background is a per-channel grayscale closing (removes every stroke
    thinner than the kernel but keeps paper tone, gradients and the broad
    smart-card pattern), smoothed, shifted to match the untouched ring around
    the block, plus noise matching the paper's grain, feathered at the edge.
    Large text blocks smear under cv2.inpaint; this does not.
    """
    mask = dilate(erase, tc["erase_dilate_px"]) & ~rules
    if not mask.any():
        return "none"
    k = tc["bg_kernel_px"]
    # Radius k (not k/2): dense Bangla lines otherwise leave a faint shadow.
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ellipse(k)).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), max(1.0, k / 6))

    dark = darkness_map(to_gray(img), k)
    # Paper samples: around the block but clear of any ink (and its edges),
    # otherwise neighbouring letters inflate the grain into visible speckle.
    inky = dilate(dark >= tc["halo_min_contrast"], 3)
    ring = dilate(mask, tc["bg_ring_px"] * 2) & ~dilate(mask, 1) & ~inky & ~rules
    src = img.astype(np.float32)
    if ring.sum() >= 20:
        shift = np.median(src[ring], axis=0) - np.median(bg[ring], axis=0)
        hp = (src - cv2.GaussianBlur(src, (0, 0), 1.5))[ring]
        grain = 1.4826 * np.median(np.abs(hp - np.median(hp, axis=0)), axis=0)   # robust std
        grain = np.minimum(grain, 6.0)
    else:
        shift, grain = np.zeros(3, np.float32), np.full(3, 2.0, np.float32)
    # One noise field for all channels: paper grain is mostly luminance.
    noise = rng.normal(0.0, 1.0, img.shape[:2] + (1,)).astype(np.float32)
    fill = bg + shift + noise * grain * tc["plain_fill_noise_scale"]

    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)
    alpha = np.maximum(alpha, mask.astype(np.float32))[..., None]
    img[:] = np.clip(np.rint(src * (1 - alpha) + fill * alpha), 0, 255).astype(np.uint8)
    return "closing_fill"


# =============================================================================
# Layout + rendering
# =============================================================================
def ink_width(font, text: str) -> int:
    b = font.getbbox(text, anchor="ls", language="bn")
    return b[2] - b[0]


def wrap(text: str, font, widths: list[int], max_lines: int) -> list[str] | None:
    """Greedy word wrap; line i may be at most widths[min(i, -1)] px wide.
    Returns None if the text needs more than max_lines lines."""
    lines, cur = [], ""
    for w in text.split(" "):
        trial = f"{cur} {w}" if cur else w
        if cur and ink_width(font, trial) > widths[min(len(lines), len(widths) - 1)]:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    if len(lines) > max_lines:
        return None
    if any(ink_width(font, t) > widths[min(i, len(widths) - 1)] for i, t in enumerate(lines)):
        return None   # a single word wider than the line
    return lines


def blank_rows(dark, x_lo, x_hi, y_lo, y_hi, thr, max_frac) -> bool:
    """True if at most `max_frac` of the columns in the band hold ink, so a
    card border or a speck does not count, but any text (labels) does."""
    if y_lo < 0 or y_hi > dark.shape[0]:
        return False
    region = dark[y_lo:y_hi, x_lo:x_hi] > thr
    if region.size == 0:
        return False
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return float(region.any(axis=0).mean()) <= max_frac


class Layout:
    """Where and how big to draw, measured once per card and font."""

    def __init__(self, lines, box, version, font_cfg, cfg, bc, dark, gray, paper, tc):
        self.ref_h = int(round(np.median([m["ref_height"] for m in lines])))
        fcfg = {"script": "bn", "font": font_cfg["font"], "font_index": font_cfg.get("font_index", 0)}
        self.size, self.anchor_off, _ = pick_font_size(fcfg, self.ref_h, cfg)
        self.font = load_font(fcfg["font"], fcfg["font_index"], self.size)
        self.font_path = font_cfg["font"]
        # Card fonts are narrower than the rendering fonts at the same height.
        self.x_scale = float(bc.get("x_scale", {}).get(version, 1.0))

        matras = [m["anchor_y"] for m in lines]
        self.pitch = (float(np.median(np.diff(matras))) if len(lines) > 1
                      else bc["line_pitch_fallback"] * self.ref_h)
        self.first_matra = matras[0]
        self.left = lines[0]["left"]
        # Right limit: the original text's right edge, extended towards the
        # box edge only up to the first printed element (lotus, photo, card
        # border) on the erased card, minus a gap.
        text_right = max(m["right"] for m in lines)
        self.right = text_right
        if box[2] > text_right + 1:
            top, bot = min(m["top"] for m in lines), max(m["bottom"] for m in lines)
            sl = (slice(top, bot + 1), slice(text_right + 1, box[2]))
            # Large solid shapes (the lotus) are not "darker than their own
            # surroundings", so also flag pixels far darker than the paper.
            band = (dark[sl] > tc["min_ink_contrast"]) | (gray[sl] < paper - tc["min_ink_contrast"])
            band = band.astype(np.uint8)
            band = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            hits = np.nonzero(band.any(axis=0))[0]
            if hits.size:
                self.right = max(text_right, text_right + 1 + int(hits[0]) - int(0.6 * self.ref_h))
            else:
                self.right = box[2]

        if len(lines) > 1:
            indent = float(np.median([m["left"] for m in lines[1:]])) - self.left
            self.indent = int(indent) if indent > 0.5 * self.ref_h else 0
        elif version == "regular":
            # Old cards hang continuation lines under the value, after "ঠিকানা: ".
            self.indent = ink_width(self.font, bc["_label"] + " ক") - ink_width(self.font, "ক")
        else:
            self.indent = 0

        # Extra lines below the original ones, only where the card is blank.
        self.orig_lines = len(lines)
        self.max_lines = len(lines)
        for _ in range(int(bc["extra_lines"])):
            k = self.max_lines
            y_lo = int(self.first_matra + (k - 0.45) * self.pitch)
            y_hi = int(self.first_matra + (k + 0.75) * self.pitch)
            if not blank_rows(dark, self.left, self.right, y_lo, y_hi,
                              tc["min_ink_contrast"], bc["blank_below_max_ink"]):
                break
            self.max_lines += 1

    def widths(self) -> list[int]:
        """Line widths in unscaled font pixels (what wrap() measures)."""
        w = self.right - self.left
        return [int(w / self.x_scale), int((w - self.indent) / self.x_scale)]


def render_layer(shape, lay: Layout, text_lines, aug, rng):
    """Draw the lines as an alpha layer (0..1) in the straight frame.
    Returns (layer, params); params['clipped'] is True if ink left the image."""
    H, W = shape

    def jitter(lo, hi):
        return int(rng.integers(lo, hi + 1)) * (1 if rng.random() < 0.5 else -1)

    dx, dy = jitter(*aug["jitter_px"]), jitter(*aug["jitter_px"])
    dx = max(dx, 1 - lay.left)          # never push text off a card that starts at x=0
    layer = np.zeros((H, W), np.float32)
    clipped = False
    for i, t in enumerate(text_lines):
        mask, (_, ay) = render_mask([t], lay.font, "bn")
        if abs(lay.x_scale - 1.0) > 1e-3:
            mask = cv2.resize(mask, (max(1, int(round(mask.shape[1] * lay.x_scale))), mask.shape[0]),
                              interpolation=cv2.INTER_AREA)
        cols = np.nonzero((mask > 40).any(axis=0))[0]
        start_x = lay.left + (lay.indent if i else 0)
        baseline = int(round(lay.first_matra + i * lay.pitch - lay.anchor_off))
        X0, Y0 = start_x + dx - int(cols[0]), baseline + dy - ay
        h, w = mask.shape
        cx1, cy1, cx2, cy2 = max(0, X0), max(0, Y0), min(W, X0 + w), min(H, Y0 + h)
        if cx2 <= cx1 or cy2 <= cy1:
            clipped = True
            continue
        kept = mask[cy1 - Y0:cy2 - Y0, cx1 - X0:cx2 - X0]
        clipped |= int(kept.astype(np.int64).sum()) < int(mask.astype(np.int64).sum())
        layer[cy1:cy2, cx1:cx2] = np.maximum(layer[cy1:cy2, cx1:cx2], kept.astype(np.float32) / 255.0)
    return layer, {"dx": dx, "dy": dy, "clipped": clipped}


def composite(blank, layer, color_bgr, aug, rng):
    """Blend the text layer onto `blank` with a darkness change and a light
    blur on the text area only. Returns (image, ink_mask, params)."""
    dark = float(rng.uniform(*aug["darkness"]))
    sigma = float(rng.uniform(*aug["blur_sigma"]))
    color = np.clip(255 - (255 - np.asarray(color_bgr, np.float32)) * dark, 0, 255)
    alpha = np.clip(layer, 0, 1)[..., None]
    out = blank.astype(np.float32) * (1 - alpha) + color * alpha
    ink = layer > 0.15
    if sigma > 0 and ink.any():
        area = dilate(ink, int(np.ceil(3 * sigma)) + 1)
        bx1, by1, bx2, by2 = mask_bbox(area)
        patch = out[by1:by2 + 1, bx1:bx2 + 1]
        blurred = cv2.GaussianBlur(patch, (0, 0), sigma)
        sel = area[by1:by2 + 1, bx1:bx2 + 1]
        patch[sel] = blurred[sel]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), ink, {"darkness": round(dark, 3), "blur": round(sigma, 3)}


# =============================================================================
# Per card
# =============================================================================
def card_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")


def write_gt(path: Path, sid: str, version: str, source: str, text: str, n_lines: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# card_id: {sid}\n# side: back\n# nid_version: {version}\n"
        f"# label_status: synthetic\n# source_card: {source}\n# lines: {n_lines}\n\n{text}\n",
        encoding="utf-8")


def process_card(img_path, coords_path, cfg, bc, data, gt_dir, out_dir, args, rows, skipped) -> int:
    name = img_path.stem
    co = read_coords(coords_path)
    rotated = co.get("rotated_180", "no").lower().startswith("y")
    version = read_version(gt_dir, name)

    img = imread(img_path)
    if rotated:
        img = cv2.rotate(img, cv2.ROTATE_180)   # coords are on the upright image
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(float(co[k])) for k in ("x0", "y0", "x1", "y1"))
    box = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))
    if box[2] - box[0] < 10 or box[3] - box[1] < 5:
        raise SynthError(f"address box too small: {box}")

    start = args.sample_start
    sample_ids = range(start, start + args.samples)
    if args.append:   # never overwrite samples from an earlier run
        # images/ may have been split into images/train and images/test
        taken = [s for s in sample_ids if any((out_dir / "images").glob(f"**/{name}_s{s:02d}.png"))]
        if taken:
            raise SynthError(f"samples already exist (s{taken[0]:02d}...); raise --sample-start")

    cfgs = scale_pixel_settings(cfg, W)
    tc = cfgs["template"]
    # A later --sample-start gets its own random stream, so it never repeats the
    # addresses of s00.. (start 0 keeps the original stream).
    rng = np.random.default_rng(card_seed(bc["seed"], name if not start else f"{name}#{start}"))

    # Straighten a tilted block: measure and draw in the straight frame.
    angle = 0.0
    if bc.get("deskew_max_deg", 0) > 0:
        rect = np.zeros((H, W), bool)
        rect[box[1]:box[3], box[0]:box[2]] = True
        d0 = darkness_map(to_gray(img), tc["bg_kernel_px"])
        angle = estimate_skew(d0 > ink_threshold(d0, rect, tc), box,
                              bc["deskew_max_deg"], bc["deskew_step_deg"])
        if abs(angle) < bc["deskew_min_deg"]:
            angle = 0.0
    frame = Frame(angle, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), (H, W))
    region = np.zeros((H, W), bool)
    region[box[1]:box[3], box[0]:box[2]] = True
    s_box = box
    if angle:
        region = frame.mask_to_straight(region)
        bb = mask_bbox(region)
        s_box = (bb[0], bb[1], bb[2] + 1, bb[3] + 1)

    img_s = frame.to_straight(img)
    lines, erase_s, color, rules_s = measure_block(img_s, s_box, tc, version, region)
    erase, rules = frame.mask_back(erase_s), frame.mask_back(rules_s)
    blank = img.copy()
    method = fill_erased(blank, erase, rules, tc, rng)
    # Free space is judged on the erased card, in the straight frame.
    gray = to_gray(frame.to_straight(blank))
    dark = darkness_map(gray, tc["bg_kernel_px"])
    paper = float(np.median(gray[erase_s])) if erase_s.any() else float(np.median(gray))
    box = s_box
    if args.save_blank:
        imwrite(out_dir / "templates" / f"{name}.png", cv2.rotate(blank, cv2.ROTATE_180) if rotated else blank)

    fonts = bc["fonts"]
    weights = np.array([f.get("weight", 1.0) for f in fonts], float)
    weights /= weights.sum()
    layouts: dict[int, Layout] = {}
    gen = FakeAddress(data, rng)
    n_done = 0
    for s in sample_ids:
        if args.budget is not None and args.budget <= 0:
            break
        fi = int(rng.choice(len(fonts), p=weights))
        if fi not in layouts:
            layouts[fi] = Layout(lines, box, version, fonts[fi], cfgs, bc, dark, gray, paper, tc)
        lay = layouts[fi]

        text_lines = None
        for _ in range(bc["max_value_tries"]):
            text_lines = wrap(gen.make(), lay.font, lay.widths(), lay.max_lines)
            if text_lines:
                break
        def skip(reason):
            print(f"    sample {s}: {reason}; skipped")
            skipped.append({"card": name, "sample": s, "reason": reason})

        if not text_lines:
            skip(f"no address fits in {lay.max_lines} line(s) "
                 f"(text height {lay.ref_h}px, line width {lay.right - lay.left}px)")
            continue

        layer_s, prm = render_layer((H, W), lay, text_lines, cfg["augment"], rng)
        layer = frame.layer_back(layer_s)
        if prm["clipped"] or layer.sum() < 0.98 * layer_s.sum():
            skip("text ran off the image")
            continue
        out, ink, prm2 = composite(blank, layer, color, cfg["augment"], rng)
        prm.update(prm2)
        if not ink.any():
            skip("nothing drawn")
            continue

        m = bc["box_margin_px"]
        bx1, by1, bx2, by2 = mask_bbox(ink)
        bx1, by1, bx2, by2 = max(0, bx1 - m), max(0, by1 - m), min(W - 1, bx2 + m), min(H - 1, by2 + m)
        sid = f"{name}_s{s:02d}"

        # Strip: the box padded by part of a line pitch, always upright.
        pad = int(round(bc["strip_pad_ratio"] * lay.pitch))
        strip = out[max(0, by1 - pad):min(H, by2 + 1 + pad), max(0, bx1 - pad):min(W, bx2 + 1 + pad)]
        imwrite(out_dir / "strips" / f"{sid}.png", strip)

        full, (fx1, fy1, fx2, fy2) = out, (bx1, by1, bx2, by2)
        if rotated:   # back to the scan's own orientation
            full = cv2.rotate(out, cv2.ROTATE_180)
            fx1, fy1, fx2, fy2 = W - 1 - bx2, H - 1 - by2, W - 1 - bx1, H - 1 - by1
        imwrite(out_dir / "images" / f"{sid}.png", full)

        flat = " ".join(text_lines)
        write_gt(out_dir / "labels" / f"{sid}.gt.txt", sid, version, name, flat, len(text_lines))
        cx, cy = (fx1 + fx2 + 1) / 2 / W, (fy1 + fy2 + 1) / 2 / H
        bw, bh = (fx2 - fx1 + 1) / W, (fy2 - fy1 + 1) / H
        (out_dir / "labels" / f"{sid}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

        if args.preview:
            pv = full.copy()
            cv2.rectangle(pv, (fx1, fy1), (fx2, fy2), (0, 0, 255), max(1, W // 500))
            imwrite(out_dir / "preview" / f"{sid}.png", pv)

        rows.append({"id": sid, "source_card": name, "nid_version": version,
                     "font": Path(lay.font_path).name, "font_size": lay.size, "lines": len(text_lines),
                     "rotated_180": "yes" if rotated else "no", "skew_deg": round(angle, 2),
                     "x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2, "image_w": W, "image_h": H,
                     "dx": prm["dx"], "dy": prm["dy"], "darkness": prm["darkness"], "blur": prm["blur"],
                     "erase_method": method, "text": flat, "text_lines": " | ".join(text_lines)})
        n_done += 1
        if args.budget is not None:
            args.budget -= 1

    lay0 =next(iter(layouts.values()), None)
    detail = f", ref height {lay0.ref_h}px, max {lay0.max_lines} line(s)" if lay0 else ""
    print(f"  {name}: {version}, {len(lines)} original line(s){detail}, erase={method}, "
          f"{n_done}/{args.samples} samples")
    return n_done


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic address values for back-side NID cards.")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    ap.add_argument("-limit", "--limit", type=int, default=None,
                    help="only process the first N cards (sorted by file name)")
    ap.add_argument("--samples", type=int, default=None, help="samples per card (default: back.samples_per_card)")
    ap.add_argument("--cards-dir", default=None, help="override back.cards_dir")
    ap.add_argument("--coords-dir", default=None, help="override back.coords_dir")
    ap.add_argument("--out", default=None, help="override back.out_dir")
    ap.add_argument("--seed", type=int, default=None, help="override back.seed")
    ap.add_argument("--preview", action="store_true", help="also save box overlays")
    ap.add_argument("--save-blank", action="store_true", help="also save each erased card")
    ap.add_argument("--sample-start", type=int, default=0,
                    help="first sample index (files are <card>_s<index>); use 5 to extend an earlier 5-per-card run")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing out_dir: keep its manifest/skipped rows, refuse to overwrite samples")
    ap.add_argument("--max-new", type=int, default=None,
                    help="stop after exactly N new samples; cards are visited in a seeded random order")
    args = ap.parse_args(argv)
    args.budget = args.max_new

    try:
        check_raqm()
        cfg = load_config(args.config)
        if "back" not in cfg:
            raise SynthError("config.yaml has no 'back:' section")
        bc = dict(cfg["back"])
        if args.seed is not None:
            bc["seed"] = args.seed
        args.samples = args.samples if args.samples is not None else int(bc["samples_per_card"])
        for f in bc["fonts"]:
            if not Path(f["font"]).is_file():
                raise SynthError(f"back font not found: {f['font']}")

        cards_dir = Path(args.cards_dir or bc["cards_dir"])
        coords_dir = Path(args.coords_dir or bc["coords_dir"])
        gt_dir = Path(bc["gt_dir"]) if bc.get("gt_dir") and Path(bc["gt_dir"]).is_dir() else None
        out_dir = cfg_path(cfg, args.out or bc["out_dir"])
        for p, what in ((cards_dir, "cards_dir"), (coords_dir, "coords_dir")):
            if not p.is_dir():
                raise SynthError(f"{what} not found: {p}")
        data = yaml.safe_load(cfg_path(cfg, bc["addresses_file"]).read_text(encoding="utf-8"))
        bc["_label"] = data["label"]

        cards, skipped = discover(cards_dir, coords_dir)
        if args.limit is not None:
            cards = cards[:args.limit]
        if not cards:
            raise SynthError(f"no card images with coords in {cards_dir}")
        cap = f", at most {args.max_new} new" if args.max_new is not None else ""
        print(f"{len(cards)} card(s), {args.samples} sample(s) each from s{args.sample_start:02d}{cap} -> {out_dir}")

        if args.max_new is not None:   # unbiased pick when the cap cuts the run short
            order = np.random.default_rng(card_seed(bc["seed"], f"order#{args.sample_start}")).permutation(len(cards))
            cards = [cards[i] for i in order]

        rows, total = [], 0
        for img_path, coords_path in cards:
            if args.budget is not None and args.budget <= 0:
                break
            try:
                total += process_card(img_path, coords_path, cfg, bc, data, gt_dir, out_dir, args, rows, skipped)
            except SynthError as e:
                print(f"  {img_path.stem}: SKIPPED ({e})")
                skipped.append({"card": img_path.stem, "sample": "all", "reason": str(e)})

        out_dir.mkdir(parents=True, exist_ok=True)
        if args.append:
            def old_rows(name):
                p = out_dir / name
                if not p.is_file():
                    return []
                with open(p, newline="", encoding="utf-8-sig") as f:
                    return list(csv.DictReader(f))
            rows = old_rows("manifest.csv") + rows
            seen, merged = set(), []
            for k in old_rows("skipped.csv") + skipped:
                if "sample" not in k:     # pre-fix skipped.csv had (card, reason) only
                    continue
                key = (k["card"], str(k["sample"]), k["reason"])
                if key not in seen:
                    seen.add(key)
                    merged.append(k)
            skipped = merged
        with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(rows)
        with open(out_dir / "skipped.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["card", "sample", "reason"])
            w.writeheader()
            w.writerows(skipped)
        produced = {r["source_card"] for r in rows}
        no_output = sorted({k["card"] for k in skipped} - produced)
        print(f"done: {total} sample(s), {len(skipped)} skipped sample(s) -> {out_dir}")
        if no_output:
            print(f"cards with no output ({len(no_output)}): {', '.join(no_output)}  (see skipped.csv)")
        return 0
    except SynthError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
