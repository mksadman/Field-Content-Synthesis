#!/usr/bin/env python3
"""
synthesize_front.py -- N new FRONT-side NID cards whose text matches the real
cards' print size exactly (built for OCR training).

It reuses the front-side building blocks of synthesize_one.py (blank template,
fake values, rendering, boxes, checks) and changes how sizes are decided:

  Exact size   Every field is drawn at the size printed on its own source card
               (matra-to-baseline height for Bangla, cap/digit height for Latin),
               one size per field: the Bangla name is NOT forced to share the
               father's/mother's size (on real cards it is ~20% larger). The
               font size is found by rendering and measuring, not by arithmetic,
               because hinting makes rendered heights jump in whole pixels.
               There is no width cap: it used to shrink DOB/NID below the height
               measured on the card.
  Bias         The height measured on real Bangla text (vowel signs, descenders)
               is ~12% larger than the same font's height on a plain reference
               string, so matching the two draws Bangla 12% too big. The size of
               every field is therefore calibrated against typical fake text of
               that field (`content_stats`), not against the reference string.
  Noise        One Bangla measurement is only good to ~7% (it depends on which
               letters the name has). Real cards keep fixed proportions between
               their six fields, so the card's scale is estimated from all six
               (weighted by their noise) and `model_weight` of each field's size
               comes from that model. A field that is clearly off (a merged
               label, a ruined ink mask) gets the modelled size; a card whose
               measurements are too inconsistent is skipped and logged.
  Long values  A value wider than the free space on its line is drawn with a
               smaller font (baseline and start fixed, smallest `min_font_scale`)
               instead of being cut short. `long_name_fraction` of the samples
               make one name long on purpose, so the model sees such cards.
  NID digits   Old cards print 10, 13 or 17 digits. The count is inferred from
               the width of each source NID next to its (exact) height, so the
               fake number has the same length. Real digits are never read.
  Typeface     Each source card keeps its own style (front.styles_file, one row
               per card, labelled by eye): Bangla name / parents' names bold or
               regular, English name CAPS or Title Case and bold or regular,
               DOB and NID bold or regular, Latin text sans or serif. Latin bold
               uses the family's bold face; Kalpurush has none, so its outline
               is thickened (front.faux_bold). Sizes are matched with the style
               applied, so a bold value is as tall as the real one. Unlabelled
               cards draw a style (seeded) from the labelled cards of their type.

The real values on the cards are never read, copied or printed.

Outputs (under front.out_dir, default synthetic/front)
  images/<id>.png                full card
  labels/<id>.txt                YOLO boxes, class ids as in config class_names
  strips/<id>_<field>.png        value crop (box + strip_pad_px), one per field
  labels/<id>_<field>.gt.txt     value text, same header format as the back side
  preview/<id>.png               box overlay (only with --preview)
  templates/                     cached erased cards + measurements (reused)
  manifest.csv                   one row per card: values, font size used, size
                                 asked for, size measured back on the output
  skipped.csv, size_model.json   dropped cards/samples and why; fitted proportions
`split` in manifest.csv is left empty (split by source card afterwards).

Usage
  python "image synthesis/synthesize_front.py" --config config.yaml
  python "image synthesis/synthesize_front.py" --config config.yaml --limit 6 --total 12 --preview
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import shutil
import sys
import time
from collections import Counter
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthesize_one as s1  # noqa: E402
from synthesize_one import CLASS_NAMES, SynthError  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
CACHE_VERSION = 1
SIZE_GRID = 0.1            # font-size step of the exact-size search (px)
SHRINK_GRID = 0.05         # font-size step of shrunk values (px)
NID_VERSION = {"old": "regular", "smart": "smart"}   # names used by the label files
GROUPS = (("name_bn", "name_en"), ("guardian",), ("mother",))   # fields that share one person
STYLE_VALUES = {"name_bn": ("regular", "bold"), "parents": ("regular", "bold"),
                "name_en_case": ("caps", "title"), "name_en": ("regular", "bold"),
                "dob": ("regular", "bold"), "nid": ("regular", "bold"), "latin": ("sans", "serif")}
WEIGHT_OF = {"name_bn": "name_bn", "guardian": "parents", "mother": "parents",
             "name_en": "name_en", "dob": "dob", "nid": "nid"}   # field -> its style column


def manifest_fields() -> list[str]:
    per_field = ("value", "font", "weight", "size", "scale", "src_h", "target_h", "size_src", "h_out",
                 "dx", "dy", "darkness", "blur")
    return (["id", "split", "source_card", "layout", "seed", "sample_seed", "slot", "retries", "shrunk",
             "nid_digits", "name_en_case", "latin", "style_src", "image_w", "image_h"]
            + [f"{n}_{k}" for n in CLASS_NAMES for k in per_field])


# =============================================================================
# Small helpers
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


def derive_seed(*parts) -> int:
    """Stable 63-bit seed from the global seed plus ids (same on every run)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & (2 ** 63 - 1)


def render_mask(lines: list[str], font, lang: str, line_advance: int = 0, stroke: float = 0.0):
    """s1.render_mask with an optional outline stroke (px, may be fractional) for faux bold."""
    if stroke <= 0:
        return s1.render_mask(lines, font, lang, line_advance)
    boxes = [font.getbbox(t, anchor="ls", language=lang, stroke_width=stroke) for t in lines]
    left = int(np.floor(min(b[0] for b in boxes)))
    top = int(np.floor(min(b[1] + i * line_advance for i, b in enumerate(boxes))))
    right = int(np.ceil(max(b[2] for b in boxes)))
    bottom = int(np.ceil(max(b[3] + i * line_advance for i, b in enumerate(boxes))))
    pad = 4
    canvas = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 0)
    draw = ImageDraw.Draw(canvas)
    ax, ay = pad - left, pad - top
    for i, t in enumerate(lines):
        draw.text((ax, ay + i * line_advance), t, font=font, fill=255, anchor="ls", language=lang,
                  stroke_width=stroke, stroke_fill=255)
    return np.array(canvas), (ax, ay)


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:   # BOM: Excel shows Bangla correctly
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# Pass 0: find the source cards
# =============================================================================
def discover(fc: dict, cfg: dict, images_dir: Path, coords_dir: Path) -> tuple[list[dict], list[dict]]:
    """-> (cards, skipped). Cards alternate between card types so --limit gives a mix."""
    per_layout, skipped = {}, []
    for folder, layout in fc["layout_folders"].items():
        if layout not in cfg["layouts"]:
            raise SynthError(f"front.layout_folders: card type '{layout}' has no block under layouts:")
        d = images_dir / folder
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            c = coords_dir / f"{p.stem}.txt"
            if c.is_file():
                per_layout.setdefault(layout, []).append({"path": p, "coords": c, "id": p.stem, "layout": layout})
            else:
                skipped.append({"card": p.stem, "sample": "all", "reason": "no coordinates file"})
    cards = [c for group in zip_longest(*per_layout.values()) for c in group if c is not None]
    dupes = {i for i, n in Counter(c["id"] for c in cards).items() if n > 1}
    for c in [c for c in cards if c["id"] in dupes]:
        skipped.append({"card": c["id"], "sample": "all", "reason": "image name used more than once"})
    return [c for c in cards if c["id"] not in dupes], skipped


# =============================================================================
# Pass 1: blank template + measurements (cached on disk)
# =============================================================================
def cache_key(card: dict, card_cfg: dict, layout_cfg: dict, seed: int) -> str:
    def st(p: Path):
        s = p.stat()
        return [s.st_size, s.st_mtime_ns]
    blob = json.dumps({"v": CACHE_VERSION, "image": st(card["path"]), "coords": st(card["coords"]),
                       "template": card_cfg["template"], "jitter": card_cfg["augment"]["jitter_px"],
                       "forbidden": layout_cfg.get("forbidden_regions", {}), "seed": seed,
                       "scripts": {n: card_cfg["fields"][n]["script"] for n in CLASS_NAMES}},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def measure_card(card: dict, cfg: dict, fc: dict, tmpl_dir: Path, rebuild: bool) -> dict:
    """Erase the six values and measure them; reuse the cached result if nothing changed.

    Returns {"fields", "W", "H", "from_cache"}. The erased card and the
    forbidden-region mask stay on disk (templates/<id>.png, <id>.forbidden.npz).
    """
    card_cfg, layout_cfg = s1.layout_settings(cfg, card["layout"])
    card_seed = derive_seed(fc["seed"], card["id"])
    key = cache_key(card, card_cfg, layout_cfg, card_seed)
    stem = tmpl_dir / card["id"]
    meta_p = stem.with_suffix(".json")
    if not rebuild and meta_p.is_file() and stem.with_suffix(".png").is_file() \
            and Path(f"{stem}.forbidden.npz").is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            return {"fields": meta["fields"], "W": meta["W"], "H": meta["H"], "from_cache": True}

    img = imread(card["path"])
    H, W = img.shape[:2]
    boxes = s1.read_yolo_labels(str(card["coords"]), W, H)   # raises unless 6 boxes, one per class
    cc = s1.scale_pixel_settings(card_cfg, W)
    cc = {**cc, "generation": {**cc["generation"], "seed": card_seed}}
    with contextlib.redirect_stdout(io.StringIO()):          # its notes are not needed here
        blank, fields, forbidden = s1.make_blank_template(img, boxes, cc, layout_cfg)
    imwrite(stem.with_suffix(".png"), blank)
    np.savez_compressed(f"{stem}.forbidden.npz", bits=np.packbits(forbidden), shape=np.array(forbidden.shape))
    meta_p.write_text(json.dumps({"key": key, "W": W, "H": H, "fields": fields}, ensure_ascii=False),
                      encoding="utf-8")
    return {"fields": fields, "W": W, "H": H, "from_cache": False}


def load_template(tmpl_dir: Path, card_id: str) -> tuple[np.ndarray, np.ndarray]:
    blank = imread((tmpl_dir / card_id).with_suffix(".png"))
    z = np.load(f"{tmpl_dir / card_id}.forbidden.npz")
    shape = tuple(int(v) for v in z["shape"])
    forbidden = np.unpackbits(z["bits"])[:shape[0] * shape[1]].reshape(shape).astype(bool)
    return blank, forbidden


# =============================================================================
# Exact sizes
# =============================================================================
def content_stats(layout: str, cfg: dict, names: dict, n_texts: int = 150,
                  heights=(10, 14, 22, 40)) -> dict[str, tuple[float, float]]:
    """Per field: (factor, sd) of the height measured on typical text relative to
    the plain reference string of the same font and size.

    The size of a font is matched to a measured height through the reference
    string, but the cards were measured on real text: Bangla with vowel signs and
    descenders reads ~12% taller than the reference string, and by a few percent
    more or less depending on the letters (sd). The factor undoes that bias; the
    sd says how noisy one measurement of that field is.
    """
    card_cfg, layout_cfg = s1.layout_settings(cfg, layout)
    gcfg = card_cfg["generation"]
    out = {}
    for name in CLASS_NAMES:
        fcfg = card_cfg["fields"][name]
        script, path, index = fcfg["script"], fcfg["font"], fcfg.get("font_index", 0)
        ref = cfg["fonts"]["reference_text"][script]
        rng = np.random.default_rng(derive_seed("content_stats", layout, name))
        gen = s1.FakeValues(names, layout_cfg, gcfg, rng)
        ratios = []
        for h in heights:
            font = s1.load_font(path, index, h / font_constants(path, index, script, ref))
            mask, _ = s1.render_mask([ref], font, s1.lang_of(script))
            h_ref = s1.text_metrics(mask > 127, script)["ref_height"]
            for _ in range(n_texts):
                if name == "dob":
                    text = gen.dob()
                elif name == "nid":
                    text = gen.nid()
                else:
                    who = "female" if name == "mother" else ("male" if name == "guardian" else
                                                             ("male" if rng.random() < 0.5 else "female"))
                    bn, en = gen.person(who, int(rng.integers(gcfg["name_words"][0], gcfg["name_words"][1] + 1)),
                                        gen.community())
                    text = bn if script == "bn" else en
                mask, _ = s1.render_mask([text], font, s1.lang_of(script))
                ratios.append(s1.text_metrics(mask > 127, script)["ref_height"] / h_ref)
        out[name] = (float(np.mean(ratios)), float(np.std(ratios)))
    s1.load_font.cache_clear()
    return out


class SizeModel:
    """Typical proportions of the six field sizes per card type.

    With z(c, f) = log(measured height of field f on card c / content factor f):
    z ~ scale(c) + ratio(f), fitted by median polish over all source cards, so a
    few bad measurements do not bend it. A card's own scale is the noise-weighted
    mean over the fields that agree with the model.
    """

    def __init__(self, fc: dict, stats: dict[str, dict]):
        self.fc, self.stats = fc, stats          # stats[layout][field] = (factor, sd)
        self.rel, self.n = {}, {}

    def _z(self, layout: str, heights) -> np.ndarray:
        return np.log(np.asarray(heights, float)) - np.log([self.stats[layout][n][0] for n in CLASS_NAMES])

    def fit(self, heights: dict[str, list[list[float]]]):
        for layout, rows in heights.items():
            self.n[layout] = len(rows)
            if len(rows) < self.fc["model_min_cards"]:
                continue
            L = np.array([self._z(layout, row) for row in rows])
            s, r = np.zeros(len(L)), np.zeros(L.shape[1])
            for _ in range(10):
                r = np.median(L - s[:, None], axis=0)
                s = np.median(L - r[None, :], axis=1)
            self.rel[layout] = r - r.mean()

    def resolve(self, layout: str, heights: list[float]) -> dict:
        """Heights measured on one card -> the sizes to draw.

        Returns targets (reference-string height to match), expected (the height the
        estimator should read on the output: target x factor), sources, and the worst
        field's residual. A field within `size_tolerance` of the model blends its own
        size with the modelled one (`model_weight`); one further off is modelled only.
        """
        fc = self.fc
        z = self._z(layout, heights)
        factor = np.array([self.stats[layout][n][0] for n in CLASS_NAMES])
        if layout not in self.rel:
            t = np.exp(z)
            return {"targets": [float(v) for v in t], "expected": [float(v) for v in t * factor],
                    "sources": ["measured"] * len(z), "worst": 0.0, "worst_field": ""}
        r = self.rel[layout]
        s = float(np.median(z - r))
        inl = np.abs(z - s - r) <= fc["size_tolerance"]
        if inl.sum() >= 2:   # noise-weighted: Latin fields count more than Bangla ones
            sd = np.array([self.stats[layout][n][1] for n in CLASS_NAMES])
            w = 1.0 / (sd ** 2 + fc["measure_noise"] ** 2)
            s = float(np.sum((w * (z - r))[inl]) / np.sum(w[inl]))
        res = z - s - r
        lam, targets, srcs = fc["model_weight"], [], []
        for i in range(len(z)):
            ok = abs(res[i]) <= fc["size_tolerance"]
            targets.append(float(np.exp((1 - lam) * z[i] + lam * (s + r[i]) if ok else s + r[i])))
            srcs.append("blend" if ok else "model")
        worst = int(np.argmax(np.abs(res)))
        return {"targets": targets, "expected": [float(t * f) for t, f in zip(targets, factor)],
                "sources": srcs, "worst": float(abs(res[worst])), "worst_field": CLASS_NAMES[worst]}

    def as_json(self) -> dict:
        g = CLASS_NAMES.index("guardian")
        return {lay: {"cards": self.n[lay],
                      "relative_size_vs_guardian": {n: round(float(np.exp(r[i] - r[g])), 3)
                                                    for i, n in enumerate(CLASS_NAMES)},
                      "measured_vs_reference_height": {n: {"factor": round(self.stats[lay][n][0], 3),
                                                           "sd": round(self.stats[lay][n][1], 3)}
                                                       for n in CLASS_NAMES}}
                for lay, r in self.rel.items()}


@lru_cache(maxsize=None)
def font_constants(path: str, index: int, script: str, ref_text: str, stroke: float = 0.0,
                   big: int = 400) -> float:
    """Reference height per px of font size, from a large render (0.25% resolution).
    `stroke` is the faux-bold outline as a share of the font size."""
    mask, _ = render_mask([ref_text], s1.load_font(path, index, big), s1.lang_of(script), 0, stroke * big)
    return s1.text_metrics(mask > 127, script)["ref_height"] / big


def pick_size(fcfg: dict, target_h: float, cfg: dict, search_px: float):
    """Font size whose rendered reference height equals `target_h`.

    Hinting makes rendered heights jump in whole pixels, so sizes around the
    exact one are rendered and measured; of those with the smallest height
    error the one nearest the exact size wins, which keeps the width exact.
    Returns (size, anchor_offset, rendered_height); anchor_offset is the
    distance from the font baseline to the measured anchor row (matra/baseline).
    """
    script, path, index = fcfg["script"], fcfg["font"], fcfg.get("font_index", 0)
    stroke = fcfg.get("stroke", 0.0)
    ref = cfg["fonts"]["reference_text"][script]
    lo, hi = cfg["fonts"]["size_range"]
    s0 = target_h / font_constants(path, index, script, ref, stroke)
    best = None
    for i in range(-int(round(search_px / SIZE_GRID)), int(round(search_px / SIZE_GRID)) + 1):
        size = round(s0 + i * SIZE_GRID, 2)
        if not lo <= size <= hi:
            continue
        mask, (_, ay) = render_mask([ref], s1.load_font(path, index, size), s1.lang_of(script), 0, stroke * size)
        m = s1.text_metrics(mask > 127, script)
        key = (abs(m["ref_height"] - target_h), abs(size - s0))
        if best is None or key < best[0]:
            best = (key, size, m["anchor_y"] - ay, m["ref_height"])
    if best is None:
        raise SynthError(f"no font size in {lo}-{hi} px reaches a height of {target_h:.1f} px")
    return best[1], best[2], best[3]


# =============================================================================
# Rendering with a size per field and sample
# =============================================================================
class SizedRenderer(s1.FieldRenderer):
    """FieldRenderer whose font size can differ per sample (long values shrink).

    `size[name]` is the size used by mask(); the baseline and start stay those
    of the exact size, so a shrunk value sits on the same line as the label.
    """

    def __init__(self, fields: dict, cfg: dict, layout_cfg: dict):
        super().__init__(fields, cfg, layout_cfg)
        self.base = {n: f["font"]["size"] for n, f in fields.items()}
        self.size = dict(self.base)

    def font(self, name):
        fnt = self.f[name]["font"]
        return s1.load_font(fnt["path"], fnt["index"], self.size[name])

    def mask(self, name, lines):
        adv = round(self.lay["wrap_line_spacing"] * self.size[name])
        stroke = self.f[name]["font"].get("stroke", 0.0) * self.size[name]   # faux bold scales with the size
        return render_mask(lines, self.font(name), s1.lang_of(self.f[name]["script"]), adv, stroke)

    def width(self, name: str, text: str, size: float) -> int:
        key = (name, text, size)
        if key not in self._width_cache:
            old, self.size[name] = self.size[name], size
            try:
                mask, _ = self.mask(name, [text])
            finally:
                self.size[name] = old
            cols = np.nonzero((mask > self.alpha_thr).any(axis=0))[0]
            self._width_cache[key] = int(cols[-1] - cols[0] + 1) if cols.size else 0
        return self._width_cache[key]

    def fit(self, name: str, text: str, allow_shrink: bool, min_scale: float):
        """(size, scale) that puts `text` inside the field's free width, or None.
        scale is 1.0 when the exact size fits; otherwise the font is shrunk."""
        base, maxw = self.base[name], self.f[name]["max_ink_width"]
        w = self.width(name, text, base)
        if w <= maxw:
            return base, 1.0
        if not allow_shrink:
            return None
        scale = maxw / w
        while scale >= min_scale:
            size = round(round(base * scale / SHRINK_GRID) * SHRINK_GRID, 2)
            if self.width(name, text, size) <= maxw:
                return size, size / base
            scale -= 0.01   # widths do not shrink perfectly linearly (hinting)
        return None


# =============================================================================
# Typeface per source card
# =============================================================================
def load_styles(path: Path) -> dict[str, dict[str, str]]:
    """front_styles.csv -> {card id: {column: value}}; blank cells are left out."""
    if not path.is_file():
        raise SynthError(f"front.styles_file not found: {path}")
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for line, row in enumerate(csv.DictReader(fh), 2):
            cid = (row.get("card") or "").strip()
            if not cid:
                continue
            own = {}
            for key, allowed in STYLE_VALUES.items():
                v = (row.get(key) or "").strip().lower()
                if v and v not in allowed:
                    raise SynthError(f"{path.name} line {line}: {key} is '{v}', expected one of {'/'.join(allowed)}")
                if v:
                    own[key] = v
            out[cid] = own
    return out


def style_shares(labels: dict, layout_of: dict[str, str]) -> dict[str, dict[str, Counter]]:
    """Per card type and style column: how often each value occurs among the labelled cards."""
    shares: dict[str, dict[str, Counter]] = {}
    for cid, own in labels.items():
        if cid in layout_of:
            for key, v in own.items():
                shares.setdefault(layout_of[cid], {}).setdefault(key, Counter())[v] += 1
    return shares


def card_style(card_id: str, layout: str, labels: dict, shares: dict, fc: dict) -> tuple[dict, str]:
    """-> (style, source). Labelled cells are used as they are; the others are drawn
    once per card (seeded) from the labelled cards of the same type, or taken from
    front.style_defaults when that type has too few labels."""
    own = labels.get(card_id, {})
    rng = np.random.default_rng(derive_seed(fc["seed"], "style", card_id))
    style = {}
    for key in STYLE_VALUES:
        cnt = shares.get(layout, {}).get(key, Counter())
        if key in own:
            style[key] = own[key]
        elif sum(cnt.values()) >= fc["style_min_labels"]:
            vals = sorted(cnt)
            p = np.array([cnt[v] for v in vals], float)
            style[key] = vals[int(rng.choice(len(vals), p=p / p.sum()))]
        else:
            style[key] = fc["style_defaults"][layout][key]
    src = "labelled" if len(own) == len(STYLE_VALUES) else ("partly labelled" if own else "unlabelled")
    return style, src


def same_file(a: str, b: str) -> bool:
    return str(Path(a)).lower() == str(Path(b)).lower()


def field_face(name: str, fcfg: dict, style: dict, fc: dict) -> dict:
    """Font file + faux-bold stroke (share of the size) of one field in a card's style.

    A Latin font that belongs to a family of front.latin_faces is swapped for that
    family's face of the wanted weight (serif cards: the serif family). Any other
    font is kept and made bold by thickening its outline.
    """
    weight = style[WEIGHT_OF[name]]
    path, index = fcfg["font"], fcfg.get("font_index", 0)
    if fcfg["script"] == "en":
        faces = fc["latin_faces"]
        fam = next((k for k, f in faces.items() if same_file(path, f["regular"]) or same_file(path, f["bold"])), None)
        if fam is not None:
            if style["latin"] == "serif":
                fam = "serif"
            return {"path": faces[fam][weight], "index": 0, "stroke": 0.0}
    return {"path": path, "index": index, "stroke": fc["faux_bold"] if weight == "bold" else 0.0}


def prepare_card(rec: dict, cfg: dict, fc: dict, tmpl_dir: Path, style: dict) -> dict:
    """Pick the exact font sizes of one source card (in its own typeface) and set up its renderer."""
    card_cfg, layout_cfg = s1.layout_settings(cfg, rec["layout"])
    card_cfg = s1.scale_pixel_settings(card_cfg, rec["W"])
    fields = json.loads(json.dumps(rec["fields"]))   # private copy
    jitter_max = max(card_cfg["augment"]["jitter_px"])
    for name in ("dob", "nid"):
        # Fixed-format values must never be drawn smaller than the real one: the
        # real value's own width (a little more, for another month's name) is
        # allowed even where it eats into the safety gap, up to the real obstacle.
        f = fields[name]
        hard = f["obstacle_x"] - f["start_x"] - jitter_max - 1
        f["max_ink_width"] = max(f["max_ink_width"], min(int(1.08 * f["orig_ink_width"]), hard))
    for i, name in enumerate(CLASS_NAMES):
        face = field_face(name, card_cfg["fields"][name], style, fc)
        fcfg = {**card_cfg["fields"][name], "font": face["path"], "font_index": face["index"], "stroke": face["stroke"]}
        size, offset, got = pick_size(fcfg, rec["targets"][i], card_cfg, fc["fit_search_px"])
        f = fields[name]
        f["src_h"], f["target_h"], f["size_src"] = f["ref_height"], rec["expected"][i], rec["sources"][i]
        f["weight"] = style[WEIGHT_OF[name]]
        f["font"] = {"path": face["path"], "index": face["index"], "stroke": face["stroke"], "size": size,
                     "draw_baseline_y": int(f["anchor_y"] - offset), "rendered_ref_height": got}
    layout_cfg = {**layout_cfg, "english_name_all_caps": style["name_en_case"] == "caps"}
    rend = SizedRenderer(fields, card_cfg, layout_cfg)

    # NID digit count: the format whose width at the exact size is closest to the original's.
    formats = fc["nid_formats"][rec["layout"]]
    sep = layout_cfg["nid_separator"]
    errs = []
    for fmt in formats:
        digits, parts, i = "8" * fmt["digits"], [], 0
        for g in fmt["groups"]:
            parts.append(digits[i:i + g])
            i += g
        w = rend.width("nid", sep.join(parts), rend.base["nid"])
        errs.append(abs(np.log(max(w, 1) / fields["nid"]["orig_ink_width"])))
    fmt = formats[int(np.argmin(errs))]
    layout_card = {**layout_cfg, "nid_digits": fmt["digits"], "nid_groups": fmt["groups"]}

    blank, forbidden = load_template(tmpl_dir, rec["id"])
    return {"fields": fields, "rend": rend, "cfg": card_cfg, "layout_cfg": layout_card,
            "blank": blank, "forbidden": forbidden, "nid_digits": fmt["digits"], "style": style}


# =============================================================================
# Fake values (names that fit, or long names that need a smaller font)
# =============================================================================
def draw_group(gen: s1.FakeValues, group: tuple, n: int, holder: str, comm: str) -> tuple:
    if group[0] == "name_bn":
        return gen.person(holder, n, comm)                   # (Bangla, English)
    return (gen.person("male" if group[0] == "guardian" else "female", n, comm)[0],)


def fit_group(rend: SizedRenderer, group: tuple, texts: tuple, allow_shrink: bool, min_scale: float):
    fit = {}
    for name, text in zip(group, texts):
        r = rend.fit(name, text, allow_shrink, min_scale)
        if r is None:
            return None
        fit[name] = r
    return fit


def choose_values(gen, rend: SizedRenderer, long_sample: bool, gcfg: dict, fc: dict, rng):
    """All six fake values -> ({field: text}, {field: (size, scale)}).

    Normally every name must fit at the exact size (a shorter name is drawn if
    not). For a long sample one random person gets a name that does not fit at
    the exact size but does at a smaller one (>= front.min_font_scale).
    """
    lo, hi = gcfg["name_words"]
    holder = "male" if rng.random() < 0.5 else "female"
    comm = gen.community()
    long_group = int(rng.integers(len(GROUPS))) if long_sample else -1
    values, sizes = {}, {}
    for gi, group in enumerate(GROUPS):
        chosen = None
        if gi == long_group:
            for _ in range(gcfg["hard_candidates"]):
                texts = draw_group(gen, group, int(rng.integers(max(lo, 3), hi + 1)), holder, comm)
                fit = fit_group(rend, group, texts, True, fc["min_font_scale"])
                if fit and any(scale < 1.0 for _, scale in fit.values()):
                    chosen = (texts, fit)
                    break
        if chosen is None:
            n = int(rng.integers(lo, hi + 1))
            for _ in range(gcfg["max_value_tries"]):
                texts = draw_group(gen, group, n, holder, comm)
                fit = fit_group(rend, group, texts, False, 1.0)
                if fit:
                    chosen = (texts, fit)
                    break
                n = max(lo, n - 1)          # too wide at the exact size: a shorter name, as in synthesize_one
        if chosen is None:
            raise SynthError(f"no name for {group} fits the free width")
        for name, text in zip(group, chosen[0]):
            values[name], sizes[name] = text, chosen[1][name]
    for name, make in (("dob", gen.dob), ("nid", gen.nid)):
        text = make()
        r = rend.fit(name, text, True, fc["min_font_scale"])
        if r is None:
            raise SynthError(f"'{name}' value does not fit the free width")
        values[name], sizes[name] = text, r
    return values, sizes


# =============================================================================
# One sample
# =============================================================================
def measure_output(img, blank, info, fields) -> dict:
    """Reference height of every drawn value, measured back on the output with
    the same code that measured the real card (half-maximum ink threshold)."""
    diff = np.abs(img.astype(np.int16) - blank.astype(np.int16)).max(axis=2)
    out = {}
    for name in CLASS_NAMES:
        area = info[name]["area"]
        if not area.any():
            out[name] = None
            continue
        thr = max(30.0, 0.5 * float(diff[area].max()))
        ink = (diff > thr) & area
        try:
            out[name] = s1.text_metrics(ink, fields[name]["script"])["ref_height"] if ink.any() else None
        except SynthError:
            out[name] = None
    return out


def make_sample(prep: dict, rec: dict, slot: int, cfg: dict, fc: dict, names: dict, out_dir: Path, args, log: list):
    """One new card (with retries). Returns a manifest row, or None if every try failed."""
    fields, rend, card_cfg = prep["fields"], prep["rend"], prep["cfg"]
    sid = f"{rec['id']}_s{slot:02d}"
    reasons = []
    for retry in range(fc["max_retries"] + 1):
        sample_seed = derive_seed(fc["seed"], rec["id"], slot, retry)
        rng = np.random.default_rng(sample_seed)
        gen = s1.FakeValues(names, prep["layout_cfg"], card_cfg["generation"], rng)
        try:
            values, sizes = choose_values(gen, rend, bool(rng.random() < fc["long_name_fraction"]),
                                          card_cfg["generation"], fc, rng)
        except SynthError as e:
            reasons = [str(e)]
            continue
        rend.size = {n: sizes[n][0] for n in CLASS_NAMES}
        img, info = s1.render_card(prep["blank"], rend, {n: [values[n]] for n in CLASS_NAMES}, card_cfg, rng)
        boxes, visible = s1.boxes_from_changes(img, prep["blank"], info, fields, prep["forbidden"], card_cfg)
        reasons = s1.check_sample(boxes, visible, info, prep["forbidden"], card_cfg)
        if reasons:
            continue

        H, W = img.shape[:2]
        imwrite(out_dir / "images" / f"{sid}.png", img)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / f"{sid}.txt").write_text(s1.yolo_lines(boxes, W, H), encoding="utf-8")
        if args.preview:
            imwrite(out_dir / "preview" / f"{sid}.png", s1.draw_boxes(img, boxes))
        if not args.no_strips:
            pad = fc["strip_pad_px"]
            for name in CLASS_NAMES:
                x1, y1, x2, y2 = boxes[name]
                imwrite(out_dir / "strips" / f"{sid}_{name}.png",
                        img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)])
                (out_dir / "labels" / f"{sid}_{name}.gt.txt").write_text(
                    f"# card_id: {sid}_{name}\n# side: front\n# field: {name}\n"
                    f"# nid_version: {NID_VERSION.get(rec['layout'], 'unknown')}\n"
                    f"# label_status: synthetic\n# source_card: {rec['id']}\n\n{values[name]}\n", encoding="utf-8")

        h_out = measure_output(img, prep["blank"], info, fields)
        row = {"id": sid, "split": "", "source_card": rec["id"], "layout": rec["layout"], "seed": fc["seed"],
               "sample_seed": sample_seed, "slot": slot, "retries": retry,
               "shrunk": ";".join(n for n in CLASS_NAMES if sizes[n][1] < 1.0),
               "nid_digits": prep["nid_digits"], "name_en_case": prep["style"]["name_en_case"],
               "latin": prep["style"]["latin"], "style_src": prep["style_src"], "image_w": W, "image_h": H}
        for n in CLASS_NAMES:
            f = fields[n]
            row.update({f"{n}_value": values[n], f"{n}_font": Path(f["font"]["path"]).name,
                        f"{n}_weight": f["weight"] + (" (faux)" if f["font"]["stroke"] > 0 else ""),
                        f"{n}_size": sizes[n][0], f"{n}_scale": round(sizes[n][1], 3),
                        f"{n}_src_h": f["src_h"], f"{n}_target_h": round(f["target_h"], 2),
                        f"{n}_size_src": f["size_src"],
                        f"{n}_h_out": h_out[n] if h_out[n] is not None else "",
                        f"{n}_dx": info[n]["dx"], f"{n}_dy": info[n]["dy"],
                        f"{n}_darkness": info[n]["darkness"], f"{n}_blur": info[n]["blur"]})
        return row
    log.append({"card": rec["id"], "sample": slot, "reason": "; ".join(dict.fromkeys(reasons))})
    return None


def allocate(ids: list[str], total: int, seed: int, tag: str) -> dict[str, int]:
    """`total` samples spread evenly over the cards; the remainder goes to random cards."""
    base, extra = divmod(total, len(ids))
    quota = {i: base for i in ids}
    for k in np.random.default_rng(derive_seed(seed, "allocation", tag)).permutation(len(ids))[:extra]:
        quota[ids[int(k)]] += 1
    return quota


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic front-side NID cards with exact text sizes.")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    ap.add_argument("--total", type=int, default=None, help="images to make (default: front.total)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N source cards (alternating card types)")
    ap.add_argument("--out", default=None, help="override front.out_dir")
    ap.add_argument("--seed", type=int, default=None, help="override front.seed")
    ap.add_argument("--preview", action="store_true", help="also save box overlays")
    ap.add_argument("--no-strips", action="store_true", help="skip the per-field strips and their .gt.txt files")
    ap.add_argument("--rebuild-templates", action="store_true", help="ignore the cached erased cards")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an earlier run in the output folder (deletes its images/strips/labels/preview)")
    args = ap.parse_args(argv)

    s1.check_raqm()
    cfg = s1.load_config(args.config)
    if "front" not in cfg:
        raise SynthError("config.yaml has no 'front:' section")
    fc = dict(cfg["front"])
    if args.seed is not None:
        fc["seed"] = args.seed
    total = args.total if args.total is not None else int(fc["total"])
    for lay in set(fc["layout_folders"].values()):
        s1.layout_settings(cfg, lay)            # fail fast on a bad card-type block or a missing font
        if lay not in fc["nid_formats"]:
            raise SynthError(f"front.nid_formats has no entry for card type '{lay}'")
        for key, allowed in STYLE_VALUES.items():
            if fc.get("style_defaults", {}).get(lay, {}).get(key) not in allowed:
                raise SynthError(f"front.style_defaults.{lay}.{key} must be one of {'/'.join(allowed)}")
    if "serif" not in fc["latin_faces"]:
        raise SynthError("front.latin_faces needs a 'serif' family (used by cards labelled latin: serif)")
    for fam, faces in fc["latin_faces"].items():
        for w in ("regular", "bold"):
            if not Path(faces.get(w, "")).is_file():
                raise SynthError(f"front.latin_faces.{fam}.{w}: font file not found: {faces.get(w)}")

    out_dir = s1.cfg_path(cfg, args.out or fc["out_dir"])
    tmpl_dir = out_dir / "templates"
    # A run that died half-way leaves images but may have no manifest, so look at the folders too.
    earlier = (out_dir / "manifest.csv").is_file() or any(
        (out_dir / sub).is_dir() and any((out_dir / sub).iterdir()) for sub in ("images", "strips", "labels"))
    if earlier:
        if not args.overwrite:
            raise SynthError(f"{out_dir} already holds a run; use --overwrite to replace it or --out for another folder")
        for sub in ("images", "strips", "labels", "preview"):
            shutil.rmtree(out_dir / sub, ignore_errors=True)
    tmpl_dir.mkdir(parents=True, exist_ok=True)
    # Full-size PNGs add up (~1.1 MB per image with its strips, ~0.8 MB more per preview).
    need_mb = total * (1.1 + (0.8 if args.preview else 0.0))
    free_mb = shutil.disk_usage(out_dir).free / 2 ** 20
    if free_mb < need_mb:
        raise SynthError(f"only {free_mb:.0f} MB free on the disk of {out_dir}; about {need_mb:.0f} MB are needed "
                         f"for {total} image(s). Free some space, lower --total, or use --out on another drive")

    names = yaml.safe_load(s1.cfg_path(cfg, cfg["paths"]["names_file"]).read_text(encoding="utf-8"))
    cards, skipped = discover(fc, cfg, s1.cfg_path(cfg, fc["images_dir"]), s1.cfg_path(cfg, fc["coords_dir"]))
    if args.limit is not None:
        cards = cards[:max(0, args.limit)]
    if not cards:
        raise SynthError("no source cards with a coordinates file found")
    labels = load_styles(s1.cfg_path(cfg, fc["styles_file"])) if fc.get("styles_file") else {}
    shares = style_shares(labels, {c["id"]: c["layout"] for c in cards})
    print(f"{len(cards)} source card(s) -> {total} new image(s) in {out_dir}")
    print(f"  typeface labels: {sum(c['id'] in labels for c in cards)} card(s) in "
          f"{fc.get('styles_file') or '(no styles_file)'}; the others draw theirs from their card type")
    t0 = time.time()

    # ---- Pass 1: erase + measure every card --------------------------------------
    recs = []
    for k, card in enumerate(cards, 1):
        try:
            m = measure_card(card, cfg, fc, tmpl_dir, args.rebuild_templates)
        except SynthError as e:
            skipped.append({"card": card["id"], "sample": "all", "reason": str(e).splitlines()[0]})
            continue
        except Exception as e:  # one broken card must not stop the batch
            skipped.append({"card": card["id"], "sample": "all", "reason": f"error: {type(e).__name__}: {e}"})
            continue
        recs.append({**card, **m})
        if not m["from_cache"]:
            print(f"  measured [{k}/{len(cards)}] {card['id']} ({card['layout']})")

    # ---- Size model: undo the measuring bias, smooth the noise, catch bad cards --
    layouts = sorted({r["layout"] for r in recs})
    stats = {lay: content_stats(lay, cfg, names) for lay in layouts}
    for lay in layouts:
        bn = stats[lay]["guardian"]
        print(f"  '{lay}': Bangla text measures x{bn[0]:.2f} (sd {bn[1]:.0%}) the reference height, "
              f"Latin x{np.mean([stats[lay][n][0] for n in ('name_en', 'dob', 'nid')]):.2f}")
    model = SizeModel(fc, stats)
    model.fit({lay: [[r["fields"][n]["ref_height"] for n in CLASS_NAMES] for r in recs if r["layout"] == lay]
               for lay in layouts})
    for lay, n in model.n.items():
        if lay not in model.rel:
            print(f"  note: only {n} '{lay}' card(s) (< model_min_cards); sizes are the bias-corrected measurements")
    usable = []
    for r in recs:
        res = model.resolve(r["layout"], [r["fields"][n]["ref_height"] for n in CLASS_NAMES])
        r.update(targets=res["targets"], expected=res["expected"], sources=res["sources"])
        if res["worst"] > fc["skip_tolerance"]:
            skipped.append({"card": r["id"], "sample": "all",
                            "reason": f"measurements inconsistent: '{res['worst_field']}' height is "
                                      f"{np.exp(res['worst']) - 1:.0%}+ off the typical proportions"})
            continue
        usable.append(r)
    if not usable:
        raise SynthError("no source card has usable measurements (see skipped.csv)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "size_model.json").write_text(json.dumps(model.as_json(), indent=2), encoding="utf-8")
    n_model = sum(s == "model" for r in usable for s in r["sources"])
    print(f"  {len(usable)} usable card(s), {len(recs) - len(usable)} dropped as inconsistent, "
          f"{n_model} field size(s) replaced by the model ({time.time() - t0:.0f} s)")

    # ---- Pass 2: make the samples (a round more if some fail) --------------------
    by_id = {r["id"]: r for r in usable}
    next_slot = {i: 0 for i in by_id}
    rows: list[dict] = []
    try:
        for rnd in range(4):
            want = total - len(rows)
            if want <= 0 or not by_id:
                break
            quota = allocate(list(by_id), want, fc["seed"], str(rnd))
            for rec_id, count in quota.items():
                if not count or rec_id not in by_id:
                    continue
                rec = by_id[rec_id]
                try:
                    style, style_src = card_style(rec_id, rec["layout"], labels, shares, fc)
                    prep = prepare_card(rec, cfg, fc, tmpl_dir, style)
                    prep["style_src"] = style_src
                    made = 0
                    for _ in range(count):
                        slot = next_slot[rec_id]
                        next_slot[rec_id] += 1
                        row = make_sample(prep, rec, slot, cfg, fc, names, out_dir, args, skipped)
                        if row:
                            rows.append(row)
                            made += 1
                except SynthError as e:
                    skipped.append({"card": rec_id, "sample": "all", "reason": str(e).splitlines()[0]})
                    del by_id[rec_id]
                    print(f"  {rec_id}: SKIPPED ({str(e).splitlines()[0]})")
                    continue
                finally:
                    s1.load_font.cache_clear()   # sizes differ per card; do not pile up open fonts
                print(f"  {rec_id} ({rec['layout']}, NID {prep['nid_digits']} digits): {made}/{count} made")
    finally:
        # Written even if the run dies (disk full, Ctrl+C), so finished samples are recorded.
        write_csv(out_dir / "manifest.csv", manifest_fields(), rows)
        write_csv(out_dir / "skipped.csv", ["card", "sample", "reason"], skipped)

    # ---- Summary -----------------------------------------------------------------
    print(f"\nSummary ({time.time() - t0:.0f} s)")
    print(f"  images made   : {len(rows)} of {total}  "
          f"({', '.join(f'{k}: {v}' for k, v in sorted(Counter(r['layout'] for r in rows).items()))})")
    print(f"  source cards  : {len({r['source_card'] for r in rows})} used, "
          f"{len({s['card'] for s in skipped if s['sample'] == 'all'})} dropped (skipped.csv)")
    shrunk = [r for r in rows if r["shrunk"]]
    print(f"  shrunk values : {len(shrunk)} image(s) with a smaller font "
          f"({', '.join(f'{k}: {v}' for k, v in sorted(Counter(n for r in shrunk for n in r['shrunk'].split(';')).items()))})")
    if shrunk:
        sc = [float(r[f"{n}_scale"]) for r in shrunk for n in r["shrunk"].split(";")]
        print(f"                  scale {min(sc):.2f}-{max(sc):.2f}, median {float(np.median(sc)):.2f}")
    print("  size check: height measured back on the output / height expected from the source card\n"
          "              (values drawn at the exact size; mean ~1.00 = no bias, sd = how much it depends on the letters)")
    for n in CLASS_NAMES:
        d = [(float(r[f"{n}_h_out"]), float(r[f"{n}_target_h"])) for r in rows
             if float(r[f"{n}_scale"]) == 1.0 and r[f"{n}_h_out"] != ""]
        if d:
            ratio = np.array([a / b for a, b in d])
            print(f"      {n:9s} mean {ratio.mean():.3f}, sd {ratio.std():.3f}, "
                  f"within 1 px: {np.mean([abs(a - b) <= 1.0 for a, b in d]):.0%}  (n={len(d)})")
    n_src = Counter(r[f"{n}_size_src"] for r in rows for n in CLASS_NAMES)
    print("  field sizes   : " + ", ".join(f"{v} {k}" for k, v in sorted(n_src.items())) +
          "  (blend = own measurement + size model; model = measurement rejected; measured = no model)")
    print("  NID digits    : " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r['nid_digits'] for r in rows).items())))
    for lay in sorted({r["layout"] for r in rows}):
        lr = [r for r in rows if r["layout"] == lay]
        def share(pred):
            return f"{sum(map(pred, lr)) / len(lr):.0%}"
        print(f"  typeface {lay:5s}: bold name_bn {share(lambda r: r['name_bn_weight'].startswith('bold'))}, "
              f"parents {share(lambda r: r['guardian_weight'].startswith('bold'))}, "
              f"name_en {share(lambda r: r['name_en_weight'].startswith('bold'))}, "
              f"dob {share(lambda r: r['dob_weight'].startswith('bold'))}, "
              f"nid {share(lambda r: r['nid_weight'].startswith('bold'))}; "
              f"CAPS {share(lambda r: r['name_en_case'] == 'caps')}, serif {share(lambda r: r['latin'] == 'serif')}; "
              f"styles {', '.join(f'{k} {v}' for k, v in sorted(Counter(r['style_src'] for r in lr).items()))} images")
    if skipped:
        for reason, count in Counter(s["reason"].split(":")[0] for s in skipped).most_common(6):
            print(f"      skipped {count:3d}  {reason}")
    print(f"\n  manifest : {out_dir / 'manifest.csv'}\n  skipped  : {out_dir / 'skipped.csv'}")
    return 0 if len(rows) >= total else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SynthError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
