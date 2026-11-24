"""Angle-and-framing variants: the card photographed crooked, sideways,
upside down, from an angle, too far away / too close, half out of the frame,
or off-center.

Unlike generate_degraded.py (which only nudges the card by a few degrees and
rejects any variant whose boxes touch the image edge), these ops are SUPPOSED
to move the card around a bigger frame and sometimes cut part of it off. So:

  - The card is pasted onto a larger "photo" canvas with a background behind
    it (procedural table/desk textures, or real photos from BACKGROUNDS_DIR
    if that folder exists), then the whole thing is done as ONE homography so
    every op composes cleanly and the labels go through the exact same math
    as the pixels.
  - 90 / 180 degree turns are applied to the finished photo with np.rot90,
    so they are pixel-exact.

Two kinds of label are supported, picked per card:

  - Text ground truth (<card>.gt.txt next to the image, as in the annotated
    dataset's front/ and back/ folders). The text doesn't change with the
    framing, so the file is copied with a few extra "# framing_..." header
    lines. There are no boxes to tell which words a cut would remove, so
    zoom-in / crop may only cut the card's blank margin: the card's ink
    (text, barcode, any dark print) is found and a variant that pushes any
    of it out of frame is rejected. The share of the card still visible is
    written to the header and the manifest too.
    Output mirrors the input: <out>/front/<name>.png + <name>.gt.txt.
  - YOLO boxes (<labels>/<card>.txt). Box corners go through the homography
    and the axis-aligned box around them is written (a 30 degree tilt gives
    a looser box). Boxes partly out of frame are clipped; under
    MIN_VISIBILITY they are dropped. Output: <out>/images/, <out>/labels/.

Only the CUTTING_OPS may push the card (or a box) out of frame; any other op
that does is rejected as a bug. The card's 4 corners in every output image
are written to the manifest either way.

Usage:
    python "image degradation/generate_framing.py"
    python "image degradation/generate_framing.py" --images "path/to/annotated-dataset" \\
        --out synthetic/framing
    python "image degradation/generate_framing.py" \\
        --images synthetic/rendered/images --labels synthetic/rendered/labels \\
        --out synthetic/framing_rendered
"""
# cv2 is a compiled extension, so pylint can't see its members and flags every
# cv2.* call as "no-member" even though they all exist at runtime.
# pylint: disable=no-member
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SEED = 7
random.seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = ["name_bn", "name_en", "guardian", "mother", "dob", "nid"]

# Annotated real cards: front/ and back/, each <card>.png + <card>.gt.txt.
DATASET_DIR = Path("C:/Users/sadman/Downloads/annotated-dataset - Copy")
YOLO_LABELS = Path("dataset/real/train/labels")
OUT_ROOT = Path("synthetic/framing")
# Optional: drop real desk/table/hand photos in here and they get used as
# backgrounds instead of the procedural ones (random crop, any size).
BACKGROUNDS_DIR = Path("synthetic/backgrounds")
GT_SUFFIX = ".gt.txt"

SINGLE_OP_COUNT = 5      # variants with exactly 1 op
CHAIN_VARIANT_COUNT = 3  # variants with a random 2-3 op combo
VARIANTS_PER_CARD = SINGLE_OP_COUNT + CHAIN_VARIANT_COUNT
WRITE_PREVIEWS = True    # box / card-outline overlays in <out>/preview

# Canvas ("the photo") relative to the card, before any 90/180 turn. 1.35x
# leaves room for the card to tilt / shift without it already being cut off.
CANVAS_SCALE = 1.35
# How much of the canvas the (rotated/warped) card fills when no zoom op is
# picked -- a normal, reasonably framed phone photo.
DEFAULT_FILL = (0.78, 0.92)
# Longest side of the output photo. The dataset has scans up to 4096px wide;
# zoom-out on those would otherwise make ~14000px photos. Over this, the
# card is shrunk to fit instead.
MAX_CANVAS_SIDE = 3000

# How far zoom-in / crop may push the card past the frame edge, as a share of
# the card's width/height per cut edge. Only the blank margin may leave the
# frame: a .gt.txt transcription can't say which words got cut off. How wide
# that margin is differs per card (the back's address and MRZ start ~2% from
# the edge on tightly cropped cards), so on top of this range every cut is
# checked against the card's own ink -- see ink_points().
CUT_RANGE = (0.02, 0.08)
# Ink = pixels this much darker than the local card background (0-255).
INK_CONTRAST = 50
# Ignore ink in this outer band of the card: scan/crop slivers along the edge.
INK_EDGE_IGNORE = 0.01
# A cut may take at most this share of the card's ink out of frame (specks).
INK_CUT_TOLERANCE = 0.001
MAX_INK_POINTS = 30000   # ink pixels sampled per card for the check
SINGLE_OP_RETRIES = 6    # fresh values tried per single op before moving to another op

MIN_VISIBILITY = 0.6     # share of a YOLO box that must stay in frame to keep it
# Safety net for the cutting ops: share of the card that must stay in frame.
MIN_CARD_VISIBLE = 0.7
# Same pixel floor as generate_degraded.py / qa_check.py. Their 1%-of-image
# area rule is left out on purpose: zoom-out makes every box a small share
# of the photo by design, while the text itself stays just as many px tall.
TINY_HEIGHT_PX = 10


# -----------------------------------------------------------------------------
# Ops. Each one only sets entries in a params dict; the dict is turned into a
# single transform by build_transform(), so chaining ops never resamples the
# image twice.
# -----------------------------------------------------------------------------
def op_rotate_small(p):
    """5-30 degrees either way."""
    p["angle"] = random.choice([-1, 1]) * random.uniform(5, 30)


def op_rotate_90(p):
    """Sideways: the whole photo turned a quarter turn, either direction."""
    p["quarter_turns"] = random.choice([1, 3])


def op_rotate_180(p):
    """Upside down: the whole photo turned a half turn."""
    p["quarter_turns"] = 2


def op_perspective(p):
    """Photo taken from an angle: tilt the card back/forward or sideways.
    One edge of the card gets shorter (further from the camera) and the
    opposite edge longer, plus a little independent corner jitter."""
    p["tilt_axis"] = random.choice(["x", "y"])
    p["tilt"] = random.choice([-1, 1]) * random.uniform(0.08, 0.22)
    p["corner_jitter"] = random.uniform(0.0, 0.03)


def op_zoom_out(p):
    """Card small in the frame."""
    p["fill"] = random.uniform(0.30, 0.55)


def op_zoom_in(p):
    """Card bigger than the frame: its edges get cut off on both sides."""
    # fill = 1 / (1 - 2 * cut) loses `cut` of the card at each of two edges.
    p["fill"] = 1 / (1 - 2 * random.uniform(*CUT_RANGE))


def op_crop_partial(p):
    """The card's margin pushed out past one or two edges of the frame."""
    p["shift"] = "out"
    # Share of the card's width/height that leaves the frame.
    p["out_amount"] = random.uniform(*CUT_RANGE)
    # One edge, or a corner (one horizontal + one vertical edge) -- never
    # left AND right, which would just cancel out.
    sides = [random.choice(["left", "right"]), random.choice(["top", "bottom"])]
    p["out_sides"] = random.sample(sides, random.choice([1, 1, 2]))


def op_off_center(p):
    """Card fully in the frame, just not in the middle of it."""
    p["shift"] = "in"


OPS = {
    "rotate_small": op_rotate_small,
    "rotate_90_ext": op_rotate_90,
    "rotate_180_ext": op_rotate_180,
    "perspective": op_perspective,
    "zoom_out": op_zoom_out,
    "zoom_in": op_zoom_in,
    "crop_partial": op_crop_partial,
    "off_center": op_off_center,
}
ALL_OP_NAMES = list(OPS)

# Unrealistic / less likely cases are named with an "_ext" suffix (phones
# auto-orient, so sideways or upside-down card photos are rare). Any variant
# using one gets "_ext" on its file name too, so they can be split out.
EXT_SUFFIX = "_ext"
EXT_OPS = {op for op in OPS if op.endswith(EXT_SUFFIX)}

# Ops that are allowed to push the card / boxes out of frame. Anything else
# doing so means the transform is wrong, so the variant is rejected.
CUTTING_OPS = {"zoom_in", "crop_partial"}

# A chain may use at most one op from each group: the ops inside a group set
# the same parameter, so the second one would silently overwrite the first.
CONFLICT_FAMILIES = [
    {"rotate_90_ext", "rotate_180_ext"},
    {"zoom_out", "zoom_in"},
    {"crop_partial", "off_center"},
]


def make_picks(n_ops):
    """Random ops for a chain, never more than one per CONFLICT_FAMILIES group."""
    pool = ALL_OP_NAMES.copy()
    random.shuffle(pool)
    chosen, used_families = [], set()
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


# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
def load_yolo_labels(txt_path):
    """YOLO label file -> [(cls, xc, yc, w, h)], boxes clamped to [0, 1]."""
    boxes = []
    with open(txt_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            cls, xc, yc, w, h = line.split()
            cls, xc, yc, w, h = int(cls), float(xc), float(yc), float(w), float(h)
            x0, y0 = min(max(xc - w / 2, 0.0), 1.0), min(max(yc - h / 2, 0.0), 1.0)
            x1, y1 = min(max(xc + w / 2, 0.0), 1.0), min(max(yc + h / 2, 0.0), 1.0)
            boxes.append((cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return boxes


def save_yolo_labels(txt_path, boxes):
    """[(cls, xc, yc, w, h)] -> YOLO label file."""
    with open(txt_path, "w", encoding="utf-8") as fh:
        for c, xc, yc, w, h in boxes:
            fh.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def save_gt_copy(src_path, dst_path, extra_header):
    """Copy a .gt.txt unchanged, with extra '# key: value' lines appended to
    its leading '#' header block (the field lines below are untouched)."""
    lines = src_path.read_text(encoding="utf-8").splitlines(keepends=True)
    n_header = 0
    while n_header < len(lines) and lines[n_header].startswith("#"):
        n_header += 1
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    if n_header and not lines[n_header - 1].endswith(("\n", "\r")):
        lines[n_header - 1] += newline
    extra = [f"# {k}: {v}{newline}" for k, v in extra_header.items()]
    dst_path.write_text("".join(lines[:n_header] + extra + lines[n_header:]), encoding="utf-8")


# -----------------------------------------------------------------------------
# Backgrounds
# -----------------------------------------------------------------------------
def _load_background_files():
    if not BACKGROUNDS_DIR.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in BACKGROUNDS_DIR.rglob("*") if p.suffix.lower() in exts)


BACKGROUND_FILES = _load_background_files()


def _random_surface_colour():
    # Desks, tables, cloth, paper: mostly muted, occasionally dark or bright.
    base = random.choice([
        (60, 90, 130), (40, 60, 85), (150, 160, 170), (200, 205, 210),
        (30, 30, 35), (90, 110, 120), (120, 140, 170), (70, 70, 70),
    ])
    return np.array([np.clip(c + random.randint(-25, 25), 0, 255) for c in base], np.float32)


def procedural_background(h, w):
    """A plain, gradient, wood-grain or fabric surface with uneven lighting."""
    kind = random.choice(["plain", "gradient", "wood", "fabric"])
    colour = _random_surface_colour()
    bg = np.ones((h, w, 3), np.float32) * colour

    if kind == "gradient":
        ang = random.uniform(0, 2 * np.pi)
        yy, xx = np.indices((h, w), dtype=np.float32)
        t = np.cos(ang) * xx / w + np.sin(ang) * yy / h
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        bg *= (random.uniform(0.6, 0.9) + random.uniform(0.2, 0.5) * t)[..., None]
    elif kind == "wood":
        yy, _ = np.indices((h, w), dtype=np.float32)
        noise = np.random.standard_normal((h, w)).astype(np.float32)
        warp = cv2.GaussianBlur(noise, (0, 0), 25) * 40
        freq = random.uniform(0.03, 0.08)
        grain = np.sin((yy + warp) * freq) * 0.5 + 0.5
        bg *= (0.8 + 0.25 * grain)[..., None]
        if random.random() < 0.5:
            bg = np.ascontiguousarray(np.transpose(bg, (1, 0, 2)))
            bg = cv2.resize(bg, (w, h))
    elif kind == "fabric":
        noise = np.random.standard_normal((h, w)).astype(np.float32)
        weave = cv2.GaussianBlur(noise, (0, 0), 1.2)
        bg *= (1 + 0.12 * weave)[..., None]

    # Uneven lighting + sensor grain on every kind.
    vignette = cv2.GaussianBlur(np.random.rand(8, 8).astype(np.float32), (0, 0), 2)
    vignette = cv2.resize(vignette, (w, h), interpolation=cv2.INTER_CUBIC)
    bg *= (0.85 + 0.3 * vignette)[..., None]
    bg += np.random.normal(0, 3, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def make_background(h, w):
    """Random crop of a BACKGROUNDS_DIR photo (if any), else a procedural one."""
    if BACKGROUND_FILES and random.random() < 0.8:
        src = cv2.imread(str(random.choice(BACKGROUND_FILES)))
        if src is not None:
            sh, sw = src.shape[:2]
            scale = max(h / sh, w / sw) * random.uniform(1.0, 1.5)
            src = cv2.resize(src, (int(np.ceil(sw * scale)), int(np.ceil(sh * scale))))
            y0 = random.randint(0, src.shape[0] - h)
            x0 = random.randint(0, src.shape[1] - w)
            return src[y0:y0 + h, x0:x0 + w].copy()
    return procedural_background(h, w)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def _apply_h(hom, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, hom).reshape(-1, 2)


def _translate(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float64)


def _card_corners(card_w, card_h):
    return np.array([[0, 0], [card_w, 0], [card_w, card_h], [0, card_h]], np.float64)


def _perspective(p, corners, card_w, card_h):
    """Homography that tilts the card as described by op_perspective()."""
    dst = corners.copy()
    if "tilt" in p:
        t = p["tilt"]
        if p["tilt_axis"] == "x":   # top or bottom edge recedes
            far = [0, 1] if t > 0 else [2, 3]
            dx = abs(t) * card_w / 2
            dst[far[0], 0] += dx if far[0] in (0, 3) else -dx
            dst[far[1], 0] += dx if far[1] in (0, 3) else -dx
            # Foreshortening: the tilted card also looks shorter.
            dst[:, 1] = (dst[:, 1] - card_h / 2) * (1 - abs(t) * 0.35) + card_h / 2
        else:                       # left or right edge recedes
            far = [0, 3] if t > 0 else [1, 2]
            dy = abs(t) * card_h / 2
            dst[far[0], 1] += dy if far[0] in (0, 1) else -dy
            dst[far[1], 1] += dy if far[1] in (0, 1) else -dy
            dst[:, 0] = (dst[:, 0] - card_w / 2) * (1 - abs(t) * 0.35) + card_w / 2
        j = p.get("corner_jitter", 0.0)
        dst += np.random.uniform(-j, j, dst.shape) * [card_w, card_h]
    persp = cv2.getPerspectiveTransform(corners.astype(np.float32), dst.astype(np.float32))
    return persp.astype(np.float64)


def build_transform(p, card_w, card_h):
    """params -> (3x3 homography card px -> canvas px, canvas (w, h))."""
    fill = p.get("fill", random.uniform(*DEFAULT_FILL))
    # Zoom-out grows the canvas instead of shrinking the card, so the card
    # keeps roughly its own resolution (a phone photo with the card far away
    # is a bigger photo, not a lower-res card) and the text stays above the
    # TINY_HEIGHT_PX floor -- up to MAX_CANVAS_SIDE.
    canvas_mult = CANVAS_SCALE * max(1.0, DEFAULT_FILL[0] / fill)
    canvas_mult = min(canvas_mult, MAX_CANVAS_SIDE / max(card_w, card_h))
    canvas_w, canvas_h = int(round(card_w * canvas_mult)), int(round(card_h * canvas_mult))
    corners = _card_corners(card_w, card_h)

    # 1. Perspective, in card coordinates.
    persp = _perspective(p, corners, card_w, card_h)

    # 2. In-plane rotation about the card centre (positive = counter-clockwise on screen).
    a = np.deg2rad(p.get("angle", 0.0))
    rot = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float64)
    mat = rot @ _translate(-card_w / 2, -card_h / 2) @ persp

    # 3. Scale so the card's bounding box fills `fill` of the canvas.
    quad = _apply_h(mat, corners)
    bw, bh = np.ptp(quad[:, 0]), np.ptp(quad[:, 1])
    s = fill * min(canvas_w / bw, canvas_h / bh)
    mat = np.diag([s, s, 1.0]) @ mat
    quad = _apply_h(mat, corners)
    cx = (quad[:, 0].min() + quad[:, 0].max()) / 2
    cy = (quad[:, 1].min() + quad[:, 1].max()) / 2
    bw, bh = bw * s, bh * s

    # 4. Placement. Default: centred, with a few px of hand wobble.
    tx = canvas_w / 2 - cx + random.uniform(-0.02, 0.02) * canvas_w
    ty = canvas_h / 2 - cy + random.uniform(-0.02, 0.02) * canvas_h
    shift = p.get("shift")
    if shift == "in":
        # Use up most of the slack on a random side, but stay fully in frame.
        slack_x, slack_y = max(canvas_w - bw, 0) / 2, max(canvas_h - bh, 0) / 2
        tx = canvas_w / 2 - cx + random.choice([-1, 1]) * random.uniform(0.5, 0.95) * slack_x
        ty = canvas_h / 2 - cy + random.choice([-1, 1]) * random.uniform(0.5, 0.95) * slack_y
    elif shift == "out":
        for side in p["out_sides"]:
            if side in ("left", "right"):
                overshoot = (canvas_w - bw) / 2 + p["out_amount"] * bw
                tx = canvas_w / 2 - cx + (-overshoot if side == "left" else overshoot)
            else:
                overshoot = (canvas_h - bh) / 2 + p["out_amount"] * bh
                ty = canvas_h / 2 - cy + (-overshoot if side == "top" else overshoot)
    mat = _translate(tx, ty) @ mat
    return mat, (canvas_w, canvas_h)


def composite(card, mat, canvas_size):
    """Warp the card onto a fresh background, with a soft shadow under it."""
    canvas_w, canvas_h = canvas_size
    bg = make_background(canvas_h, canvas_w).astype(np.float32)
    h, w = card.shape[:2]

    warped = cv2.warpPerspective(card, mat, (canvas_w, canvas_h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), mat, (canvas_w, canvas_h),
                               flags=cv2.INTER_LINEAR)

    # Soft contact shadow under the card so it doesn't look pasted on.
    shadow = cv2.GaussianBlur(mask, (0, 0), max(canvas_w, canvas_h) * 0.012)
    off = np.float32([[1, 0, canvas_w * random.uniform(0.003, 0.012)],
                      [0, 1, canvas_h * random.uniform(0.005, 0.02)]])
    shadow = cv2.warpAffine(shadow, off, (canvas_w, canvas_h))
    bg *= (1 - random.uniform(0.25, 0.5) * shadow)[..., None]

    out = warped.astype(np.float32) * mask[..., None] + bg * (1 - mask[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def ink_points(img):
    """Pixel coords (card px) of the card's ink: anything clearly darker than
    the card background around it -- text strokes, barcode bars, MRZ. Large
    dark areas (the photo) blend into the background estimate and are not
    counted; they aren't part of the transcription anyway."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = max(3, (w // 40) | 1)   # wider than a text stroke
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    ink = (bg.astype(np.int16) - gray) > INK_CONTRAST
    ex, ey = int(w * INK_EDGE_IGNORE), int(h * INK_EDGE_IGNORE)
    ink[:ey], ink[h - ey:] = False, False
    ink[:, :ex], ink[:, w - ex:] = False, False
    ys, xs = np.nonzero(ink)
    pts = np.stack([xs, ys], axis=1).astype(np.float64) + 0.5
    if len(pts) > MAX_INK_POINTS:
        pts = pts[np.random.choice(len(pts), MAX_INK_POINTS, replace=False)]
    return pts


def _in_frame(quad, canvas_size):
    """-> (share of the quad's area inside the canvas, intersection polygon or None)."""
    canvas_w, canvas_h = canvas_size
    frame = np.array([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]], np.float32)
    quad = np.asarray(quad, np.float32)
    full_area = cv2.contourArea(quad)
    inter_area, inter_poly = cv2.intersectConvexConvex(quad, frame)
    vis = inter_area / full_area if full_area > 0 else 0.0
    return vis, inter_poly


def transform_labels(boxes, mat, card_w, card_h, canvas_size):
    """-> (kept YOLO boxes in canvas coords, [(class, visibility)] dropped, any_clipped)."""
    canvas_w, canvas_h = canvas_size
    kept, dropped, any_clipped = [], [], False
    for cls, xc, yc, w, h in boxes:
        x0, y0 = (xc - w / 2) * card_w, (yc - h / 2) * card_h
        x1, y1 = (xc + w / 2) * card_w, (yc + h / 2) * card_h
        quad = _apply_h(mat, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        vis, inter_poly = _in_frame(quad, canvas_size)
        if vis < MIN_VISIBILITY or inter_poly is None:
            dropped.append((cls, vis))
            continue
        if vis < 0.999:
            any_clipped = True
        # Box around the part of the field that is actually in frame.
        pts = inter_poly.reshape(-1, 2)
        bx0, by0 = max(pts[:, 0].min(), 0), max(pts[:, 1].min(), 0)
        bx1, by1 = min(pts[:, 0].max(), canvas_w), min(pts[:, 1].max(), canvas_h)
        kept.append((cls, (bx0 + bx1) / 2 / canvas_w, (by0 + by1) / 2 / canvas_h,
                     (bx1 - bx0) / canvas_w, (by1 - by0) / canvas_h))
    return kept, dropped, any_clipped


def rotate_quarter(img, boxes, quad, k):
    """np.rot90 k times counter-clockwise; boxes (normalised) and the card
    quad (pixels) follow exactly."""
    out = np.ascontiguousarray(np.rot90(img, k))
    new = []
    for c, xc, yc, w, h in boxes:
        for _ in range(k):
            xc, yc, w, h = yc, 1 - xc, h, w   # one CCW quarter turn in normalised coords
        new.append((c, xc, yc, w, h))
    quad = np.asarray(quad, np.float64)
    cur_w, cur_h = img.shape[1], img.shape[0]
    for _ in range(k):
        quad = np.stack([quad[:, 1], cur_w - quad[:, 0]], axis=1)   # (x, y) -> (y, W - x)
        cur_w, cur_h = cur_h, cur_w
    return out, new, quad


def validate(res, op_names):
    """-> (True, None) if the variant is usable, else (False, reason)."""
    cutting = bool(CUTTING_OPS & set(op_names))
    if not cutting and res["card_visible"] < 0.999:
        return False, f"non-cutting op pushed the card out of frame ({res['card_visible']:.2f})"
    if res["card_visible"] < MIN_CARD_VISIBLE:
        return False, f"only {res['card_visible']:.2f} of the card left in frame"
    if res["ink_cut"] > INK_CUT_TOLERANCE:
        return False, f"cut would take {res['ink_cut'] * 100:.1f}% of the card's ink out of frame"
    if res["boxes"] is None:      # text-label card: nothing else to check
        return True, None
    if not cutting and (res["dropped"] or res["any_clipped"]):
        lost = [CLASS_NAMES[c] for c, _ in res["dropped"]]
        return False, f"non-cutting op lost/clipped boxes ({lost})"
    if not res["boxes"]:
        return False, "no boxes left in frame"
    img_h = res["image"].shape[0]
    for c, _, _, _, bh in res["boxes"]:
        if bh * img_h < TINY_HEIGHT_PX:
            return False, f"{CLASS_NAMES[c]} box too tiny ({bh * img_h:.1f}px tall)"
    return True, None


def draw_preview(img, boxes, quad):
    """Image with the card outline, plus each box and its class name if any."""
    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, round(max(h, w) / 600))
    cv2.polylines(out, [np.round(quad).astype(np.int32)], True, (0, 255, 255), thick)
    colours = [(0, 0, 255), (0, 160, 255), (0, 200, 0), (255, 120, 0), (255, 0, 200), (200, 200, 0)]
    for c, xc, yc, bw, bh in boxes or []:
        colour = colours[c % len(colours)]
        p0 = (int((xc - bw / 2) * w), int((yc - bh / 2) * h))
        p1 = (int((xc + bw / 2) * w), int((yc + bh / 2) * h))
        cv2.rectangle(out, p0, p1, colour, thick)
        cv2.putText(out, CLASS_NAMES[c], (p0[0], max(p0[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return out


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def make_variant(img, boxes, op_names, ink=None):
    """Apply a list of op names to one card. `boxes` is None for text-label
    cards; `ink` (from ink_points) is checked against the frame if given.
    -> dict with image, boxes, dropped, any_clipped, quad, card_visible,
    ink_cut, params."""
    card_h, card_w = img.shape[:2]
    params = {}
    for op in op_names:
        OPS[op](params)
    mat, canvas_size = build_transform(params, card_w, card_h)
    out_img = composite(img, mat, canvas_size)
    quad = _apply_h(mat, _card_corners(card_w, card_h))
    card_visible, _ = _in_frame(quad, canvas_size)
    ink_cut = 0.0
    if ink is not None and len(ink) and card_visible < 0.999:
        pts = _apply_h(mat, ink)
        canvas_w, canvas_h = canvas_size
        outside = ((pts[:, 0] < 0) | (pts[:, 0] >= canvas_w)
                   | (pts[:, 1] < 0) | (pts[:, 1] >= canvas_h))
        ink_cut = float(outside.mean())
    out_boxes, dropped, any_clipped = None, [], False
    if boxes is not None:
        out_boxes, dropped, any_clipped = transform_labels(boxes, mat, card_w, card_h,
                                                           canvas_size)
    k = params.get("quarter_turns", 0)
    if k:
        out_img, out_boxes, quad = rotate_quarter(out_img, out_boxes or [], quad, k)
        if boxes is None:
            out_boxes = None
    return {"image": out_img, "boxes": out_boxes, "dropped": dropped,
            "any_clipped": any_clipped, "quad": quad, "card_visible": card_visible,
            "ink_cut": ink_cut, "params": params}


def _fmt_params(p):
    return ";".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in sorted(p.items()))


def _fmt_quad(quad):
    return ";".join(f"{x:.0f},{y:.0f}" for x, y in quad)


def find_label(img_path, labels_dir):
    """-> ("gt", path) for a <card>.gt.txt next to the image, ("yolo", path) for
    <labels_dir>/<card>.txt, else None."""
    gt = img_path.with_name(img_path.stem + GT_SUFFIX)
    if gt.exists():
        return "gt", gt
    yolo = labels_dir / f"{img_path.stem}.txt"
    if yolo.exists():
        return "yolo", yolo
    return None


def generate_variants(image_path, label, out_root, rel_dir):
    """Write all variants for one card; returns their manifest rows."""
    label_type, label_path = label
    source_id = image_path.stem
    img = cv2.imread(str(image_path))
    boxes = load_yolo_labels(label_path) if label_type == "yolo" else None
    # YOLO cards don't need it: clipped/dropped boxes are relabelled exactly.
    ink = ink_points(img) if label_type == "gt" else None
    ext = image_path.suffix.lower()

    if label_type == "gt":
        images_dir = labels_dir = out_root / rel_dir   # same layout as the dataset
    else:
        images_dir, labels_dir = out_root / "images" / rel_dir, out_root / "labels" / rel_dir
    preview_dir = out_root / "preview" / rel_dir
    for d in (images_dir, labels_dir, preview_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows, variant_idx, attempts = [], 1, 0
    max_attempts = VARIANTS_PER_CARD * 10
    n_singles = min(SINGLE_OP_COUNT, len(ALL_OP_NAMES))
    n_wanted = n_singles + CHAIN_VARIANT_COUNT

    def try_save(op_names):
        nonlocal variant_idx
        res = make_variant(img, boxes, op_names, ink)
        ok, reason = validate(res, op_names)
        if not ok:
            print(f"  [{source_id}] dropped {op_names}: {reason}")
            return False
        name = f"{source_id}_frm_{variant_idx:02d}"
        if EXT_OPS & set(op_names):
            name += EXT_SUFFIX
        cv2.imwrite(str(images_dir / f"{name}{ext}"), res["image"])
        if label_type == "gt":
            save_gt_copy(label_path, labels_dir / f"{name}{GT_SUFFIX}", {
                "framing_image": f"{name}{ext}",
                "framing_source": source_id,
                "framing_ops": "+".join(op_names),
                "framing_ext": "yes" if name.endswith(EXT_SUFFIX) else "no",
                "framing_card_visible": f"{res['card_visible']:.3f}",
                "framing_card_quad": _fmt_quad(res["quad"]),
            })
        else:
            save_yolo_labels(labels_dir / f"{name}.txt", res["boxes"])
        if WRITE_PREVIEWS:
            cv2.imwrite(str(preview_dir / f"{name}.jpg"),
                        draw_preview(res["image"], res["boxes"], res["quad"]))
        dropped_names = "+".join(f"{CLASS_NAMES[c]}({v:.2f})" for c, v in res["dropped"])
        n_boxes = "" if res["boxes"] is None else len(res["boxes"])
        rows.append([(rel_dir / f"{name}{ext}").as_posix(), source_id, label_type,
                     "+".join(op_names), f"{res['card_visible']:.3f}", n_boxes, dropped_names,
                     _fmt_quad(res["quad"]), _fmt_params(res["params"]), SEED])
        variant_idx += 1
        return True

    # Phase 1: one op each, all different. A failed single op is retried with
    # fresh random values; if it keeps failing (e.g. zoom-in on a card whose
    # text runs right up to every edge) the next op in the shuffled list
    # takes its slot.
    order = random.sample(ALL_OP_NAMES, len(ALL_OP_NAMES))
    n_single_done = 0
    for op in order:
        if n_single_done >= n_singles:
            break
        for _ in range(SINGLE_OP_RETRIES):
            attempts += 1
            if try_save([op]):
                n_single_done += 1
                break
        else:
            print(f"  [{source_id}] gave up on {op} after {SINGLE_OP_RETRIES} tries")

    # Phase 2: random 2-3 op combos.
    n_chain_done = 0
    while n_chain_done < CHAIN_VARIANT_COUNT and attempts < max_attempts:
        attempts += 1
        if try_save(make_picks(random.randint(2, 3))):
            n_chain_done += 1

    if len(rows) < n_wanted:
        print(f"  [{source_id}] WARNING: only generated {len(rows)}/{n_wanted}")
    return rows


def _read_gt_header(gt_path):
    """Leading `# key: value` lines of a .gt.txt -> dict."""
    header = {}
    with open(gt_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            header[key.strip()] = value.strip()
    return header


def check_gt_pairs(out_root):
    """Every output image must have exactly one .gt.txt with the same stem
    (and vice versa), whose header names that image, and whose _ext flag and
    ops agree with the file name. -> list of problems (empty = consistent)."""
    problems = []
    img_exts = {".png", ".jpg", ".jpeg"}
    for d in sorted({p.parent for p in out_root.rglob(f"*{GT_SUFFIX}")}):
        if "preview" in d.relative_to(out_root).parts:
            continue
        images = {p.stem: p for p in d.iterdir() if p.suffix.lower() in img_exts}
        gts = {p.name[:-len(GT_SUFFIX)]: p for p in d.glob(f"*{GT_SUFFIX}")}
        for stem in sorted(images.keys() - gts.keys()):
            problems.append(f"{images[stem]}: no {GT_SUFFIX}")
        for stem in sorted(gts.keys() - images.keys()):
            problems.append(f"{gts[stem]}: no image")
        for stem in sorted(images.keys() & gts.keys()):
            hdr = _read_gt_header(gts[stem])
            is_ext = stem.endswith(EXT_SUFFIX)
            ops = set(hdr.get("framing_ops", "").split("+"))
            if hdr.get("framing_image") != images[stem].name:
                problems.append(f"{gts[stem]}: framing_image={hdr.get('framing_image')!r}")
            if hdr.get("framing_ext") != ("yes" if is_ext else "no"):
                problems.append(f"{gts[stem]}: framing_ext={hdr.get('framing_ext')!r}")
            if bool(EXT_OPS & ops) != is_ext:
                problems.append(f"{gts[stem]}: ops {sorted(ops)} vs _ext in name")
    return problems


def main():
    """Parse args, wipe the previous output, generate every card's variants."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    ap.add_argument("--images", type=Path, default=DATASET_DIR,
                    help="folder searched recursively for cards (png/jpg)")
    ap.add_argument("--labels", type=Path, default=YOLO_LABELS,
                    help="YOLO label folder, for cards without a .gt.txt next to them")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="only the first N cards (0 = all)")
    args = ap.parse_args()

    # Wipe the previous run so stale variants don't sit next to new ones --
    # but only a folder this script made (it has our manifest), never some
    # other non-empty folder passed as --out by mistake.
    manifest_path = args.out / "manifest.csv"
    if args.out.exists() and any(args.out.iterdir()):
        if not manifest_path.exists():
            sys.exit(f"{args.out} is not empty and has no manifest.csv from a previous run "
                     "-- refusing to wipe it. Pick another --out.")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    if BACKGROUND_FILES:
        print(f"Using {len(BACKGROUND_FILES)} background photos from {BACKGROUNDS_DIR}"
              " (+ procedural)")

    all_rows = []
    images = sorted(p for p in args.images.rglob("*")
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        images = images[:args.limit]
    if not images:
        print(f"No images found in {args.images}")
    for img_path in images:
        label = find_label(img_path, args.labels)
        if label is None:
            print(f"Skipping {img_path.name}: no {GT_SUFFIX} next to it and no label in "
                  f"{args.labels}")
            continue
        if label[0] == "yolo":
            try:
                n_boxes = len(load_yolo_labels(label[1]))
            except ValueError as e:
                print(f"Skipping {img_path.name}: malformed label file ({label[1].name}) -- {e}")
                continue
            if n_boxes != 6:
                print(f"Skipping {img_path.name}: expected 6 label lines, "
                      f"found {n_boxes} in {label[1].name}")
                continue
        print(f"Generating framing variants for {img_path.name}...")
        rel_dir = img_path.parent.relative_to(args.images)
        # Broad on purpose (same as generate_degraded.py): one bad card is
        # logged and skipped instead of killing the whole run.
        try:
            all_rows.extend(generate_variants(img_path, label, args.out, rel_dir))
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Skipping {img_path.name}: failed while generating variants -- {e}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["output_file", "source_card", "label_type", "operations",
                         "card_visible", "n_boxes", "dropped_boxes", "card_quad", "params",
                         "seed"])
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} variants written to {args.out}.")

    problems = check_gt_pairs(args.out)
    if problems:
        print(f"\nIMAGE / {GT_SUFFIX} MISMATCHES ({len(problems)}):")
        for msg in problems:
            print(f"  {msg}")
        sys.exit(1)
    print(f"Checked: every image has a matching {GT_SUFFIX}.")


if __name__ == "__main__":
    main()
