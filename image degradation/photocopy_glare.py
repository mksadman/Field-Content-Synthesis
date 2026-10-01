import cv2
import random
import numpy as np


def photocopy_effect(img):
    """Old photocopy/scan look: crushed contrast, toner speckle, faded ink.
    No moderate/heavy split -- strength is sampled from one continuous range
    every time this runs, so two calls won't look identical."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    alpha = random.uniform(1.2, 1.7)
    beta = random.randint(-35, -15)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    speckle_amount = random.uniform(0.008, 0.03)
    speckle_mask = np.random.rand(*gray.shape) < speckle_amount
    gray[speckle_mask] = np.random.randint(0, 60, size=speckle_mask.sum())

    fade = random.uniform(0.05, 0.12)
    gray = cv2.addWeighted(gray, 1 - fade, np.full_like(gray, 255), fade, 0)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def salt_pepper_noise(img):
    """Scatters random pure-white and pure-black pixels, like impulse noise."""
    amount = random.uniform(0.008, 0.025)
    out = img.copy()
    h, w = out.shape[:2]
    n = int(amount * h * w * 0.5)
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    out[ys, xs] = 255
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    out[ys, xs] = 0
    return out


def low_light_effect(img):
    """Darkens the image and adds sensor noise, like a badly lit phone photo."""
    alpha = random.uniform(0.55, 0.78)
    beta = random.randint(-22, -8)
    noise_std = random.uniform(8, 16)
    dark = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    noise = np.random.normal(0, noise_std, dark.shape).astype(np.int16)
    return np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def white_balance_tint(img, warm=True):
    """Shifts the whole image toward a warm (orange) or cool (blue) tint."""
    strength = random.uniform(1.08, 1.2)
    inv = 1 / strength
    out = img.astype(np.float32)
    if warm:
        out[:, :, 2] *= strength   # boost red (BGR order)
        out[:, :, 0] *= inv        # reduce blue
    else:
        out[:, :, 0] *= strength   # boost blue
        out[:, :, 2] *= inv        # reduce red
    return np.clip(out, 0, 255).astype(np.uint8)


def soft_shadow(img):
    """A soft-edged dark polygon cast across part of the card."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    pts = np.array([[np.random.randint(0, w), np.random.randint(0, h)] for _ in range(4)])
    cv2.fillPoly(mask, [pts], 1.0)
    mask = cv2.GaussianBlur(mask, (61, 61), 0)
    darken = random.uniform(0.25, 0.5)
    out = img.astype(np.float32) * (1 - mask[..., None] * darken)
    return np.clip(out, 0, 255).astype(np.uint8)


def finger_phone_shadow(img, shape="finger"):
    """A shadow shaped like a finger (thin oval poking in from an edge) or a
    phone (a rectangle blocking part of the card). Both are now randomized
    across the WHOLE card -- finger can come from any of the 4 edges, phone
    rectangle can land anywhere -- instead of always top edge / left half."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    if shape == "finger":
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            cx, cy, axes = random.randint(0, w), 0, (w // 10, h // 2)
        elif edge == "bottom":
            cx, cy, axes = random.randint(0, w), h, (w // 10, h // 2)
        elif edge == "left":
            cx, cy, axes = 0, random.randint(0, h), (w // 2, h // 10)
        else:  # right
            cx, cy, axes = w, random.randint(0, h), (w // 2, h // 10)
        cv2.ellipse(mask, (cx, cy), axes, np.random.randint(0, 180), 0, 360, 1.0, -1)
    else:  # phone
        rw, rh = w // 3, h // 2
        x0 = random.randint(0, max(0, w - rw))
        y0 = random.randint(0, max(0, h - rh))
        cv2.rectangle(mask, (x0, y0), (x0 + rw, y0 + rh), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    darken = random.uniform(0.3, 0.55)
    out = img.astype(np.float32) * (1 - mask[..., None] * darken)
    return np.clip(out, 0, 255).astype(np.uint8)


def glossy_streak(img):
    """Laminated-card style: a bright diagonal reflection band. Position,
    angle, and width are randomized each call so it can land anywhere on the
    card. Unlike a simple brightness add (which still leaves dark text
    readable underneath), this BLENDS toward solid white at the band's
    core -- strong enough to genuinely erase whatever text is under it,
    like a real glare blowout off a laminated surface, with a soft falloff
    back to the original image at the edges."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    angle = random.uniform(-70, 70)  # degrees from horizontal
    cx, cy = random.randint(0, w), random.randint(0, h)  # band can center anywhere
    # random length: sometimes a short, contained patch, sometimes long
    # enough to fully cross the card corner-to-corner (previously this was
    # always fixed at 1.5x the card's longest side, so it ALWAYS ran off
    # both edges -- that's why every streak looked like a full stripe).
    length = int(random.uniform(0.25, 1.4) * max(w, h))
    dx = int(length / 2 * np.cos(np.radians(angle)))
    dy = int(length / 2 * np.sin(np.radians(angle)))
    pt1 = (cx - dx, cy - dy)
    pt2 = (cx + dx, cy + dy)
    thickness = random.randint(int(0.06 * min(w, h)), int(0.22 * min(w, h)))
    cv2.line(mask, pt1, pt2, 1.0, thickness)
    mask = cv2.GaussianBlur(mask, (35, 35), 0)

    # peak_opacity close to 1 -> the band's core blends almost fully to
    # white regardless of how dark the underlying ink was.
    peak_opacity = random.uniform(0.7, 1.0)
    alpha = (mask * peak_opacity)[..., None]
    white = np.full_like(img, 255, dtype=np.float32)
    out = img.astype(np.float32) * (1 - alpha) + white * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def hologram_patch(img):
    """Smart-card style: an iridescent rainbow-tinted patch."""
    h, w = img.shape[:2]
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    radius = random.randint(min(w, h) // 6, min(w, h) // 4)
    hue_band = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hue_band, (cx, cy), radius, 255, -1)
    hue_band = cv2.GaussianBlur(hue_band, (45, 45), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    rainbow_hue = np.random.randint(0, 179)
    mask_norm = hue_band.astype(np.float32) / 255.0
    sat_boost = random.uniform(40, 80)
    hsv[..., 0] = (hsv[..., 0] * (1 - mask_norm) + rainbow_hue * mask_norm).astype(np.int16)
    hsv[..., 1] = np.clip(hsv[..., 1] + (mask_norm * sat_boost), 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def framing_margin(img, labels, pad_frac=0.04):
    """Adds extra white background margin around the card."""
    h, w = img.shape[:2]
    pad = int(pad_frac * max(h, w))
    out = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    nh, nw = out.shape[:2]
    new_labels = [(c, (xc * w + pad) / nw, (yc * h + pad) / nh, bw * w / nw, bh * h / nh)
                  for c, xc, yc, bw, bh in labels]
    return out, new_labels


def framing_overcrop(img, labels, crop_frac=0.03):
    """Crops slightly into the card edge, like a badly framed photo.
    Any box that would get clipped by the crop is dropped, per Rule 5."""
    h, w = img.shape[:2]
    x0, y0 = int(crop_frac * w), int(crop_frac * h)
    x1, y1 = w - x0, h - y0
    out = img[y0:y1, x0:x1]
    nh, nw = out.shape[:2]
    new_labels = []
    for c, xc, yc, bw, bh in labels:
        ax0, ay0 = xc * w - bw * w / 2 - x0, yc * h - bh * h / 2 - y0
        ax1, ay1 = xc * w + bw * w / 2 - x0, yc * h + bh * h / 2 - y0
        if ax0 < 0 or ay0 < 0 or ax1 > nw or ay1 > nh:
            continue  # clipped -> drop, per Rule 5
        new_labels.append((c, (ax0 + ax1) / 2 / nw, (ay0 + ay1) / 2 / nh, (ax1 - ax0) / nw, (ay1 - ay0) / nh))
    return out, new_labels
