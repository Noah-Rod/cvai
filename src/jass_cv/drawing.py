"""Draw detections onto images (PIL only, so the frontend doesn't need OpenCV)."""

from __future__ import annotations

from collections.abc import Iterable

from PIL import Image, ImageDraw, ImageFont

from jass_cv.schemas import Detection

SUIT_COLORS = {
    "eicheln": (46, 139, 87),
    "schilten": (30, 144, 255),
    "rosen": (220, 20, 60),
    "schellen": (255, 165, 0),
}


def draw_detections(image: Image.Image, detections: Iterable[Detection]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    line = max(2, round(min(out.size) / 300))
    font_px = max(14, round(min(out.size) / 35))
    try:
        font = ImageFont.load_default(size=font_px)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for d in detections:
        color = SUIT_COLORS.get(d.suit, (255, 255, 255))
        b = d.bbox
        if d.polygon:
            pts = [tuple(pt) for pt in d.polygon]
            draw.line([*pts, pts[0]], fill=color, width=line, joint="curve")
        else:
            draw.rectangle((b.x1, b.y1, b.x2, b.y2), outline=color, width=line)
        text = f"{d.display_name} {d.confidence:.2f}"
        text_h = draw.textbbox((0, 0), text, font=font)[3]
        tx, ty = b.x1, max(0, b.y1 - text_h - 2 * line)
        tb = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle((tb[0] - line, tb[1] - line, tb[2] + line, tb[3] + line), fill=color)
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    return out
