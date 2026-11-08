"""Cut a 360 panorama into the flat views a detector is run on.

    python demo/cut_panorama.py <equirectangular.jpg> [out_dir] [--size 640] [--fov 90]

Writes one JPEG per view (`view_00.jpg` ...) and `contact_sheet.jpg`, all
sixteen views labelled with where they point. Needs only numpy and Pillow:
the projection is `src/vision/projection.py`, the same code the detectors use.

Any 2:1 equirectangular image works. Poly Haven (https://polyhaven.com/hdris)
publishes indoor panoramas under CC0 if you need one to try.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.projection import default_views, render  # noqa: E402


def describe(lat_deg: float) -> str:
    if lat_deg >= 89:
        return "straight up"
    if lat_deg <= -89:
        return "straight down"
    if lat_deg == 0:
        return "horizon"
    return f"{'up' if lat_deg > 0 else 'down'} {abs(lat_deg):.0f} deg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama")
    ap.add_argument("out", nargs="?", default="views")
    ap.add_argument("--size", type=int, default=640, help="view width and height in pixels")
    ap.add_argument("--fov", type=float, default=90.0, help="horizontal field of view in degrees")
    args = ap.parse_args()

    pano = np.asarray(Image.open(args.panorama).convert("RGB"))
    h, w = pano.shape[:2]
    if abs(w - 2 * h) > 2:
        raise SystemExit(f"expected a 2:1 equirectangular image, got {w} x {h}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    views = default_views(size=args.size, fov_deg=args.fov)
    tiles = []
    for i, spec in enumerate(views):
        img = Image.fromarray(render(pano, spec))
        img.save(out / f"view_{i:02d}.jpg", quality=90)
        tiles.append((img, f"{i + 1}: {describe(spec.lat_deg)}, lon {spec.lon_deg:.0f}"))

    cols, t = 4, args.size // 2
    sheet = Image.new("RGB", (cols * t, (len(tiles) + cols - 1) // cols * t), (7, 8, 10))
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(tiles):
        x, y = (i % cols) * t, (i // cols) * t
        sheet.paste(img.resize((t - 4, t - 4)), (x + 2, y + 2))
        draw.rectangle((x + 2, y + 2, x + t - 3, y + 18), fill=(7, 8, 10))
        draw.text((x + 6, y + 4), label, fill=(238, 240, 242))
    sheet.save(out / "contact_sheet.jpg", quality=90)
    print(f"{len(views)} views of {args.fov:.0f} deg at {args.size} px -> {out}")


if __name__ == "__main__":
    main()
