"""devc4-test2: spinner (software startup/loading screen) logo is the cartoon sun asset."""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SPINNER = ROOT / "system" / "ui" / "spinner.py"
ASSET = ROOT / "sunnypilot" / "selfdrive" / "assets" / "images" / "spinner_sunny_pilot_sun.png"


def test_spinner_uses_sun_asset():
  src = SPINNER.read_text()
  assert '"../../sunnypilot/selfdrive/assets/images/spinner_sunny_pilot_sun.png", TEXTURE_SIZE, TEXTURE_SIZE' in src
  assert "spinner_sunnypilot.png" not in src


def test_sun_asset_square_transparent_centered():
  im = Image.open(ASSET)
  assert im.mode == "RGBA"
  w, h = im.size
  assert w == h  # loaded as TEXTURE_SIZE x TEXTURE_SIZE, so square keeps the aspect ratio
  px = im.load()
  assert px[0, 0][3] == 0 and px[w - 1, h - 1][3] == 0  # transparent bg
  # alpha bounding box of non-transparent pixels
  box = im.getbbox()
  assert box is not None
  left, top, right, bottom = box
  bw, bh = right - left, bottom - top
  assert abs((left + right) / 2 - w / 2) <= 2 and abs((top + bottom) / 2 - h / 2) <= 2  # centered in ring
  assert 0.65 <= bw / w <= 0.85  # fits inside the rotating track with margin
  c = px[w // 2, int(h * 0.4)]
  assert c[3] > 200 and c[0] > 200 and c[1] > 150 and c[2] < 120  # yellow sun, not the white sunnypilot mark
