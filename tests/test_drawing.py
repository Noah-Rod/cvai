from PIL import Image

from jass_cv.detector import make_detection
from jass_cv.drawing import draw_detections


def test_draws_upright_and_rotated_boxes():
    img = Image.new("RGB", (200, 200), "black")
    dets = [
        make_detection(26, 0.9, (10, 10, 60, 80)),
        make_detection(
            0, 0.8, (100, 100, 160, 170), polygon=[[110, 100], [160, 110], [150, 170], [100, 160]]
        ),
    ]
    out = draw_detections(img, dets)
    assert out.size == img.size
    assert out.getpixel((10, 40)) != (0, 0, 0)  # left edge of the upright box
    assert out.getpixel((105, 130)) != (0, 0, 0)  # left edge of the rotated box
