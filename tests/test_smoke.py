import cv2
import numpy as np

import vehicle_damage


def test_package_imports():
    assert vehicle_damage.__version__ == "0.1.0"


def test_opencv_png_roundtrip():
    image = np.full((32, 48, 3), (0, 128, 255), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    assert decoded.shape == (32, 48, 3)
    assert np.array_equal(decoded, image)
