import numpy as np

from vehicle_damage.draw import draw_polygons


def test_draw_polygons_colours_inside_only_and_keeps_input():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    square = np.array([[20, 20], [60, 20], [60, 60], [20, 60]], dtype=np.int32)
    out = draw_polygons(image, [(0, square)], {0: "Dent"})
    assert image.sum() == 0
    assert out[40, 40].sum() > 0
    assert out[90, 90].sum() == 0
