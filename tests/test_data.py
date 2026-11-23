import numpy as np

from vehicle_damage.data import assign_splits, polygon_to_line, read_yolo_polygons, source_name


def test_source_name_drops_roboflow_suffix():
    assert source_name("0001_jpeg_jpg.rf.9df83a165fd4b3946ca7ea91bfe2d0dd.jpg") == "0001_jpeg_jpg"


def test_read_yolo_polygons_scales_to_pixels(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("2 0.1 0.2 0.5 0.2 0.5 0.8\n1 0.1 0.1 0.2 0.2\n", encoding="utf-8")
    polygons = read_yolo_polygons(label, 100, 50)
    assert len(polygons) == 1
    class_id, points = polygons[0]
    assert class_id == 2
    assert points.tolist() == [[10, 10], [50, 10], [50, 40]]
    assert points.dtype == np.int32


def test_polygon_to_line_normalises_and_clips():
    line = polygon_to_line(1, [0, 0, 100, 0, 120, 25], 100, 50)
    assert line == "1 0.000000 0.000000 1.000000 0.000000 1.000000 0.500000"


def test_polygon_to_line_rejects_two_points():
    assert polygon_to_line(0, [0, 0, 10, 10], 100, 100) is None


def test_assign_splits_is_deterministic_and_near_target():
    sizes = {f"img{i}": 1 + i % 3 for i in range(1000)}
    first = assign_splits(sizes, seed=0)
    assert first == assign_splits(sizes, seed=0)
    total = sum(sizes.values())
    train = sum(sizes[k] for k, v in first.items() if v == "train")
    assert abs(train / total - 0.70) < 0.02
    assert set(first.values()) == {"train", "val", "test"}
