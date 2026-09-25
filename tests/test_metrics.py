from src.metrics import precision, recall, f05, fbeta

def test_precision():
    assert precision(10, 0) == 1.0
    assert precision(10, 10) == 0.5
    assert precision(0, 10) == 0.0
    assert precision(0, 0) == 0.0

def test_recall():
    assert recall(10, 0) == 1.0
    assert recall(10, 10) == 0.5
    assert recall(0, 10) == 0.0
    assert recall(0, 0) == 0.0

def test_f05():
    # p=1.0, r=0.5
    f = f05(10, 0, 10)
    # (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    assert abs(f - 0.8333) < 0.001
