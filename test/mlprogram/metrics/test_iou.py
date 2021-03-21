import numpy as np

from mlprogram.metrics import Iou


class TestIou(object):
    def test_simple_case(self):
        iou = Iou()
        gt = np.array([False, True, False], dtype=np.bool)
        assert np.allclose(
            0.5,
            iou(expected=gt,
                actual=np.array([True, True, False], dtype=np.bool)
                ))

    def test_gt_is_empty(self):
        iou = Iou()
        gt = np.array([False, False, False], dtype=np.bool)
        assert np.allclose(
            1.0 / 3,
            iou(expected=gt,
                actual=np.array([True, True, False], dtype=np.bool)
                ))
        assert np.allclose(
            1.0,
            iou(expected=gt,
                actual=np.array([False, False, False], dtype=np.bool)
                ))
