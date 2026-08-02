import unittest

import numpy as np

from utils.graspnet_target import normalize_binary_mask, projected_points_in_mask


class NormalizeBinaryMaskTest(unittest.TestCase):
    def test_resizes_and_binarizes_mask(self) -> None:
        source = np.array([[0.0, 0.8], [0.0, 0.0]], dtype=np.float32)

        mask = normalize_binary_mask(source, (4, 4))

        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(set(np.unique(mask)), {0, 1})
        self.assertEqual(int(mask.sum()), 4)


class ProjectedPointsInMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.K = np.array(
            [
                [100.0, 0.0, 5.0],
                [0.0, 100.0, 5.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def test_keeps_only_centers_projected_inside_sam_mask(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[4:7, 4:7] = 1
        translations = np.array(
            [
                [0.0, 0.0, 1.0],  # (5, 5)，位于 mask 内
                [0.03, 0.0, 1.0],  # (8, 5)，位于 mask 外
                [0.0, 0.0, -1.0],  # 相机后方
                [np.nan, 0.0, 1.0],  # 无效坐标
            ]
        )

        keep = projected_points_in_mask(translations, self.K, mask)

        np.testing.assert_array_equal(keep, [True, False, False, False])

    def test_margin_dilates_mask_for_small_alignment_error(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1
        translations = np.array([[0.01, 0.0, 1.0]])  # 投影到 (6, 5)

        strict = projected_points_in_mask(translations, self.K, mask)
        tolerant = projected_points_in_mask(translations, self.K, mask, margin_px=1)

        np.testing.assert_array_equal(strict, [False])
        np.testing.assert_array_equal(tolerant, [True])


if __name__ == "__main__":
    unittest.main()
