import unittest

import numpy as np

from modules.image_editor import editor_to_inpaint


class TestImageEditor(unittest.TestCase):
    def test_cleared_editor(self):
        self.assertIsNone(editor_to_inpaint(None))
        self.assertIsNone(editor_to_inpaint({"background": None, "layers": []}))

    def test_uploaded_image_with_no_strokes(self):
        background = np.full((2, 3, 4), 255, dtype=np.uint8)
        converted = editor_to_inpaint({"background": background, "layers": []})
        np.testing.assert_array_equal(converted["image"], background[:, :, :3])
        np.testing.assert_array_equal(converted["mask"], np.zeros((2, 3, 3), dtype=np.uint8))

    def test_stroke_masks_use_alpha_regardless_of_color(self):
        background = np.zeros((2, 3, 4), dtype=np.uint8)
        black_brush = background.copy()
        black_brush[0, 0] = [0, 0, 0, 255]
        red_brush = background.copy()
        red_brush[1, 2] = [255, 0, 0, 200]
        converted = editor_to_inpaint({"background": background, "layers": [black_brush, red_brush]})
        expected = np.zeros((2, 3, 3), dtype=np.uint8)
        expected[0, 0] = 255
        expected[1, 2] = 200
        np.testing.assert_array_equal(converted["mask"], expected)
