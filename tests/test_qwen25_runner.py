"""Identity and logical-prefix boundaries independent of model weights."""
import unittest

from PIL import Image

from mmimpress.qwen25.runner import _pil_rgb_sha256


class ImageIdentityTests(unittest.TestCase):
    def test_same_rgb_bytes_with_different_geometry_are_distinct(self):
        raw = bytes([10, 20, 30, 40, 50, 60])
        wide = Image.frombytes("RGB", (2, 1), raw)
        tall = Image.frombytes("RGB", (1, 2), raw)
        self.assertNotEqual(_pil_rgb_sha256(wide), _pil_rgb_sha256(tall))
        self.assertEqual(_pil_rgb_sha256(wide), _pil_rgb_sha256(wide.copy()))


if __name__ == "__main__":
    unittest.main()
