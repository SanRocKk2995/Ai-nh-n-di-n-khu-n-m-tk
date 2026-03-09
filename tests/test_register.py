"""
Integration-style registration test.
Place sample images in tests/data/person1_1.jpg, person1_2.jpg to enable.
"""

import os
import unittest

import cv2

from services.face_service import FaceService


class TestRegister(unittest.TestCase):
    def setUp(self):
        self.sample_paths = [
            "tests/data/person1_1.jpg",
            "tests/data/person1_2.jpg",
        ]

    def _load_samples(self):
        images = []
        for path in self.sample_paths:
            if not os.path.isfile(path):
                return None
            img = cv2.imread(path)
            if img is None:
                return None
            images.append(img)
        return images

    def test_register_person(self):
        images = self._load_samples()
        if images is None:
            self.skipTest("Add sample images under tests/data/ to run this test.")

        service = FaceService()
        service.delete_person("unittest_user")
        count = service.register_person("unittest_user", "Unit Test", images)
        self.assertGreaterEqual(count, len(images))


if __name__ == "__main__":
    unittest.main()
