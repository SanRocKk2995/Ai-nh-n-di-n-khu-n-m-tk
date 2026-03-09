"""
Recognition test using local sample images (skips if samples are absent).
"""

import os
import unittest

import cv2

from services.face_service import FaceService


class TestRecognize(unittest.TestCase):
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

    def test_recognize_registered_face(self):
        images = self._load_samples()
        if images is None:
            self.skipTest("Add sample images under tests/data/ to run this test.")

        service = FaceService()
        service.delete_person("recognize_user")
        service.register_person("recognize_user", "Recognize User", images)

        result = service.recognize_face(images[0])
        self.assertEqual(result.get("person_id"), "recognize_user")


if __name__ == "__main__":
    unittest.main()
