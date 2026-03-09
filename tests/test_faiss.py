"""
FAISS index build/search sanity check using synthetic embeddings.
"""

import os
import unittest

import numpy as np

from services.search_service import SearchService


class TestFaiss(unittest.TestCase):
    def setUp(self):
        self.index_path = "index/test_faiss.index"
        meta = f"{self.index_path}.meta.json"
        for path in (self.index_path, meta):
            if os.path.isfile(path):
                os.remove(path)

    def tearDown(self):
        meta = f"{self.index_path}.meta.json"
        for path in (self.index_path, meta):
            if os.path.isfile(path):
                os.remove(path)

    def test_build_and_search(self):
        embeddings = {
            "alice": [self._random_vec(), self._random_vec()],
            "bob": [self._random_vec()],
        }
        service = SearchService(index_path=self.index_path)
        service.build_faiss_index(embeddings)
        query = embeddings["alice"][0]
        results = service.search_similar(query, k=2)
        self.assertGreater(len(results), 0)
        # Best match should be alice itself
        self.assertEqual(results[0][0], "alice")

    @staticmethod
    def _random_vec():
        vec = np.random.randn(512).astype("float32")
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec


if __name__ == "__main__":
    unittest.main()
