import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protocollens.provenance import file_sha256


class ProvenanceTests(unittest.TestCase):
    def test_streaming_sha256_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.txt"
            path.write_bytes(b"ProtocolLens evidence")
            first = file_sha256(path, chunk_size=3)
            second = file_sha256(path, chunk_size=1024)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
