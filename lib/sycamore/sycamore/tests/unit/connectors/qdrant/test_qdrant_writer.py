import unittest
from sycamore.connectors.qdrant.qdrant_writer import QdrantWriter

class TestQdrantWriter(unittest.TestCase):

    def test_compatible_with(self):
        writer = QdrantWriter()

        # Test case where both are compatible
        self.assertTrue(writer.compatible_with(QdrantWriter()))

        # Test case where the other writer is not a QdrantWriter
        class DummyWriter:
            pass

        self.assertFalse(writer.compatible_with(DummyWriter()))

        # Test case where the other writer is None
        self.assertFalse(writer.compatible_with(None))

if __name__ == '__main__':
    unittest.main()
