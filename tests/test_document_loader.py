import unittest

from src.document_loader import SourceDocument, chunk_documents


class ChunkDocumentsTests(unittest.TestCase):
    def test_short_document_produces_one_chunk(self) -> None:
        chunks = chunk_documents([SourceDocument(source="short", text="짧은 문서입니다.")], 900, 140)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "짧은 문서입니다.")

    def test_final_chunk_does_not_repeat_trailing_overlap(self) -> None:
        text = "a" * 1_000
        chunks = chunk_documents([SourceDocument(source="long", text=text)], 900, 140)

        self.assertEqual(len(chunks), 2)
        self.assertEqual([len(chunk.text) for chunk in chunks], [900, 240])

    def test_benchmark_document_keeps_stable_id_and_separate_title(self) -> None:
        chunks = chunk_documents(
            [SourceDocument(source="musique:content-hash", title="Shared Article", text="Benchmark passage.")],
            900,
            140,
        )

        self.assertEqual(chunks[0].source, "musique:content-hash")
        self.assertEqual(chunks[0].title, "Shared Article")


if __name__ == "__main__":
    unittest.main()
