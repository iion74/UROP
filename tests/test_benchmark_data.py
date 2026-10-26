import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark_data import (
    iter_json_array,
    normalise_2wiki_record,
    normalise_hotpot_record,
    normalise_musique_record,
    validate_example,
)
from experiments.preprocess_multihop import _load_2wiki_aliases, _sample_records


class BenchmarkDataTests(unittest.TestCase):
    def test_hotpot_preserves_sentences_and_groups_duplicate_titles(self):
        row = {
        "id": "h1",
        "question": "Who founded it?",
        "answer": "Ada",
        "type": "bridge",
        "level": "hard",
        "context": {
            "title": ["Target", "Target", "Bridge"],
            "sentences": [["Ada founded it."], ["A second passage."], ["See Target."]],
        },
        "supporting_facts": {"title": ["Bridge", "Target"], "sent_id": [0, 0]},
        }

        example = normalise_hotpot_record(row, "validation")
        validate_example(example)

        self.assertEqual(example["hop_count"], 2)
        self.assertEqual(example["documents"][0]["sentences"], ["Ada founded it."])
        target = next(support for support in example["gold_supports"] if support["title"] == "Target")
        self.assertEqual(len(target["doc_ids"]), 2)


    def test_musique_uses_paragraph_idx_instead_of_ambiguous_title(self):
        row = {
        "id": "2hop__1_2",
        "question": "Who is the spouse of the performer?",
        "answer": "Miquette Giraudy",
        "answer_aliases": [],
        "answerable": True,
        "paragraphs": [
            {"idx": 2, "title": "Green", "paragraph_text": "A colour.", "is_supporting": False},
            {"idx": 10, "title": "Green", "paragraph_text": "Performed by Steve Hillage.", "is_supporting": True},
            {"idx": 5, "title": "Steve Hillage", "paragraph_text": "Spouse Miquette.", "is_supporting": True},
        ],
        "question_decomposition": [
            {"id": 1, "question": "Green >> performer", "answer": "Steve Hillage", "paragraph_support_idx": 10},
            {"id": 2, "question": "Who is #1's spouse?", "answer": "Miquette Giraudy", "paragraph_support_idx": 5},
        ],
        }

        example = normalise_musique_record(row, "train")
        validate_example(example)

        self.assertEqual(example["hop_count"], 2)
        self.assertEqual([support["paragraph_idx"] for support in example["gold_supports"]], [10, 5])
        self.assertEqual(example["gold_supports"][0]["doc_ids"], ["musique:2hop__1_2:p10"])
        self.assertNotEqual(example["documents"][0]["content_id"], example["documents"][1]["content_id"])


    def test_2wiki_retains_relation_chain_and_four_support_documents(self):
        row = {
        "_id": "w1",
        "type": "bridge_comparison",
        "question": "Which director was born later?",
        "answer": "Film A",
        "answer_id": "Q1",
        "entity_ids": "Q2_Q3",
        "context": [[f"Doc {index}", [f"Sentence {index}."]] for index in range(1, 5)],
        "supporting_facts": [[f"Doc {index}", 0] for index in range(1, 5)],
        "evidences": [["Film A", "director", "Person A"], ["Person A", "birth", "1920"]],
        "evidences_id": [["Q2", "director", "Q3"], ["Q3", "birth", "1920"]],
        }

        example = normalise_2wiki_record(
            row,
            "train",
            answer_aliases_by_id={"Q1": ("Movie A", "Film Alpha")},
        )
        validate_example(example)

        self.assertEqual(example["hop_count"], 2)
        self.assertEqual(example["support_document_count"], 4)
        self.assertEqual(example["reasoning_chain"][0]["relation"], "director")
        self.assertEqual(example["metadata"]["reasoning_step_count"], 2)
        self.assertEqual(example["answer_aliases"], ["Movie A", "Film Alpha"])

    def test_2wiki_alias_loader_combines_aliases_and_demonyms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "id_aliases.json"
            path.write_text(
                '{"Q_id":"Q16","aliases":["Canada","CAN"],"demonyms":["Canadian","Canada"]}\n',
                encoding="utf-8",
            )
            aliases = _load_2wiki_aliases(path)

        self.assertEqual(aliases["Q16"], ("Canada", "CAN", "Canadian"))


    def test_validation_rejects_missing_gold_document(self):
        row = {
        "id": "broken",
        "question": "Question?",
        "answer": "Answer",
        "context": {"title": ["A"], "sentences": [["Text."]]},
        "supporting_facts": {"title": ["Missing"], "sent_id": [0]},
        }
        example = normalise_hotpot_record(row, "train")

        with self.assertRaisesRegex(ValueError, "후보 문서와 연결되지"):
            validate_example(example)

    def test_invalid_raw_sentence_label_is_reported_without_losing_document_gold(self):
        row = {
            "id": "typo",
            "question": "Question?",
            "answer": "Answer",
            "context": {"title": ["A"], "sentences": [["First.", "Second."]]},
            "supporting_facts": {"title": ["A", "A"], "sent_id": [0, 902]},
        }

        example = normalise_hotpot_record(row, "validation")
        validate_example(example)

        support = example["gold_supports"][0]
        self.assertEqual(support["sentence_ids"], [0])
        self.assertEqual(support["invalid_sentence_ids"], [902])


    def test_stream_json_array_handles_objects_across_small_chunks(self):
        rows = [{"id": index, "text": "x" * 30} for index in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            loaded = list(iter_json_array(path, chunk_size=17))

        self.assertEqual(loaded, rows)

    def test_stream_sample_is_reproducible_and_not_only_a_prefix(self):
        rows = [{"id": index} for index in range(100)]

        first = list(_sample_records(iter(rows), 10, 42))
        second = list(_sample_records(iter(rows), 10, 42))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertNotEqual({row["id"] for row in first}, set(range(10)))


if __name__ == "__main__":
    unittest.main()
