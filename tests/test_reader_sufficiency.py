import unittest

from src.reader_sufficiency import (
    CHAIN_STRUCTURE_FEATURE_NAMES,
    READER_SUFFICIENCY_FEATURE_NAMES,
    chain_structure_features,
    parse_evidence_chain,
    reader_sufficiency_features,
)


class ReaderSufficiencyTests(unittest.TestCase):
    def test_parse_chain_requires_structured_json(self) -> None:
        self.assertEqual(
            parse_evidence_chain('{"evidence_chain":["Alpha links to Beta"],"answer":"Beta"}'),
            ("Alpha links to Beta",),
        )
        self.assertEqual(parse_evidence_chain("Alpha links to Beta"), ())

    def test_features_capture_grounding_and_answer_stability(self) -> None:
        values = reader_sufficiency_features(
            "Where was Alpha's director born?",
            "Seoul",
            ("Alpha was directed by Beta", "Beta was born in Seoul"),
            (
                {"title": "Alpha", "text": "Alpha was directed by Beta."},
                {"title": "Beta", "text": "Beta was born in Seoul."},
            ),
            prior_predictions=("unknown", "Seoul"),
            prior_chains=(("Alpha was directed by Beta",), ("Beta was born in Seoul",)),
        )
        by_name = dict(zip(READER_SUFFICIENCY_FEATURE_NAMES, values, strict=True))
        self.assertEqual(by_name["answer_is_unknown"], 0.0)
        self.assertGreater(by_name["answer_document_fraction"], 0.0)
        self.assertEqual(by_name["previous_answer_exact"], 1.0)
        self.assertGreater(by_name["chain_average_document_support"], 0.0)

    def test_unknown_is_observable_without_gold(self) -> None:
        values = reader_sufficiency_features("Who?", "unknown", (), ())
        by_name = dict(zip(READER_SUFFICIENCY_FEATURE_NAMES, values, strict=True))
        self.assertEqual(by_name["answer_is_unknown"], 1.0)
        self.assertEqual(by_name["chain_is_empty"], 1.0)
        self.assertFalse(any("gold" in name or "dataset" in name for name in by_name))

    def test_comparison_requires_explicit_facts_for_both_branches(self) -> None:
        question = "Which film has the director born later, Film Alpha or Film Beta?"
        incomplete = chain_structure_features(
            question,
            ("Film Alpha was directed by Alice, who was born in 1950.", "Film Beta"),
        )
        complete = chain_structure_features(
            question,
            (
                "Film Alpha was directed by Alice, who was born in 1950.",
                "Film Beta was directed by Bob, who was born in 1960.",
            ),
        )
        incomplete_by_name = dict(zip(CHAIN_STRUCTURE_FEATURE_NAMES, incomplete, strict=True))
        complete_by_name = dict(zip(CHAIN_STRUCTURE_FEATURE_NAMES, complete, strict=True))
        self.assertEqual(incomplete_by_name["comparison_branches_complete"], 0.0)
        self.assertEqual(complete_by_name["comparison_branches_complete"], 1.0)

    def test_answer_must_match_question_type_and_explicit_clause(self) -> None:
        values = chain_structure_features(
            "How many times did the plague occur?",
            ("The plague occurred 22 times in Venice.",),
            "Venice",
        )
        by_name = dict(zip(CHAIN_STRUCTURE_FEATURE_NAMES, values, strict=True))
        self.assertEqual(by_name["answer_in_explicit_clause"], 1.0)
        self.assertEqual(by_name["answer_type_compatible"], 0.0)


if __name__ == "__main__":
    unittest.main()
