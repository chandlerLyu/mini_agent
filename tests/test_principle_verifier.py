from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import ModelConfig
from interfaces import Message
from models.deterministic import DeterministicModel
from principles.jsonl import read_jsonl, write_jsonl
from principles.schema import Chunk, Principle, PrincipleCandidate, VerificationRecord
from principles.verifier import build_verified_principle, resolve_evidence_chunks, verify_principle
from scripts.verify_principles import _verify_with_progress


class RaisingModel:
    def query(self, *, messages, tools, config):
        raise RuntimeError("provider timed out")


def candidate(principle_id: str = "P0001", chunk_id: str = "c1") -> PrincipleCandidate:
    return PrincipleCandidate.model_validate(
        {
            "principle_id": principle_id,
            "name": "Feedback mechanisms require boundaries",
            "domain": "financial_reasoning",
            "summary": "When beliefs affect financing conditions, feedback can reinforce fundamentals until a constraint breaks the loop.",
            "when_to_apply": ["belief-driven markets"],
            "how_to_apply": ["identify belief", "trace financing feedback", "look for constraints"],
            "failure_modes": ["ignoring policy intervention"],
            "evidence": [{"source_file": "book.pdf", "chunk_id": chunk_id, "page": 10}],
            "confidence": 0.8,
        }
    )


def chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_file="book.pdf",
        page=10,
        text="Market beliefs can change financing conditions, which can reinforce prices until constraints intervene.",
    )


def verifier_json(*, decision: str = "verified", revised: dict | None = None) -> str:
    revision_instruction = '"narrow scope"' if revised is not None else "null"
    return (
        "{"
        '"principle_id":"P0001",'
        '"principle_type":"mechanism",'
        '"type_confidence":"high",'
        f'"decision":"{decision}",'
        '"decomposition":{"condition":"beliefs affect financing","outcome":"feedback reinforces fundamentals","mechanism":"financing access changes behavior"},'
        '"evidence_bindings":[{"principle_part":"mechanism","source_chunk_id":"c1","support_type":"direct","explanation":"The source links beliefs, financing, and reinforcement."}],'
        '"clarity":{"label":"pass","comment":"The conditional mechanism is clear."},'
        '"reasoning_link":{"label":"strong","comment":"The evidence directly supports the mechanism."},'
        '"generalization_audit":{"label":"appropriate","comment":"The scope is limited to belief-driven markets.","suggested_scope":null},'
        '"actionability":{"label":"actionable","comment":"It gives concrete analysis steps."},'
        '"boundary_awareness":{"label":"sufficient","comment":"It names constraints and intervention."},'
        '"counterexample_check":{"has_counterexample":true,"counterexample":"Policy intervention can break the loop.","needed_boundary_condition":"Check intervention risk."},'
        f'"revision_needed":{str(revised is not None).lower()},'
        f'"revision_instruction":{revision_instruction},'
        f'"revised_principle":{_json_or_null(revised)},'
        '"verifier_rationale":"Supported, scoped, and actionable."'
        "}"
    )


def _json_or_null(value: dict | None) -> str:
    if value is None:
        return "null"
    import json

    return json.dumps(value)


class PrincipleVerifierTests(unittest.TestCase):
    def test_resolve_evidence_chunks_by_id(self) -> None:
        self.assertEqual(resolve_evidence_chunks(candidate(), {"c1": chunk()})[0].chunk_id, "c1")
        self.assertEqual(resolve_evidence_chunks(candidate(chunk_id="missing"), {"c1": chunk()}), [])

    def test_verifier_accepts_extra_model_fields(self) -> None:
        model = DeterministicModel([Message(role="assistant", content=verifier_json()[:-1] + ', "extra":"ignored"}')])

        result = verify_principle(
            candidate=candidate(),
            evidence_chunks=[chunk()],
            model=model,
            config=ModelConfig(),
        )

        self.assertEqual(result.decision, "verified")
        self.assertEqual(result.principle_type, "mechanism")

    def test_verifier_extracts_json_from_wrapped_response(self) -> None:
        model = DeterministicModel(
            [Message(role="assistant", content="Here is the verification:\n```json\n" + verifier_json() + "\n```")]
        )

        result = verify_principle(
            candidate=candidate(),
            evidence_chunks=[chunk()],
            model=model,
            config=ModelConfig(),
        )

        self.assertEqual(result.decision, "verified")

    def test_verifier_treats_string_revised_principle_as_instruction(self) -> None:
        raw = verifier_json(decision="needs_revision").replace(
            '"revised_principle":null',
            '"revised_principle":"Narrow the WTO claim to trade disputes with human-rights-sensitive import restrictions."',
        )
        model = DeterministicModel([Message(role="assistant", content=raw)])

        result = verify_principle(
            candidate=candidate(),
            evidence_chunks=[chunk()],
            model=model,
            config=ModelConfig(),
        )

        self.assertEqual(result.decision, "needs_revision")
        self.assertIsNone(result.revised_principle)
        self.assertIn("Narrow the WTO claim", result.revision_instruction or "")

    def test_build_verified_principle_from_revision(self) -> None:
        revised = {
            "summary": "When market beliefs affect financing conditions, feedback can reinforce fundamentals in belief-driven markets until constraints or intervention break the loop.",
            "failure_modes": ["policy intervention breaks feedback", "financing constraints bind"],
        }
        result = verify_principle(
            candidate=candidate(),
            evidence_chunks=[chunk()],
            model=DeterministicModel([Message(role="assistant", content=verifier_json(decision="needs_revision", revised=revised))]),
            config=ModelConfig(),
        )

        principle = build_verified_principle(candidate(), result)

        self.assertEqual(principle.status, "needs_revision")
        self.assertIn("belief-driven markets", principle.summary)
        self.assertEqual(principle.evidence[0].chunk_id, "c1")

    def test_verify_cli_helper_writes_verified_and_rejected_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [candidate("P0001"), candidate("P0002")]
            chunks_by_id = {"c1": chunk()}
            model = DeterministicModel(
                [
                    Message(role="assistant", content=verifier_json(decision="verified")),
                    Message(role="assistant", content=verifier_json(decision="reclassify_as_observation").replace("P0001", "P0002")),
                ]
            )

            state = _verify_with_progress(
                candidates=candidates,
                chunks_by_id=chunks_by_id,
                model=model,
                config=ModelConfig(),
                output_path=root / "principles.jsonl",
                results_path=root / "results.jsonl",
                rejected_path=root / "rejected.jsonl",
                errors_path=root / "errors.jsonl",
                initial_verified=[],
                initial_results=[],
                initial_rejected=[],
                start_principle_number=1,
                total_principles=2,
                checkpoint_every=1,
                progress_every=1,
                verbose=False,
                overwrite=True,
            )

            self.assertEqual(len(state["verified"]), 1)
            self.assertEqual(len(state["rejected"]), 1)
            self.assertEqual(len(read_jsonl(root / "principles.jsonl", Principle)), 1)
            self.assertEqual(len(read_jsonl(root / "results.jsonl", VerificationRecord)), 2)
            self.assertEqual(len(read_jsonl(root / "rejected.jsonl", VerificationRecord)), 1)

    def test_verify_cli_helper_records_bad_response_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = DeterministicModel(
                [
                    Message(role="assistant", content="not json"),
                    Message(role="assistant", content=verifier_json(decision="verified").replace("P0001", "P0002")),
                ]
            )

            state = _verify_with_progress(
                candidates=[candidate("P0001"), candidate("P0002")],
                chunks_by_id={"c1": chunk()},
                model=model,
                config=ModelConfig(),
                output_path=root / "principles.jsonl",
                results_path=root / "results.jsonl",
                rejected_path=root / "rejected.jsonl",
                errors_path=root / "errors.jsonl",
                initial_verified=[],
                initial_results=[],
                initial_rejected=[],
                start_principle_number=1,
                total_principles=2,
                checkpoint_every=1,
                progress_every=1,
                verbose=False,
                overwrite=True,
            )

            self.assertEqual(state["errors"], 1)
            self.assertEqual(len(state["verified"]), 1)
            self.assertIn('"principle_id": "P0001"', (root / "errors.jsonl").read_text(encoding="utf-8"))

    def test_verify_cli_helper_records_provider_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            state = _verify_with_progress(
                candidates=[candidate("P0001")],
                chunks_by_id={"c1": chunk()},
                model=RaisingModel(),
                config=ModelConfig(),
                output_path=root / "principles.jsonl",
                results_path=root / "results.jsonl",
                rejected_path=root / "rejected.jsonl",
                errors_path=root / "errors.jsonl",
                initial_verified=[],
                initial_results=[],
                initial_rejected=[],
                start_principle_number=1,
                total_principles=1,
                checkpoint_every=1,
                progress_every=1,
                verbose=False,
                overwrite=True,
            )

            self.assertEqual(state["errors"], 1)
            self.assertEqual(state["verified"], [])
            self.assertIn("provider timed out", (root / "errors.jsonl").read_text(encoding="utf-8"))

    def test_verify_cli_helper_preserves_initial_outputs_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing_result = verify_principle(
                candidate=candidate("P0001"),
                evidence_chunks=[chunk()],
                model=DeterministicModel([Message(role="assistant", content=verifier_json(decision="verified"))]),
                config=ModelConfig(),
            )
            existing_principle = build_verified_principle(candidate("P0001"), existing_result)
            existing_record = VerificationRecord(candidate=candidate("P0001"), result=existing_result)

            state = _verify_with_progress(
                candidates=[candidate("P0002")],
                chunks_by_id={"c1": chunk()},
                model=DeterministicModel([Message(role="assistant", content=verifier_json(decision="verified").replace("P0001", "P0002"))]),
                config=ModelConfig(),
                output_path=root / "principles.jsonl",
                results_path=root / "results.jsonl",
                rejected_path=root / "rejected.jsonl",
                errors_path=root / "errors.jsonl",
                initial_verified=[existing_principle],
                initial_results=[existing_record],
                initial_rejected=[],
                start_principle_number=2,
                total_principles=2,
                checkpoint_every=1,
                progress_every=1,
                verbose=False,
                overwrite=False,
            )

            self.assertEqual([principle.principle_id for principle in state["verified"]], ["P0001", "P0002"])
            self.assertEqual([record.candidate.principle_id for record in state["results"]], ["P0001", "P0002"])


if __name__ == "__main__":
    unittest.main()
