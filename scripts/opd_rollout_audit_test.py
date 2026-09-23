"""CPU-only contract test for the training audit; uses no model or network."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from opd_canonicalize import CanonicalizingIWOPDTrainer
from opd_rollout_summary import summarize


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "rollouts-test.jsonl"
        step_path = Path(d) / "rollout-steps-test.jsonl"
        path.with_suffix(".meta.json").write_text(json.dumps({
            "run_id": "test", "max_steps": 1, "effective_batch_size": 2,
            "step_log": step_path.name,
        }))
        trainer = SimpleNamespace(
            processing_class=_Tokenizer(), audit_run_id="test",
            audit_opener_text="Hic", state=SimpleNamespace(global_step=0),
            _buffer_step=1, args=SimpleNamespace(gradient_accumulation_steps=1),
            accelerator=SimpleNamespace(process_index=0), rollout_log_path=str(path),
        )
        inputs = {
            "prompts": torch.tensor([[0, 65, 66], [67, 68, 69]]),
            "prompt_attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "input_ids": torch.tensor([[0, 65, 66, 72, 105, 0],
                                       [67, 68, 69, 79, 75, 33]]),
            "labels": torch.tensor([[-100, -100, -100, 72, 105, -100],
                                    [-100, -100, -100, 79, 75, 33]]),
            "audit_ended": torch.tensor([True, False]),
            "audit_retokenized": torch.tensor([False, True]),
            "audit_moved": torch.tensor([0, 1]),
            "audit_sampled_lengths": torch.tensor([3, 3]),
            "audit_sampled_ids": torch.tensor([[72, 105, 99], [79, 75, 33]]),
        }
        teacher = {
            "actual_logprobs": torch.tensor([[-.2, -.3, float("-inf")],
                                               [-.4, -.5, -.6]]),
            "topk_token_ids": torch.tensor([[[72, 4], [105, 5], [0, 0]],
                                             [[79, 1], [75, 2], [33, 3]]]),
            "topk_logprobs": torch.tensor([[[-.2, -1.2], [-.3, -1.3],
                                             [float("-inf"), float("-inf")]],
                                            [[-.4, -1.4], [-.5, -1.5],
                                             [-.6, -1.6]]]),
        }
        CanonicalizingIWOPDTrainer._write_rollout_audit(trainer, inputs, teacher)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0]["sampled_completion_ids"] == [72, 105, 99]
        assert rows[0]["completion_ids"] == [72, 105]
        assert rows[0]["starts_with_opener"] is True
        assert [len(r["teacher_actual_logprobs"]) for r in rows] == [2, 3]
        assert [len(x) for x in rows[0]["teacher_topk"]] == [2, 2]

        step_path.write_text(json.dumps({"run_id": "test", "completed_step": 1}) + "\n")
        report = summarize(path, 1)
        assert report["rows"] == 2 and report["committed_steps"] == 1
        assert not report["missing_steps"] and not report["wrong_size_steps"]

        before = path.read_bytes()
        teacher["actual_logprobs"][0, 0] = float("-inf")
        try:
            CanonicalizingIWOPDTrainer._write_rollout_audit(trainer, inputs, teacher)
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing teacher score should fail")
        assert path.read_bytes() == before
    print("rollout audit contract passed")


if __name__ == "__main__":
    main()
