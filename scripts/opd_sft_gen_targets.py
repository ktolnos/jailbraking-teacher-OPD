"""Teacher completions on the 980 TRAINING prompts, for SFT targets.

Purpose: build (prompt, target) pairs where target = PREFILL + teacher's
continuation, so `opd_sft_train.py` can train the student to EMIT the
prefill itself rather than receive it externally. This is the data half of
the safety question: can ordinary distillation on a teacher's jailbroken
completions transfer the jailbreak behavior (not just content) into a model
that starts every generation from scratch, no scaffolding supplied?

Same wrapper, prefill, and decoding contract as everywhere else in this
project (student_eval.py / opd_arm_scihazard.py / scihazard_conditional_gap
.teacher_gen): greedy, thinking off, 400-token budget, continue_final_message
for the prefill. Reuses the existing sqlite cache and rate limiter, so any
overlap with prior teacher calls costs nothing.

API only -- no GPU. Writes jsonl to $OUTDIR. Prints counts and length
aggregates only -- never question or completion text.
"""
import json, os, sys
import concurrent.futures as cf
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scihazard_conditional_gap import teacher_gen, refused  # noqa: E402
from opd_arm_scihazard import TRAIN_WRAPPER, PREFILL          # noqa: E402
from opd_split import split                                   # noqa: E402

OUT = os.environ.get("OUTDIR", "/scratch/eop/jailbreak-opd")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 980
    train_rows, _eval = split()
    train_rows = train_rows[:n]
    print(f"generating teacher SFT targets on {len(train_rows)} training prompts",
          flush=True)

    def work(r):
        user = TRAIN_WRAPPER.format(q=r["question"])
        txt = teacher_gen(user, PREFILL)
        return r, txt

    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(work, train_rows))

    path = f"{OUT}/sft_targets.jsonl"
    nfail, lens = 0, []
    with open(path, "w") as f:
        for r, txt in res:
            if txt is None or not txt.strip():
                nfail += 1
                continue
            lens.append(len(txt.split()))
            f.write(json.dumps(dict(
                question=r["question"], category=r["category"],
                risk=r.get("risk_level"), completion=txt)) + "\n")
    kept = len(res) - nfail
    print(f"wrote {kept} targets ({nfail} failed/empty) -> {path}", flush=True)
    if lens:
        import statistics as st
        print(f"  completion words: mean={st.mean(lens):.1f} "
              f"median={st.median(lens):.1f} min={min(lens)} max={max(lens)}",
              flush=True)
    refc = sum(1 for _, t in res if t and refused(t))
    print(f"  teacher refused under prefill: {refc}/{len(res)} "
          f"({100*refc/len(res):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
