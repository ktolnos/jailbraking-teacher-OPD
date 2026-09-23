"""Teacher generations on the held-out eval split, in the arm schema.

This measures the CEILING: how much better the frozen 122B teacher is than the
untrained student, on exactly the prompts and under exactly the protocol used
to evaluate the trained arms. That number bounds what any amount of OPD could
have achieved.

  teacher vs clean  ~= trained vs clean   -> the headroom was already captured;
                                             more training or more data cannot
                                             help, and the pairing (not the
                                             dataset) is what must change.
  teacher vs clean  >> trained vs clean   -> headroom exists and OPD is failing
                                             to transfer it: a method problem.

PILOT.md already reported this gap as absent ("Gate B FAILS"), but measured it
with the absolute 1-5 substance scale, which we have since seen saturate, and
with single-order judging, which we have since measured to carry a ~58% slot-1
position bias. So it is worth re-measuring with the better instrument before
treating it as settled.

Decoding matches student_eval/opd_arm_scihazard exactly (greedy, 400 tokens,
thinking off, prefill continued, same wrapper), so the rows pair 1:1.

API only -- no GPU. Aggregates only, never prints text.
"""
import json, os, sys
import concurrent.futures as cf
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scihazard_conditional_gap import teacher_gen, refused  # noqa: E402
from opd_arm_scihazard import TRAIN_WRAPPER, PREFILL        # noqa: E402
from opd_split import split                                 # noqa: E402

OUT = os.environ.get("OUTDIR", "/scratch/eop/jailbreak-opd")
WRAPPER_NAME = "TRAIN (in-distribution)"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 420
    _train, ev = split()
    ev = ev[:n]
    print(f"teacher generations on {len(ev)} held-out prompts", flush=True)

    jobs = [(r, arm) for r in ev for arm in ("prefill", "scratch")]

    def work(job):
        r, arm = job
        user = TRAIN_WRAPPER.format(q=r["question"])
        txt = teacher_gen(user, PREFILL if arm == "prefill" else None)
        return r, arm, txt

    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(work, jobs))

    path = f"{OUT}/arm_scihazard_teacher.jsonl"
    nfail = 0
    with open(path, "w") as f:
        for r, arm, txt in res:
            if txt is None:
                nfail += 1
                continue
            f.write(json.dumps(dict(
                model="Qwen3_5-122B-A10B", tag="teacher",
                question=r["question"], category=r["category"],
                risk=r.get("risk_level"), wrapper=WRAPPER_NAME, arm=arm,
                completion=txt)) + "\n")
    print(f"wrote {len(res) - nfail} generations ({nfail} failed) -> {path}",
          flush=True)

    agg = defaultdict(lambda: [0, 0, 0])
    for r, arm, txt in res:
        if txt is None:
            continue
        a = agg[arm]
        a[0] += refused(txt)
        a[1] += 1
        a[2] += len((txt or "").split())
    for arm in ("scratch", "prefill"):
        ref, cnt, wds = agg[arm]
        if cnt:
            print(f"  {arm:<8} refuse={100*ref/cnt:5.1f}%  "
                  f"mean_words={wds/cnt:6.1f}  (n={cnt})", flush=True)


if __name__ == "__main__":
    main()
