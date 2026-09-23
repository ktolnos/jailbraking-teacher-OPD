"""Gate A: the teacher-student capability gap on WMDP, paired per question.

Both sides answer the identical questions in identical order, so this does a
paired comparison rather than differencing two aggregates: McNemar's exact test
on the discordant items, plus a per-config breakdown.

OPD can only transfer what the teacher has and the student lacks. If the gap is
not clearly positive, the harm arm's transfer story has no headroom regardless
of how well the prefix attack works.
"""
import json, os, sys
from collections import defaultdict
from math import comb

HOME = os.path.expanduser("~/jailbraking-teacher-OPD")
T = f"{HOME}/results/teacher_wmdp.jsonl"
S = os.environ.get("STUDENT_WMDP", "/scratch/eop/jailbreak-opd/student_wmdp.jsonl")


def load(path, tag):
    if not os.path.exists(path):
        sys.exit(f"missing {tag} rows: {path}")
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    # teacher rows carry no idx; order within a config is the dataset order
    for cfg, rs in by.items():
        for i, r in enumerate(rs):
            r.setdefault("idx", i)
    return by


def mcnemar(b, c):
    """Exact two-sided binomial test on discordant pairs."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(p, 1.0)


def main():
    t, s = load(T, "teacher"), load(S, "student")
    cfgs = [c for c in ("wmdp-bio", "wmdp-cyber", "wmdp-chem") if c in t and c in s]
    print(f"{'config':<12}{'teacher':>9}{'student':>9}{'gap':>8}{'T+/S-':>8}{'T-/S+':>8}{'p':>10}")
    print("-" * 64)
    TB = TC = 0
    for cfg in cfgs:
        ti = {r["idx"]: r for r in t[cfg]}
        si = {r["idx"]: r for r in s[cfg]}
        common = sorted(set(ti) & set(si))
        if not common:
            print(f"{cfg:<12}  (no overlapping items)"); continue
        ta = 100 * sum(ti[i]["ok"] for i in common) / len(common)
        sa = 100 * sum(si[i]["ok"] for i in common) / len(common)
        b = sum(1 for i in common if ti[i]["ok"] and not si[i]["ok"])   # teacher only
        c = sum(1 for i in common if si[i]["ok"] and not ti[i]["ok"])   # student only
        TB += b; TC += c
        print(f"{cfg:<12}{ta:>8.1f}%{sa:>8.1f}%{ta-sa:>+7.1f}{b:>8}{c:>8}{mcnemar(b,c):>10.2e}"
              f"   (n={len(common)})")
    print("-" * 64)
    print(f"{'POOLED':<12}{'':>9}{'':>9}{'':>8}{TB:>8}{TC:>8}{mcnemar(TB,TC):>10.2e}")
    print("\nT+/S- = teacher right, student wrong (the headroom OPD could transfer).")
    print("T-/S+ = student right, teacher wrong (no headroom there).")
    if TB <= TC:
        print("\n*** Gate A FAILS: no net capability headroom. ***")
    else:
        print(f"\nGate A headroom: {TB-TC} net items the teacher knows and the student does not.")


if __name__ == "__main__":
    main()
