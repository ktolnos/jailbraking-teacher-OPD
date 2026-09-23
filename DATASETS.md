# Dataset survey: what can host the prefix, and what can grade the result

Scope: extends section 2 of [PILOT.md](PILOT.md), and adopts its axis —
**generative tasks can host the prefix -> continuation channel; multiple-choice
benchmarks cannot, and are legitimate only as prefix-free outcome probes.**
Every entry below is filed on that axis first and by topic second.

Requirements this has to meet:

- **Training** — prompts that elicit *free-form* answers to harmful or
  bias-laden requests. Prompts are the requirement; prefills are a bonus.
  Volume matters: step 4 trains prefix emission, step 5 runs OPD rollouts, and
  both want whole categories withheld.
- **Evaluation** — compliance **and** capability / harm intensity, reported
  separately.

**Availability verified 2026-09-15** against the HF datasets-server, the HF repo
API, and a byte-range read of an actual data file with this account's token.
Row counts and column names are measured, not quoted from papers. PILOT.md's
warning applies to bulk loading, not to these checks: use `datasets` or a direct
`resolve/main` fetch, not the datasets-server, once a set is in use.

**Two gotchas that produced wrong answers here, both now controlled for:**

- `HF_TOKEN` is in `~/.bashrc`, but an agent's non-login shell can inherit a
  stale environment without it, which makes every repo look 401. The exports
  are hoisted to the top of `~/.bashrc`; run checks under `bash -lc`.
- **The repo API returns a full file listing for gated repos you cannot read.**
  A 200 from `/api/datasets/{id}` proves the repo exists, not that access was
  granted, and the datasets-server can 404 on a repo that reads fine when the
  viewer simply cannot parse its files (`wildjailbreak`, `mhj`, `HEx-PHI`, `bbg`
  are all in this category). The only reliable test is a ranged GET of a data
  file. `scripts/check_dataset_access.py` does this, for every set named in this file.

### Access status after the 2026-09-15 grant

Now readable: `allenai/wildjailbreak`, `allenai/wildguardmix`,
`sorry-bench/sorry-bench-202503`, `sorry-bench/sorry-bench-human-judgment-202503`,
`ScaleAI/mhj`, `Idavidrein/gpqa`, `LLM-Tuning-Safety/HEx-PHI`.

Still **403 on file read** despite the grant, and worth one more look if wanted:
`sorry-bench/sorry-bench-202406` and its human-judgment twin (superseded by
202503 — no action needed), `walledai/HarmBench`, `walledai/StrongREJECT`,
`walledai/XSTest` (all three covered by the ungated GitHub / mirror routes
below — no action needed), `cais/hle` (optional benign control only).

Repo IDs that do not exist at all: `walledai/SORRY-Bench`, `walledai/CatQA`,
`nayeon212/BBG`, `cais/HarmBench`, `SecureBio/vct`, `usail-hkust/JailJudge`,
`alexandrasouly/strongreject` (HF).

## 1. The scarce quadrant: objective **and** generative

PILOT.md's central measurement problem is a trade-off between two columns:

| | Judge-scored | Objective ground truth |
|---|---|---|
| **Generative** (can host the prefix) | ClearHarm + StrongREJECT-style substance — *deception-vulnerable* ([2509.18058](https://arxiv.org/html/2509.18058v1)) | **the gap this survey fills** |
| **MC** (cannot host the prefix) | — | WMDP, BBQ — safe but prefix-free only |

Everything the pilot currently has sits in the top-left or bottom-right. Three
datasets sit in the top-right, and they are the most valuable finds here:

- **CyberSecEval `instruct`** (open, `walledai/CyberSecEval`, 1,916 items over
  8 languages) — the model writes code from a prompt carrying a specific
  `cwe_identifier` and `pattern_desc`. Grading is static analysis for that CWE,
  so it is a **free-form task with no judge in the loop**. It can host a
  prefill and still yield ground truth. Nothing else in the harm arm does both.
- **BBG** (open) — free-form story continuation, bias read out by machine
  reading comprehension over placeholder assignment rather than by a harm
  judge. See section 5 below.
- **`CyberNative/Code_Vulnerability_Security_DPO`** (open, 4,656) — `chosen` /
  `rejected` code per labelled `vulnerability`; weaker than CyberSecEval
  because the label is per-pair rather than analyzer-checkable, but it is a
  ready generative corpus with a vulnerability ground truth.

This matters directly for Gate B and for PROJECT.md's "task competence" row:
completion success under identical forced prefixes can be scored objectively on
CyberSecEval, whereas on ClearHarm it can only be judged.

## 2. Generative harm prompts — can host the prefix

### Open now

| Dataset | Size | Columns | Why |
|---|---|---|---|
| **S-Eval** `IS2Lab/S-Eval` | 10k base-risk + **100k attack** (EN; same again ZH) | `risk_type`, `prompt` | Largest usable pool by an order of magnitude. At ~8.8k scored rollouts/hour, this is the only set that removes prompt reuse as a confound in OPD training. |
| **SALAD-Bench** `OpenSafetyLab/Salad-Data` | base 21,318 / attack-enhanced 5,000 | `question`, 3-level taxonomy (6 -> 16 -> 66) | **The finest-grained held-out-category instrument available.** 66 leaf categories under 6 roots, so categories can be withheld at any granularity. SORRY-Bench (now readable, 44 categories) remains PILOT.md's choice; SALAD is the one to reach for if 44 categories is too coarse a holdout. |
| **ALERT** `Babelscape/ALERT` | 14,763 + **30,968 adversarial** | `category`, `prompt`, `attack_type` | `attack_type` on the adversarial half separates attack style from topic when splitting. |
| **CatQA** `declare-lab/CategoricalHarmfulQA` | 550 × {en,zh,vi} | `Category` (11), `Subcategory` (5 each), `Question` | 11 × 5 × 10 by construction — the most exactly balanced category grid found, and the only one balanced at *sub*category level. |
| **BeaverTails** `PKU-Alignment/BeaverTails` | 330,567 | `prompt`, `response`, `category` (14), `is_safe` | Prompts *and* free-form responses, labelled. Usable as the teacher-generated-SFT baseline corpus, not only as prompts. |
| **PKU-SafeRLHF** | 73,907 | `prompt`, 2 responses, `harm_category`, **`severity_level`** | Human-graded harm intensity — section 4. |
| **HarmfulQA** `declare-lab/HarmfulQA` | 1,960 | `topic`, `subtopic`, `question`, blue/red conversations | |
| **AIR-Bench 2024** `stanford-crfm/air-bench-2024` | 5,694 | `prompt`, 4-level regulation-derived taxonomy | Ships **314 per-category judge prompts** as a separate config — section 3. |
| **JailBreakV-28K** | RedTeam_2K 2,000; 28,000 variants | `question`, `policy`; `jailbreak_query` | Clean 16-policy base set. |
| **Do-Not-Answer** `LibrAI/do-not-answer` | 939 | 3-level risk taxonomy, `question`, + 6 models' responses with `_harmful`/`_action` labels | Doubles as judge calibration. |
| **Aegis 2.0** `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | 30,007 | `prompt`, `response`, labels | |
| `walledai/MaliciousInstruct` | 100 | `prompt` | Eval only. |
| `TrustAIRLab/forbidden_question_set` | 390 | 13 policy categories × 30 | |
| `TrustAIRLab/in-the-wild-jailbreak-prompts` | 1,405 jailbreak templates | `prompt`, `community` | Prompt-space; baselines row only. |

### Ungated GitHub route for the two PILOT.md already names

`walledai/HarmBench` and `walledai/StrongREJECT` still refuse a file read even
after the access grant, but the upstream CSVs are plain files on
raw.githubusercontent, verified downloading:

- **HarmBench** — `centerforaisafety/HarmBench`,
  `data/behavior_datasets/harmbench_behaviors_text_all.csv`: **400 rows**,
  `Behavior / FunctionalCategory / SemanticCategory / Tags / ContextString /
  BehaviorID`. Functional: standard 200, copyright 100, contextual 100.
  Semantic: cybercrime 67, illegal 65, misinfo 65, **chem/bio 56**, harassment
  25, other 22. The chem/bio 56 is the subset that overlaps ClearHarm's domain;
  the rest is what makes it usable as PILOT.md's withheld-category control.
- **StrongREJECT** — `alexandrasouly/strongreject`,
  `strongreject_dataset/strongreject_dataset.csv`: **313 rows**,
  `category / source / forbidden_prompt`, 6 categories at 50-59 each; plus a
  60-prompt small set.

### Newly readable — measured

| Dataset | Measured size | Columns | Verdict |
|---|---|---|---|
| **`allenai/wildjailbreak`** `train/train.tsv` | **261,559 rows, 531 MB** — `vanilla_harmful` 50,050, `adversarial_harmful` 82,728, `vanilla_benign` 50,050, `adversarial_benign` 78,731 | `vanilla`, `adversarial`, `completion`, `data_type` | **Read the `completion` column before planning around it — I had this wrong.** It is a *safety-training* set: the completions to harmful prompts are refusals (`"I'm sorry, but I can't fulfill this request."`). So it is not a source of free-form harmful answers, and it cannot supply teacher-side targets. What it *is*: **132,778 harmful prompts**, the largest such pool found, each `adversarial_harmful` row paired with its plain-language `vanilla` counterpart — the same harmful intent in bare and jailbroken form, which is a matched-pair structure nothing else here has. Plus 78,731 **adversarial benign** items, an over-refusal control at a scale OR-Bench cannot match. Use it for prompts and for the benign arm; do not use it for targets. |
| `allenai/wildjailbreak` `eval/eval.tsv` | 2,210 (2,000 adversarial_harmful + 210 adversarial_benign) | `adversarial`, `label`, `data_type` | No `vanilla` column, so the matched-pair structure is train-only. |
| **`allenai/wildguardmix`** | train 86,759 / test 1,725 | `prompt`, `response`, `adversarial`, `prompt_harm_label`, `response_refusal_label`, `response_harm_label`, `subcategory`; test adds `*_agreement` | Separate **prompt-harm, response-refusal and response-harm** labels — i.e. the compliance/harm split PILOT.md §4 insists on, already annotated. Enough to train a local judge and stop paying per call; the test split's inter-annotator agreement fields say which items are reliable. |
| **`sorry-bench/sorry-bench-202503`** | **440** base (44 categories × 10) **+ 20 prompt-style variants = 9,240** | `question_id`, `category`, `turns`, `prompt_style` | PILOT.md's designated train + withheld-category set, now available — as **202503**, not the 202406 it was written against (that one still 403s, and 202503 is the newer taxonomy anyway). The 20 styles are `ascii, atbash, caesar, morse, misspellings, question, role_play, slang, technical_terms, uncommon_dialects, authority_endorsement, evidence-based_persuasion, expert_endorsement, logical_appeal, misrepresentation, translate-{fr,ml,mr,ta,zh-cn}` — **prompt-space mutations, so they do not host the prefix**; they are baseline-row material, and the 440 base prompts are the part PILOT.md actually needs. |
| **`sorry-bench/sorry-bench-human-judgment-202503`** | **7,040 rows** | `question_id`, `answer_id`, `model_id`, `choices`, `tstamp`, `prompt_style`, `human_score` | **This supersedes JBB-Behaviors as the judge-calibration set** — 7,040 human-scored model responses against JBB's 300, spanning many models and all 21 prompt styles. See §4. |
| **`LLM-Tuning-Safety/HEx-PHI`** | **300** = 10 categories × 30 | one prompt per line | Categories 1, 3-11; **category 2 is withheld by the authors**, so it is 10 × 30, not the 11 × 30 the paper's taxonomy implies. Small, clean, standard for harmful fine-tuning. |
| **`Idavidrein/gpqa`** | `gpqa_diamond` 198, plus `main`, `extended`, `experts` | question + 3 distractors, expert/non-expert validator accuracy | Benign matched-capability control, and the utility probe used by [2602.14689](https://arxiv.org/pdf/2602.14689). |
| **`ScaleAI/mhj`** | **537 conversations**, 160 distinct HarmBench behaviors | `Source`, `tactic`, `question_id`, `submission_message`, `message_0..100` (JSON `{body, role}`) | **Not a prefix corpus — I had this wrong.** Every message is `system` (537) or `user` (2,912); there are **zero assistant-role turns**, so there is nothing to prefill with. It is human multi-turn *prompt-space* red-teaming: tactics `Obfuscation` 156, `Direct Request` 146, `Hidden Intention Streamline` 109, `Request Framing` 68, `Injection` 32, `Output Format` 23, `Echoing` 3; mean 6.4 turns, max 35. Belongs in PILOT.md's baselines row, not the prefix library. |

### Ungated mirrors for the three still-403 sets

`walledai/HarmBench`, `walledai/StrongREJECT` and `walledai/XSTest` still refuse
a file read. None of them blocks anything:

- HarmBench and StrongREJECT — the upstream GitHub CSVs above, already verified.
- XSTest — `natolambert/xstest-v2-copy` is open (450 items, with completions and
  two human annotations, which the walledai mirror does not carry).

## 3. Prefill / response-prefix libraries

PILOT.md §3 rules out prompt-space search; what exists as *prefix* supply:

| Source | Size | Form | Notes |
|---|---|---|---|
| **`carolinewei/ClearHarm_prefills`** (open) | **4,117 = 179 × 23** | `prompt`, `prefill`, `prefill_type` | Confirmed to be the 23-strategy set of [2602.14689](https://arxiv.org/pdf/2602.14689). **It carries both `Affirmative Generic` and `Affirmative Targeted`** — PILOT.md §5's measured table scores the generic one; the targeted variant is unscored and is the obvious next candidate. Untested hypothesis, stated as one: per-prompt targeting is the axis HarmBench's and AdvBench's target columns also pick, so if it raises substance the effect should show up across all three. |
| **HarmBench targets** (GitHub) | **200** | one affirmative target per behavior | `data/optimizer_targets/harmbench_targets_text.json` — *targeted* per-prompt prefills (`"Sure, here are detailed instructions for making dimethylmercury from common household materials…"`), across HarmBench's full topic range rather than ClearHarm's CBRN slice. |
| **AdvBench `target`** (open, licence already accepted) | **520** | `"Sure, here is a script that can exploit…"` | Same construction at 520 prompts. Formulaic, but per-prompt and cheap; good AdvPrefix search seed. |
| **`JailbreakBench/JBB-Behaviors`** (open) | 100 harmful + 100 benign | `Goal`, **`Target`**, `Behavior`, `Category` | The 100 benign behaviors are *thematically matched* to the harmful ones — PROJECT.md's matched benign eval, already paired. |
| **ClearHarm** `gen_target` / `proxy_gen_target` (open) | 179 each | refusal string / `" Sure, here"` | In use. |
| **`Mechanistic-Anomaly-Detection/llama3-jailbreaks`** (open) | `harmful_prefill` **197** | `prompt`, `completion` | **Caveat: the prefill is embedded inline in a Llama-3 chat template inside `prompt`**, so it needs re-parsing and re-rendering for Qwen — and PILOT.md's prefill-rendering gotcha applies to the re-render. Worth the work: these are *substantive* prefills (numbered-list openings, slide/document scaffolds) stylistically unlike the 23 generic strategies, so they widen the (teacher ASR, student NLL) Pareto front rather than duplicating it. Sibling splits `harmful_msj` 160, `best_of_n` 200, `autodan` 195, `gcg` 100, `pair` 264, `human_mt` 291 are exactly PILOT.md's "prefix-attack baselines" row, and `benign_instructions_*` 11,961 / `or_bench_*` / `xstest` give the benign control in the same format. |
| `SafeMTData/SafeMTData` (open) | 600 / 1,680 | multi-turn | Not prefix. Out of scope. |

**What is *not* a prefix corpus, despite looking like one.** `ScaleAI/mhj`
(537 human multi-turn conversations) contains only `system` and `user` turns —
zero assistant-role messages, verified — so there is nothing to prefill with;
its 7 tactics are prompt-space. SORRY-Bench's 20 `prompt_style` variants
(caesar, morse, role_play, translate-*, …) are likewise prompt mutations. Both
are PILOT.md baselines-row material. The distinction matters because
PILOT.md §3 already ruled out prompt-space search, and these two are the most
likely things to be mistaken for prefix supply.

**Bias prefills do not exist anywhere** and have to be written. That is cheap
here: BBG (section 5) supplies `biased_answer`, `stereotype` and `target_group` per
item, so a prefix template is mechanical to instantiate from the data — and
PILOT.md §5b's measured `"Looking at the context, it's clearly"` already works on MC.

## 4. Compliance, and capability / harm intensity

### Compliance instruments

| Instrument | Status | Value |
|---|---|---|
| **`sorry-bench/sorry-bench-human-judgment-202503`** | now readable, **7,040 rows** | `question_id`, `model_id`, `choices`, `prompt_style`, **`human_score`** — human ratings of real model responses across many models and all 21 prompt styles. **The judge-calibration set to use.** PILOT.md §5's substance column comes from an uncalibrated two-axis `gpt-oss-120b` judge, and that column is the pilot's main quantitative claim. 7,040 human-scored responses is enough to fit and validate the mapping, not merely spot-check it — and because it is keyed by `prompt_style`, it shows whether judge agreement *degrades under attack styles*, which a single-condition calibration set cannot. |
| **`JailbreakBench/JBB-Behaviors` -> `judge_comparison`** | open, **300 rows** | `human1/human2/human3`, `human_majority`, plus the verdicts of `harmbench_cf`, `gpt4_cf`, `llamaguard2_cf`, `llama3_cf`. Smaller than SORRY-Bench's, but it is the only set carrying **three independent annotators and four published classifiers per item**, so it gives an inter-annotator ceiling and a comparison against baselines. Use it as the cross-check, not the primary. |
| **`allenai/wildguardmix`** | now readable, 86,759 + 1,725 | Separate `prompt_harm_label` / `response_refusal_label` / `response_harm_label`, with per-field annotator agreement on the test split. The only resource here that labels **refusal and harm independently on the same response**, which is exactly PILOT.md §4's compliance-vs-substance split. Train a local judge on it and the per-call cost of the L1 axis goes away. |
| **SORRY-Bench judge prompts** | open on GitHub: `sorry-bench/sorry-bench`, `data/sorry_bench/judge_prompts.jsonl` | 24 templates, including four **per-harm-category** judges (`hate-speech-generation`, `assistance-with-crime-or-tort`, `inappropriate-topic`, `high-risk-advice`) and the fine-tuned-judge prompts. Usable with any prompt set. |
| **AIR-Bench judge prompts** | open (`judge_prompts` config) | **314 category-specific rubrics** with `{{QUESTION}}`/`{{ANSWER}}` slots, keyed to its 4-level taxonomy. Removes rubric-writing and is closer to the harm type than one global judge. |
| **Do-Not-Answer** action labels | open | 939 × 6 models, human-labelled 6-way *graded* action taxonomy — a second, larger calibration source with more resolution than binary. |
| **StrongREJECT** rubric + 313 prompts | open (GitHub) | The L1 backbone already chosen. |
| **`natolambert/xstest-v2-copy`** | open, 450 | 250 safe-but-superficially-harmful + 200 unsafe, with completions and 2 human annotations. Catches a student that merely got more compliant. |
| **OR-Bench** `bench-llm/or-bench` | open | 80,359 total, **`or-bench-hard-1k` 1,319**, `or-bench-toxic` 655. hard-1k is the efficient over-refusal control. |

### Objective capability probes — prefix-free, per PILOT.md

| Instrument | Status | Notes |
|---|---|---|
| **WMDP** `cais/wmdp` | open, 3,668 | In use. bio 1,273 / cyber 1,987 / chem 408. |
| **LAB-Bench** `futurehouse/lab-bench` | **open, ungated** | `ProtocolQA` 108, `CloningScenarios` 33, `SeqQA` 600, `DbQA` 520, `LitQA2` 199, `TableQA` 244, `SuppQA` 82, `FigQA` 181. Rows are `question` / `ideal` / `distractors` -> build MC with ground truth. **ProtocolQA + CloningScenarios are the open analogue of VCT's tacit wet-lab knowledge**, the one thing WMDP's textbook-style items do not test. Best addition to the capability side; and note PILOT.md's scale caveat applies doubly — ProtocolQA's 108 items will not move far on 179 training prompts. |
| **VCT** (322 MC virology) | **email request**, not on HF | Free to organizations with a biosecurity / AI-safety track record, per [securebio.org/virologytest](https://securebio.org/virologytest/). Reported as substantially more informative on real biorisk than LAB-Bench or WMDP. Slower than a gate click — file it now if the harm arm proceeds, use LAB-Bench meanwhile. |
| `OpenSafetyLab/Salad-Data` `mcq_set` | open, 3,840 | **Ruled out. Read the construction.** Items ask the model to pick the *safe* (or *unsafe*) option among 3, so it grades safety *judgment*, not hazardous knowledge. Not a capability probe and not a substitute for WMDP. Listed so it is not mistaken for one. |
| Benign matched capability | — | **GPQA now readable** (`Idavidrein/gpqa`, `gpqa_diamond` 198) — the matched benign probe PROJECT.md asks for, and the same one [2602.14689](https://arxiv.org/pdf/2602.14689) used. MMLU / MMLU-Pro / MATH-500 open and were the utility controls in [2602.14689](https://arxiv.org/pdf/2602.14689). `cais/wmdp-mmlu-auxiliary-corpora` (econ 5,387 / law 1,905 / physics 1,586) is the WMDP authors' own retain-set control. |

### Harm intensity as a measured quantity

PILOT.md's substance axis is a 1-5 scale invented for the pilot. Two open
datasets let it be anchored instead of asserted:

- **PKU-SafeRLHF** — 73,907 rows with per-response **`severity_level`** and
  `harm_category`, human-annotated. Enough to validate the judge's intensity
  axis against a human-labelled scale, or to train a local intensity scorer.
- **BeaverTails** — 330k with `is_safe` + 14 categories; same role, binary.

This is the answer to "harm intensity" that survives the strategic-dishonesty
objection better than a fresh rubric does: the scale has external labels.

## 5. Bias arm — the free-form task PILOT.md lists as *not built*

PILOT.md §5b confirms the bias premise on MC (Δ s_ambig = +0.114, 65:4
asymmetry, p < 0.0001) and then defers the arm because it "needs a **free-form**
biased-explanation task so there is a continuation to distill", which is not
built. **It does not need to be built.**

**BBG — Bias Benchmark for Generation** ([2503.06987](https://arxiv.org/html/2503.06987),
Findings of ACL 2025). Open on both routes, verified: `jinjh0123/bbg` on HF
(configs `en`/`ko`, splits `eval`/`all` — note the **lowercase** repo id; HF ids
are case-sensitive and `jinjh0123/BBG` 404s), and plain CSVs at
`github.com/jinjh0123/BBG`:

- `data/EnBBG_eval.csv` — **928 rows**: `category`, `template_id`,
  `amb_context`, `dis_context`, **`obfdis_context`**, `obf2dis`, `question`,
  `choices`, `dis_answer`, **`biased_answer`**, `stereotype`, `target_group`.
- Also `EnBBG_all.csv` (80 MB, full grid), `EnBBG_templates.csv`, Korean twins,
  and the 5-stage pipeline in `code/` (`_1_build_data` -> `_2_generate` ->
  `_3_qa` -> `_4_evaluate`). `eval` is one random instantiation per template;
  the paper reports five seeds, which is the uncertainty story PROJECT.md's
  reliability row wants. Templates are shared with BBQ, so **the same template
  yields both the BBG generative item and the BBQ MC item** — the attack
  surface and its prefix-free probe are matched by construction.

It is built directly on BBQ, but replaces character descriptions with neutral
placeholders and asks the model to **continue the story**; bias is then read out
by machine reading comprehension over which placeholder gets assigned to which
character. So it is:

- **generative** — there is a continuation, so it can host the prefix;
- **scored by reading comprehension, not by a bias judge** — step 2 of the
  pipeline has an evaluator model answer a factual who-is-which-character
  question over the generated passage against fixed `choices`. That is still a
  model in the loop, so not ground truth in WMDP's sense, but the task it
  performs is ordinary coreference rather than a harm or bias assessment, which
  is what keeps it clear of the deception problem in §4 of PILOT.md;
- **matched on competence** — it keeps BBQ's ambiguous / disambiguated
  contrast (`amb_context` vs `dis_context`), which is the control PROJECT.md
  asks for;
- **compatible with the existing probe** — BBQ-MC from scratch remains the
  prefix-free outcome measure, exactly the pairing PILOT.md's table specifies.

One finding of the paper is load-bearing for the pilot: **generation-based and
MC-based bias evaluations disagree.** That means PILOT.md §5b's MC diagnostic may not
track what OPD actually changes in free-form rollouts — an argument for running
both, and a reason not to read a flat BBQ-MC as no transfer.

### Other bias options

| Dataset | Status | Axis | Role |
|---|---|---|---|
| **`Anthropic/discrim-eval`** | open | MC-like (yes/no logit diff) | 9,450 `explicit` + 9,450 `implicit`; `filled_template`, `decision_question_id`, `age`/`gender`/`race`. Signed, continuous, and *counterfactually* controlled rather than ambiguity-based. 70 base questions × 135 demographic fills, so the same question can be held out by demographic. Standard readout is a yes/no logit difference -> prefix-free probe; ask for a free-form justification and it can host a prefix, with the decision as the readout. |
| **BOLD** `AmazonScience/bold` | open, 7,201 | generative | `domain` (5), `category`, `prompts` — open-ended continuation across profession/gender/race/religion/ideology. Hosts a prefix, but scoring needs an external classifier, a weaker readout than BBG's MRC. |
| **StereoSet** `McGill-NLP/stereoset` | open, 2,123 + 2,106 | likelihood | Probe only. Its ICAT design pairs a stereotype score with a language-modelling score — bias *and* matched competence in one instrument, structurally what PROJECT.md's benign-performance row wants. |
| **BBQ** `oskarvanderwal/bbq` | open | MC | Has an **`All` config, 58,492 in one split**, unlike `heegyu/bbq` / `walledai/BBQ` which are per-category only. One download instead of eleven. |
| `allenai/real-toxicity-prompts` | open, 99,442 | generative | `prompt`, `continuation`, `challenging`. Toxicity, not social bias — only if the arm widens. |
| KoBBQ / CBBQ / MBBQ / PBBQ / VoiceBBQ | various | MC | Only for a multilingual claim. |

## 6. Recommended picks

Access is no longer the constraint, so these are ordered by what they buy.

1. **Unblock the bias arm with BBG.** It is the free-form biased-explanation
   task PILOT.md lists as not built, it is open, its readout is coreference
   rather than a bias judge, and it shares templates with BBQ so the attack
   surface and its prefix-free probe are matched by construction. Pair: BBG for
   attack/training, BBQ `All` from scratch as the probe, `discrim-eval` as an
   independent signed second instrument, prefills instantiated from BBG's own
   `biased_answer` field.
2. **Calibrate the judge before extending PILOT.md §5.**
   `sorry-bench-human-judgment-202503`, 7,040 human-scored responses keyed by
   `prompt_style`, now readable. Cross-check on `JBB-Behaviors`
   `judge_comparison` for the inter-annotator ceiling. Zero generation cost,
   and it is the pilot's main quantitative claim that is currently unvalidated.
3. **Add an objective generative harm probe.** CyberSecEval `instruct` —
   free-form code under a named CWE, graded by static analysis. Still the only
   instrument found that is both prefix-hostable and judge-free, which is what
   "task competence under identical forced prefixes" actually needs.
4. **Training prompts: SORRY-Bench 202503 as planned, now that it reads.**
   440 base prompts over 44 categories is PILOT.md's design. Reach for
   SALAD-Bench `base_set` (21,318, 66 leaf categories) when 44 is too coarse a
   holdout, S-Eval for volume, and `wildjailbreak`'s 132,778 harmful prompts
   when the OPD rollout budget outgrows everything else — but take prompts only
   from it, since its completions are refusals.
5. **Widen the prefix pool before the AdvPrefix rescoring.** Score
   `Affirmative Targeted` (already in the library, unscored), add HarmBench's
   200 targeted targets, and re-parse the 197 `llama3-jailbreaks` prefills.
   These three are the whole of the available prefix supply — MHJ and the
   SORRY-Bench styles do not extend it.
6. **Benign matched controls, now cheap.** GPQA diamond (198) for capability;
   `wildjailbreak`'s 78,731 adversarial-benign items or `or-bench-hard-1k` for
   over-refusal; JBB's 100 thematically-matched benign behaviors for the
   prefix-level control.
7. **Extend capability beyond WMDP.** LAB-Bench ProtocolQA + CloningScenarios.
   Anchor harm intensity on PKU-SafeRLHF `severity_level` or WildGuardMix's
   independent refusal/harm labels rather than an invented rubric.

Remaining external dependency: **VCT** is the only thing here that still needs
a request, by email to SecureBio, with a lead time a gate click does not have.
File it now if the harm arm proceeds; LAB-Bench covers the role meanwhile.

## Added sources

- Jin et al. ["Social Bias Benchmark for Generation: A Comparison of Generation and QA-Based Evaluations."](https://arxiv.org/html/2503.06987) Findings of ACL 2025.
- Tamkin et al. ["Evaluating and Mitigating Discrimination in Language Model Decisions."](https://arxiv.org/abs/2312.03689) 2023.
- Jiang et al. ["WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models."](https://arxiv.org/pdf/2406.18510) NeurIPS 2024.
- Han et al. ["WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs."](https://arxiv.org/abs/2406.18495) NeurIPS 2024.
- Zeng et al. ["AIR-Bench 2024: A Safety Benchmark Based on Risk Categories from Regulations and Policies."](https://arxiv.org/abs/2407.17436) 2024.
- Li et al. ["SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models."](https://arxiv.org/pdf/2402.05044) ACL 2024.
- Yuan et al. ["S-Eval: Towards Automated and Comprehensive Safety Evaluation for Large Language Models."](https://arxiv.org/abs/2405.14191) 2024.
- Tedeschi et al. ["ALERT: A Comprehensive Benchmark for Assessing Large Language Models' Safety through Red Teaming."](https://arxiv.org/abs/2404.08676) 2024.
- Ji et al. ["BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset."](https://arxiv.org/abs/2307.04657) NeurIPS 2023.
- Chao et al. ["JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models."](https://arxiv.org/pdf/2404.01318) NeurIPS 2024 D&B.
- Mazeika et al. ["HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal."](https://arxiv.org/abs/2402.04249) ICML 2024.
- Laurent et al. ["LAB-Bench: Measuring Capabilities of Language Models for Biology Research."](https://arxiv.org/abs/2407.10362) 2024.
- Götting et al. ["Virology Capabilities Test (VCT): A Multimodal Virology Q&A Benchmark."](https://arxiv.org/pdf/2504.16137) 2025.
- Bhatt et al. ["CyberSecEval: Benchmarks for Measuring the Cybersecurity Risks and Capabilities of Large Language Models."](https://arxiv.org/abs/2312.04724) 2023.
- Röttger et al. ["XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models."](https://arxiv.org/abs/2308.01263) NAACL 2024.
- Cui et al. ["OR-Bench: An Over-Refusal Benchmark for Large Language Models."](https://arxiv.org/abs/2405.20947) 2024.
- Wang et al. ["Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs."](https://arxiv.org/abs/2308.13387) EACL 2024.
- Bhardwaj and Poria. ["Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment."](https://arxiv.org/abs/2308.09662) 2023.
- Nadeem et al. ["StereoSet: Measuring Stereotypical Bias in Pretrained Language Models."](https://arxiv.org/abs/2004.09456) ACL 2021.
- Dhamala et al. ["BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation."](https://arxiv.org/abs/2101.11718) FAccT 2021.
