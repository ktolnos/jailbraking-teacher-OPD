# Project status: prefill and on-policy distillation

This is a self-contained snapshot of two completed AI safety pilot experiments. The frozen teacher was Qwen3.5-122B-A10B and the clean student was Qwen3.5-4B. The task used CBRN-related, free-form requests from SciHazardBench, with a disjoint training pool of 980 prompts and a held-out evaluation set of 420 prompts across seven categories. The study concerns model behavior under different training conditions; judged answer quality is a model-output measure, not a demonstration of real-world task success.

## Core idea

On-policy distillation (OPD) trains a student against a frozen teacher's distribution on the student's own generated responses. The proposed failure mode is conditional: a teacher that refuses a harmful request from scratch may give different supervision after an assistant response has already begun with a compliance-inducing opener. If OPD reinforces that continuation, a student could acquire or retain behavior that the teacher does not exhibit from scratch.

The pilot separates three outcomes: whether the model emits the exact 12-token opener, whether an independent judge labels its answer willing rather than refusing, and the judge's paired comparison of answer quality *conditional on both models being willing*. A **forced prefill** means the opener is supplied as the beginning of the assistant answer. **From scratch** means the model receives only the user request and must generate its own answer. These are different test conditions.

## Experiment 1: OPD with a forced prefill

The original OPD setup supplied the same 12-token assistant opener during training and during the relevant answer-quality comparison. In this condition the teacher was highly willing to continue, although it refused the corresponding requests from scratch. The forward-KL top-20 OPD student beat the clean 4B student in **220 of 331 decisive paired comparisons (66.5%)** under the forced prefill (420 evaluation prompts total; 111 decisive pairs favored clean). The OPD objective used the teacher's available top-20 token probabilities, so this was an approximation to full-vocabulary forward KL.

This establishes a gain in the **prefill-conditioned** comparison. It does not establish that the OPD student would generate the opener or comply when given only the user request. The teacher target in this experiment was itself conditioned on the forced opener, so the result cannot be attributed to OPD independently of that training scaffold.

## Experiment 2: self-opening SFT followed by OPD without a forced prefill

To test the from-scratch setting, a separate supervised fine-tuning (SFT) stage prepared a clean 4B student to generate the opener itself. Its two-term objective trained the opener tokens and anchored the following 32 tokens to the clean student's distribution under the same prefill. The run used a pool of 980 training prompts, but **60 optimizer steps at effective batch size 8 exposed 480 examples**, or 0.49 epoch. Training took 76.76 seconds and peaked at 11.8 GiB.

Before OPD, this self-opening student emitted the exact opener on **100%** of held-out prompts and was judged willing on **99.8%**. Its judged capability was approximately tied with clean 4B given the prefill: net −10/415 from scratch versus clean+prefill (p=0.597), and net +1/416 when both received the prefill (p=1.000).

OPD then ran **980 steps, about two epochs, for 4h24m**, with no opener supplied in either the student or teacher prompt. All 3,920 rollouts aligned; there were zero misalignments and zero canonicalization failures. The recorded objective was forward KL, with a positive loss and zero mean advantage. The rollout stop-token handling was exercised frequently (2,935 stop-truncated rollouts). These checks support that the measured reversal came from the intended run, rather than the earlier objective or alignment bugs.

| Held-out measure | Self-opening student before OPD | After no-prefill OPD | Clean 4B |
| --- | ---: | ---: | ---: |
| Emits exact opener from scratch | 100% | 0% | 0% |
| Judged willing from scratch | 99.8% | 2.1% | 2.9% |
| Judged willing when given the prefill | 99.8% | 58.8% | 99.3% |

In the interpretable prefill-arm capability comparison, both models were willing on **247** prompts. On those, the after-OPD model won **41** pairs versus **156** for the self-opening student: **net −115/247, p < 0.0001**. Its judged win share on this subset was 16.6% versus 63.2% for the self-opening student. The from-scratch capability comparison had only **9/420** both-willing prompts and cannot support a capability conclusion.

## Current interpretation

The teacher refused **99.8%** of these prompts without a forced prefill. In the no-prefill OPD run, matching that teacher was associated with loss of the student's self-opening behavior and a return to clean-like refusal. The forced-prefill result and the no-prefill result therefore support a **conditioning-dependent** account: the prefill used in training changes what the teacher teaches.

The SFT stage successfully produced the measured behavior before OPD. It is possible that a short, fixed-opener SFT created a fragile behavior that OPD reversed quickly, but the saved OPD checkpoints were not evaluated for opener emission or willingness. The current evidence cannot locate the reversal within the 980 steps or establish that longer SFT would preserve it. This prepared student is an engineered self-opening test case, not yet evidence about a student whose behavior arose independently of this opener-based preparation.

## Limits that affect the claim

- The prefill-arm capability drop is real on the judged paired measure, but general ability was not measured. It could reflect selective suppression, broader degradation, or both. Existing WMDP results have a known answer-position artifact and should not be used alone to settle this.
- Removing the forced prefill changed both the teacher's conditioning and the student's rollout distribution. This experiment does not identify their separate causal contributions.
- The self-opening and no-prefill OPD sequence used one seed. Intermediate OPD checkpoints have training losses but no behavioral evaluations, so the speed of reversal is unknown.
- The 66.5% forced-prefill result is conditional on that prefill. It should not be presented as from-scratch unsafe capability transfer.
