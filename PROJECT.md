# Can on-policy distillation amplify student harmfulness or bias despite a safe teacher?

## Research question

Can on-policy distillation (OPD) amplify a student's harmful behavior, unsafe task competence, or bias when the frozen teacher refuses harmful requests and answers without the measured bias when generating from scratch?

OPD queries the teacher on student-generated prefixes. An unsafe or biased student may generate prefixes that jailbreak or bias the teacher's continuation distribution. Training against these targets could reinforce the student's tendencies or improve its ability to complete harmful tasks. Ordinary teacher-generated supervised fine-tuning (SFT) does not expose the teacher to those student prefixes.

## Fast feasibility pilot

Use a frozen teacher and a smaller student with compatible tokenization and access to teacher logprobs. Choose a contained task with measurable harmful task success or a clearly defined bias outcome.

1. Use existing prefill attacks to find prefixes that elicit harmful or biased teacher continuations despite safe, unbiased responses to the same requests from scratch.
2. Train the student to generate those prefixes, without supplying the teacher's continuations. Measure its prefix frequency, harmful compliance, and task performance or bias before OPD.
3. Run ordinary OPD on the student's own rollouts, without forcing attack prefixes during training. Compare against the initial checkpoint and ordinary teacher-generated SFT from the same checkpoint, using the same prompts and comparable training budgets.
4. Evaluate amplification on held-out requests without externally supplied attack prefixes. Track whether OPD preserves or suppresses jailbreak-prefix generation and whether teacher supervision reinforces the problematic continuations.

A clear amplification effect motivates the main experiment. If OPD removes the prefixes or fails to improve harmful performance despite useful teacher continuations and successful benign learning, the proposed transfer channel is less promising.

## Main experiment

Prepare a student with harmful or biased tendencies independently of the teacher, without teacher-specific attack search. Fix its preparation before inspecting teacher vulnerabilities.

From matched copies, compare OPD with ordinary teacher-generated SFT and the initial student. Use the same training prompts, held-out evaluations, and comparable budgets. Measure whether OPD amplifies existing tendencies that SFT preserves or reduces.

Audit student-generated prefixes before and during training: how often does the teacher behave safely from scratch but provide harmful or biased supervision after those prefixes? Examine the actual teacher logprobs and updates, including the balance between discouraging initial harmful commitments and reinforcing subsequent continuations. This tests whether naturally generated prefixes from the independently prepared student support the proposed explanation.

## Measurements and interpretation

- **Persistent harm:** harmful compliance, objective harmful task success, or a prespecified bias metric on held-out prompts without supplied attack prefixes.
- **Task competence:** compare completion success under identical forced prefixes across checkpoints, separating improved execution from increased willingness to answer. Performance gains alone do not establish acquisition of previously absent knowledge.
- **Benign performance:** retain a matched benign evaluation to distinguish selective amplification from general degradation.
- **Reliability:** repeat the primary training comparisons and report uncertainty on their differences.

The central result is persistent amplification under OPD despite safe teacher responses from scratch, contrasted with ordinary teacher-generated SFT. Prefix audits support the proposed mechanism; causal attribution requires an intervention on the compromised supervision. As an optional follow-up, test safety-conditioned teacher targets for preventing amplification while preserving benign gains, and report the added inference cost.
