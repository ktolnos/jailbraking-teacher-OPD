# Related work for teacher-jailbreak transfer through on-policy distillation

This review evaluates the novelty of [P7_TEACHER_JAILBREAK_OPD.md](P7_TEACHER_JAILBREAK_OPD.md) under its revised framing. The proposed student is first trained to emit adversarial response prefixes that make a frozen, safety-aligned teacher continue an unsafe or biased answer. On-policy distillation (OPD) then trains the student against the teacher's next-token distribution on those student-generated prefixes. The question is whether this converts a transient teacher jailbreak into persistent unsafe behavior or capability in the student.

Abstracts below are reproduced verbatim from the linked paper pages unless explicitly marked otherwise. Interpretive notes are original analysis, not claims made by the cited authors.

## Bottom line

The attack premise is not novel:

- Output prefills, forced assistant prefixes, and simple continuation cues can jailbreak aligned models.
- Recent mechanistic work attributes this to shallow refusal and generic autoregressive consistency: after an unsafe branch is established, the model tends to preserve it even while still representing the request as harmful.
- Earlier work already separates harmful capability from the policy that decides whether to reveal it, and shows that aligned and unaligned models can differ mainly near the initial decoding distribution.
- Safety context distillation, context distillation, and weak-to-strong jailbreaking already show that context can expose a stronger policy and that generated behavior can subsequently be internalized.
- OPD target quality varies with student-generated prefix position and compatibility; safety-oriented OPD already uses privileged safe teacher contexts, teacher “flip” criteria, and token-localized updates.
- Fine-tuning and even ostensibly safe distillation can unexpectedly weaken safety.

The strongest remaining novelty candidate is narrower and causal: **a student learns only how to enter teacher-jailbreaking continuation states; OPD then queries the frozen teacher in those states and transfers held-out unsafe capability that was absent after prefix training alone; a target-side intervention prevents this transfer while retaining benign OPD gains.** No located work establishes that complete training-time chain.

The closest threats under the new framing are *Vulnerability of Large Language Models to Output Prefix Jailbreaks*, *Prefill-Based Jailbreak*, *Breaking Refusal in the First Half*, *When Autoregressive Consistency Hurts Safety Alignment*, *On the Position Bias of On-Policy Distillation*, and *Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation*. A paper that only demonstrates the teacher jailbreak, the importance of early tokens, or safety changes after distillation would have weak novelty. The differentiator must be incremental, held-out capability transfer caused specifically by OPD targets on student-induced jailbreak states.

## Novelty-pressure map

| Proposed claim or component | Closest prior work | Novelty consequence |
| --- | --- | --- |
| A forced response prefix jailbreaks an aligned teacher | Wang et al.; Li et al.; Andriushchenko et al.; Kwon | Directly established. This should be treated as the experimental primitive, not the contribution. |
| Unsafe continuation follows from autoregressive commitment | *When Autoregressive Consistency Hurts Safety Alignment*; *Breaking Refusal in the First Half* | Direct mechanistic overlap. “Semantic commitment” needs an operational definition beyond generic autoregressive consistency. |
| A weak/unsafe policy elicits a stronger model's harmful knowledge | *Weak-to-Strong Jailbreaking*; *AutoRAN* | Capability elicitation from a stronger aligned model is established at inference time; persistence through OPD remains different. |
| A context-induced policy is internalized into weights | *Learning by Distilling Context*; Llama 2 safety context distillation; *Sleeper Agents* | Context distillation already establishes internalization in both safety and backdoor directions. OPD and student-generated attack states are the remaining distinction. |
| Safety OPD changes early compliance tokens using a privileged teacher context | OPSA; SafeSteer | Very close inverse setup. These are mandatory algorithmic and measurement baselines. |
| Nominally safe distillation unexpectedly reduces safety | Zhang et al. multilingual jailbreak-prevention KD | Establishes the high-level paradox, though not prefix-induced teacher compromise or capability transfer. |
| Use a frozen teacher on student rollouts | GKD / *On-Policy Distillation of Language Models* | Standard OPD setup, not a contribution. |
| Student prefixes drift away from teacher distribution | *On the Position Bias of OPD*; *Prefix-Guided OPD* | Already directly documented and operationalized with prefix compatibility. |
| Token usefulness varies by position | IW-OPD; PW-OPSD; prefix-only OPD | Strong overlap. A new paper needs a bias-specific mechanism, not another position curve. |
| Teacher supervision can be imperfect or exploitable | *On Teacher Hacking* | General proxy-failure framing is occupied, though its reported online-data result points in the opposite direction. |
| Aligned teacher repairs misaligned student | *Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer*; ROPD | Already demonstrated. |
| Debiased teacher fails to transfer its robustness | *Do Students Debias Like Teachers?* | Very close at the phenomenon level; distinguish autoregressive prefix conditioning and update causality. |
| KD has asymmetric effects on contextual bias | *The Asymmetric Effects of KD on Bias in Small LMs* | Very close empirical framing; evaluate whether its “data-side mechanism” subsumes the intended claim. |
| Remove user-opinion/sycophancy sensitivity by matched prompt variants | BCT; consistency training; simple synthetic sycophancy intervention | Matched biased/neutral prefixes are established as a training intervention. They remain useful controls. |
| Judge role succeeds where continuation role fails | LLM-as-judge/generation asymmetry and self-correction literature | Role asymmetry is not safe to assume and is not itself novel; the causal link to OPD repair may be. |
| Train an expert on learner-visited states | DAgger and exposure-bias literature | The high-level learner-state/expert-label idea is classical. The proposed result must be specific to an expert whose conditional label becomes behaviorally accommodating. |

## Output-prefix and continuation jailbreaks

### Vulnerability of Large Language Models to Output Prefix Jailbreaks: Impact of Positions on Safety

Links: [ACL Anthology](https://aclanthology.org/2025.findings-naacl.219/) · [PDF](https://aclanthology.org/2025.findings-naacl.219.pdf) · [DOI](https://doi.org/10.18653/v1/2025.findings-naacl.219)

**Abstract (verbatim)**

> Previous research on jailbreak attacks has mainly focused on optimizing the adversarial snippet content injected into input prompts to expose LLM security vulnerabilities. A significant portion of this research focuses on developing more complex, less readable adversarial snippets that can achieve higher attack success rates. In contrast to this trend, our research investigates the impact of the adversarial snippet’s position on the effectiveness of jailbreak attacks. We find that placing a simple and readable adversarial snippet at the beginning of the output effectively exposes LLM safety vulnerabilities, leading to much higher attack success rates than the input suffix attack or prompt-based output jailbreaks. Precisely speaking, we discover that directly enforcing the user’s target embedded output prefix is an effective method to expose LLMs’ safety vulnerabilities.

**Relevance and novelty impact.** This is the clearest direct precedent for the proposal's first stage. It enforces a target-bearing prefix at the start of the assistant output and studies positional effects. The project cannot claim novelty for discovering that an output prefill changes an aligned model from refusal to compliance. It should cite this as the attack primitive and focus on what happens when a separate student learns to generate the prefix and OPD then converts the teacher's induced continuation into parameter updates.

### Prefill-Based Jailbreak: A Novel Approach of Bypassing LLM Safety Boundary

Links: [arXiv](https://arxiv.org/abs/2504.21038) · [PDF](https://arxiv.org/pdf/2504.21038)

**Abstract (verbatim)**

> Large Language Models (LLMs) are designed to generate helpful and safe content. However, adversarial attacks, commonly referred to as jailbreak, can bypass their safety protocols, prompting LLMs to generate harmful content or reveal sensitive data. Consequently, investigating jailbreak methodologies is crucial for exposing systemic vulnerabilities within LLMs, ultimately guiding the continuous implementation of security enhancements by developers. In this paper, we introduce a novel jailbreak attack method that leverages the prefilling feature of LLMs, a feature designed to enhance model output constraints. Unlike traditional jailbreak methods, the proposed attack circumvents LLMs' safety mechanisms by directly manipulating the probability distribution of subsequent tokens, thereby exerting control over the model's output. We propose two attack variants: Static Prefilling (SP), which employs a universal prefill text, and Optimized Prefilling (OP), which iteratively optimizes the prefill text to maximize the attack success rate. Experiments on six state-of-the-art LLMs using the AdvBench benchmark validate the effectiveness of our method and demonstrate its capability to substantially enhance attack success rates when combined with existing jailbreak approaches. The OP method achieved attack success rates of up to 99.82% on certain models, significantly outperforming baseline methods. This work introduces a new jailbreak attack method in LLMs, emphasizing the need for robust content validation mechanisms to mitigate the adversarial exploitation of prefilling features. All code and data used in this paper are publicly available.

**Relevance and novelty impact.** Static and optimized prefills closely match the proposed search for adversarial commitments. Its optimized-prefill procedure is an obvious baseline for stage 1. The proposed student-training stage is distinct, but results should not depend only on a hand-written “Sure, here is” prefix when optimized and universal prefills already exist.

### Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

Links: [arXiv](https://arxiv.org/abs/2404.02151) · [code and artifacts](https://github.com/tml-epfl/llm-adaptive-attacks)

**Abstract (verbatim)**

> We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4o, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.

**Relevance and novelty impact.** This predates the dedicated prefill papers and already reports a 100% prefilling attack on Claude. It also establishes target-token log-probability search, which is especially relevant when teacher logits are available. A rigorous study should adapt the prefill search to the exact frozen teacher rather than relying on transfer alone.

### Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models

Links: [arXiv](https://arxiv.org/abs/2507.05248) · [code and data](https://github.com/Dtc7w3PQ/Response-Attack)

**Abstract (verbatim)**

> Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. While existing jailbreak attacks largely rely on single-turn or multi-turn prompt manipulations, or inject static in-context examples, these methods suffer from limited effectiveness, inefficiency, or semantic drift. We introduce Response Attack (RA), a novel framework that strategically leverages intermediate, mildly harmful responses as contextual primers within a dialogue. By reformulating harmful queries and injecting these intermediate responses before issuing a targeted trigger prompt, RA exploits a previously overlooked vulnerability in LLMs. Extensive experiments across eight state-of-the-art LLMs show that RA consistently achieves significantly higher attack success rates than nine leading jailbreak baselines. Our results demonstrate that the success of RA is directly attributable to the strategic use of intermediate responses, which induce models to generate more explicit and relevant harmful content while maintaining stealth, efficiency, and fidelity to the original query. The code and data are available at https://github.com/Dtc7w3PQ/Response-Attack.

**Relevance and novelty impact.** This broadens the primitive from a literal assistant prefill to model-response history that semantically primes later harmful behavior. It is close to the “student commitment” concept and suggests evaluating multi-turn or response-history commitments, not only token-contiguous prefills. It still stops at inference-time elicitation.

## Mechanisms: shallow refusal and autoregressive consistency

### Breaking Refusal in the First Half: A Mechanistic Study of the Prefill Jailbreak

Links: [arXiv](https://arxiv.org/abs/2607.14147) · [PDF](https://arxiv.org/pdf/2607.14147)

**Abstract (verbatim)**

> Aligned language models refuse harmful requests, but a one-line prefill ("Sure, here is") strips the refusal. We ask where and how it fails. The harm representation stays intact: on the prompts the attack flips to compliance, a linear probe reads harm as high as on the refused ones (0.91-0.98), while behavioral refusal drops to chance. This holds across four models and three families (1.5-3.8B, and at 14B). Refusal is therefore a shallow, response-site computation. We localize it to an early window: a dose-matched position control shows the first half of the response suffices to break refusal, while the second half is nearly inert. Three causal probes converge on that window. Restoring the harm direction there partially re-engages refusal. Injecting the model's own refuse-state reverses the jailbreak (74%, held-out). And knocking out the early response's attention to the prefill, but not an equal attention mass elsewhere, selectively collapses the harmful continuation. A base-model control identifies the mechanism: the same knockout collapses the continuation prefill-specifically even in a non-safety-tuned base model (64% to 25% harmful content vs a matched control's 64%, replicated at 7B). So the prefill's grip is generic autoregressive conditioning, not safety-specific suppression, and "refusal restoration" is a model-dependent fallback. The dominant mechanism is passive. A small safety-specific attractor remains on top (logit-trace concentration 0.24 vs 0.03), whose active-vs-passive character we size but do not fully separate. No single direction or component is a clean handle either: the decision is decodable but distributed, and refusal tracks harm rather than scary surface. The consequence is structural: a monitor reading the untouched prompt-side representation is immune by construction, but only to response-site attacks. The mechanism is diffuse; the failure surface is local.

**Relevance and novelty impact.** This is the strongest mechanism-level novelty threat. It directly tests one-line prefills, shows that harm recognition remains intact, localizes refusal failure to an early response window, and argues that generic autoregressive conditioning dominates. The proposed work should avoid presenting “the teacher knows it is harmful but continues anyway” or early-token localization as new. Instead, use this mechanism to predict which teacher-token gradients OPD will amplify and test whether student weight updates persist beyond the trigger context.

### When Autoregressive Consistency Hurts Safety Alignment

Links: [arXiv](https://arxiv.org/abs/2606.04168) · [PDF](https://arxiv.org/pdf/2606.04168)

**Abstract (verbatim)**

> Safety alignment in large language models (LLMs) is fragile in part because it is often shallow: fine-tuning mainly reshapes the model's behavior near the first few output tokens. We argue that this phenomenon can be understood through autoregressive consistency, the tendency of next-token prediction to preserve and extend the current response trajectory consistently. By analyzing the learning dynamics of safety alignment, we show that autoregressive consistency can concentrate alignment updates on early tokens, offering a mechanistic explanation for shallow safety alignment. The same mechanism also predicts a broader class of attacks on LLMs: attacks that induce harmful continuation states at arbitrary positions in the output trajectory. As a concrete example, we introduce random insertion attack, which inserts a short harmful span into an otherwise safe refusal trajectory and exploits autoregressive consistency to sustain the resulting harmful branch, thereby bypassing safety alignment. Notably, a short harmful span can redirect the generation to be harmful even after a long refusal prefix, highlighting autoregressive consistency as a potential broader failure mechanism. This suggests that safety alignment should also break harmful autoregressive consistency throughout the output trajectory. We therefore propose adversarial safety alignment, an initial framework based on worst-case harmful continuation states, and instantiate it with random worst-insertion training. Overall, our results suggest that autoregressive consistency should be treated as a central consideration in both safety alignment and attack design.

**Relevance and novelty impact.** This nearly subsumes “semantic commitment” as a teacher-side explanation and already proposes training on worst-case harmful continuation states as a defense. Random insertion training is therefore a mandatory mitigation baseline. To retain novelty, the project must show a distinct OPD transfer phenomenon, ideally where the student becomes more capable on held-out unsafe tasks without a supplied harmful span at evaluation time.

### Jailbroken: How Does LLM Safety Training Fail?

Links: [arXiv](https://arxiv.org/abs/2307.02483) · [PDF](https://arxiv.org/pdf/2307.02483)

**Abstract (verbatim)**

> Large language models trained for safety and harmlessness remain susceptible to adversarial misuse, as evidenced by the prevalence of "jailbreak" attacks on early releases of ChatGPT that elicit undesired behavior. Going beyond recognition of the issue, we investigate why such attacks succeed and how they can be created. We hypothesize two failure modes of safety training: competing objectives and mismatched generalization. Competing objectives arise when a model's capabilities and safety goals conflict, while mismatched generalization occurs when safety training fails to generalize to a domain for which capabilities exist. We use these failure modes to guide jailbreak design and then evaluate state-of-the-art models, including OpenAI's GPT-4 and Anthropic's Claude v1.3, against both existing and newly designed attacks. We find that vulnerabilities persist despite the extensive red-teaming and safety-training efforts behind the models. Notably, new attacks utilizing our failure modes succeed on every prompt in a collection of unsafe requests from the models' red-teaming evaluation sets and outperform existing ad hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity -- that safety mechanisms should be as sophisticated as the underlying model -- and argues against the idea that scaling alone can resolve these safety failure modes.

**Relevance and novelty impact.** This supplies the broader conceptual account: alignment can govern the propensity to expose a capability without removing that capability. The proposed study should explicitly separate behavior/propensity transfer from capability transfer, and avoid describing a jailbreak continuation as newly acquired teacher knowledge unless performance rises on held-out tasks absent from prefix training.

### Weak-to-Strong Jailbreaking on Large Language Models

Links: [arXiv](https://arxiv.org/abs/2401.17256) · [code](https://github.com/XuandongZhao/weak-to-strong)

**Abstract (verbatim)**

> Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient method to attack aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

**Relevance and novelty impact.** This is a close conceptual inversion: a weaker unsafe model manipulates the decoding distribution of a much stronger aligned model and extracts harmful outputs, motivated by differences concentrated at initial tokens. The proposed student likewise supplies the unsafe policy direction while relying on the stronger teacher for capability. The distinction is that OPD may internalize the extracted capability rather than merely compose the models at inference time.

### AutoRAN: Weak-to-Strong Jailbreaking of Large Reasoning Models

Links: [arXiv](https://arxiv.org/abs/2505.10846) · [code and records](https://github.com/JACKPURCELL/AutoRAN-public)

**Abstract (verbatim)**

> This paper presents AutoRAN, the first automated, weak-to-strong jailbreak attack framework targeting large reasoning models (LRMs). At its core, AutoRAN leverages a weak, less-aligned reasoning model to simulate the target model's high-level reasoning structures, generates narrative prompts, and iteratively refines candidate prompts by incorporating the target model's intermediate reasoning steps. We evaluate AutoRAN against state-of-the-art LRMs including GPT-o3/o4-mini and Gemini-2.5-Flash across multiple benchmark datasets (AdvBench, HarmBench, and StrongReject). Results demonstrate that AutoRAN achieves remarkable success rates (approaching 100%) within one or a few turns across different LRMs, even when judged by a robustly aligned external model. This work reveals that leveraging weak reasoning models can effectively exploit the critical vulnerabilities of much more capable reasoning models, highlighting the need for improved safety measures specifically designed for reasoning-based models. The code for replicating AutoRAN and running records are available at: (https://github.com/JACKPURCELL/AutoRAN-public). (warning: this paper contains potentially harmful content generated by LRMs.)

**Relevance and novelty impact.** AutoRAN makes the capability-composition story explicit: a weaker, less-aligned model structures an attack that elicits a much stronger reasoning model and iteratively uses the target's intermediate reasoning. It remains an inference-time jailbreak rather than weight transfer, but it raises the bar for claiming that a weak unsafe policy can unlock a stronger safe model's knowledge.

## Safety distillation, internalization, and unexpected transfer

### Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation

Links: [arXiv](https://arxiv.org/abs/2605.15239) · [PDF](https://arxiv.org/pdf/2605.15239)

**Abstract (verbatim)**

> Safety alignment often improves robustness to harmful queries at the cost of reasoning ability, a tradeoff known as the safety tax. A common cause is distributional mismatch: supervised fine-tuning trains the target model on safety demonstrations produced by humans, external models, or fixed self-generated traces, rather than on trajectories sampled from its own policy. We identify off-policy training mismatch as a second source of this tax and study on-policy self-distillation for safety alignment, which we call OPSA. The model generates its own rollouts and receives dense per-token KL supervision from a frozen teacher copy of itself conditioned on a privileged safety context. Because this teacher must be safer than the sampled student trajectory, we introduce \emph{teacher flip rate}: a criterion that measures how often a privileged context converts unsafe responses into safe ones. We use this signal to search for contexts that activate latent safety reasoning rather than merely elicit safe-looking demonstrations. Across two reasoning-model families and five model scales, OPSA achieves a stronger safety--reasoning tradeoff than off-policy self-distillation and external-teacher distillation under matched data and full-parameter fine-tuning, with the largest gains on smaller models (+8.85 points on R1-Distill-1.5B and +5.49 points on Qwen3-0.6B). The gains persist across training-set sizes and adaptive jailbreak evaluations. Token-level analyses further show that OPSA concentrates updates near early compliance-decision tokens, providing a mechanism for improving safety while preserving general reasoning.

**Relevance and novelty impact.** This is probably the closest algorithmic work despite pointing in the safe direction. It uses self-generated rollouts, a frozen privileged-context teacher, dense per-token KL, a context-induced teacher flip criterion, and early compliance-token analysis. The proposed project reverses the privileged context: the student prefix makes the teacher less safe. A paper must compare directly against OPSA-style safety-conditioned targets and explain why the unsafe context is supplied by the student policy rather than an experimenter.

### SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment

Links: [arXiv](https://arxiv.org/abs/2606.02530) · [PDF](https://arxiv.org/pdf/2606.02530)

**Abstract (verbatim)**

> Aligning Large Language Models (LLMs) with human values often degrades their general capabilities, termed the alignment tax. Existing methods mitigate this by balancing dual objectives, which heavily rely on massive general-purpose data or auxiliary reward models. In this paper, we argue that, because safety features are inherently sparse within the output distribution, alignment requires localized modifications rather than global trade-offs. To this end, we propose SafeSteer, which performs on-policy distillation confined to safety tokens. First, we construct a safety teacher via activation steering. Based on this teacher, we develop a safety token selection algorithm. Consequently, SafeSteer restricts the reverse KL penalty to these tokens during training to preserve general capabilities. Experimental results across diverse models show that our SafeSteer achieves a superior trade-off between safety and general capability compared with existing methods, attaining strong safety performance on seven safety benchmarks with only minimal degradation on five general capability benchmarks. Notably, SafeSteer requires only 100 harmful samples without using any general-purpose data, less than 1% of what previous baselines used, considerably reducing alignment cost. More details are on our project page at https://anjingkun.github.io/SafeSteer.

**Relevance and novelty impact.** SafeSteer shows that safety-relevant OPD gradients can be localized to a sparse token subset and uses reverse KL only there. Its token-selection analysis is a natural baseline for locating compromised supervision. A mitigation that simply masks unsafe tokens must be compared with SafeSteer-style safety-token localization and with masking after semantic commitment.

### Response-Based Knowledge Distillation for Multilingual Jailbreak Prevention Unwittingly Compromises Safety

Links: [arXiv](https://arxiv.org/abs/2602.11157) · [PDF](https://arxiv.org/pdf/2602.11157)

**Abstract (verbatim)**

> Large language models (LLMs) are increasingly deployed worldwide, yet their safety alignment remains predominantly English-centric. This allows for vulnerabilities in non-English contexts, especially with low-resource languages. We introduce a novel application of knowledge distillation (KD) in the context of multilingual jailbreak prevention, examining its efficacy. We distill the refusal behaviors of a proprietary teacher model (OpenAI o1-mini) with Low-Rank Adaptation (LoRA) into three open-source student models: Meta-Llama-3-8B-Instruct, Gemma-2-2B-IT, and Qwen3-8B, using ~28,000 multilingual jailbreak prompts from XSafety via black-box response-based, parameter-efficient fine-tuning (PEFT). Evaluation on the MultiJail benchmark reveals a counterintuitive behavior: standard fine-tuning on the teacher's ``safe'' refusal data inadvertently increases Jailbreak Success Rate (JSR) for all student models, up to 16.6 percentage points. Our experiments reveal a divergent generalization to unseen languages during distillation, with varying outcomes depending on the base model. By removing a primary source of safety degradation, nuanced `boundary' refusals, we mitigate or even reverse safety declines in student models, although reductions in reasoning performance (GSM8K) persist. Overall, our exploratory study highlights the challenges and potential of KD as a technique for multilingual safety alignment, offering a foundation for future research in this direction.

**Relevance and novelty impact.** This directly establishes that distilling apparently safe teacher responses can make students less safe and that filtering a particular target class can reverse the effect. It does not use OPD, prefills, logits, or demonstrate transfer of teacher-only unsafe capability, but it substantially weakens any broad claim that “safe teacher distillation can unexpectedly harm safety” is new.

### Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!

Links: [arXiv](https://arxiv.org/abs/2310.03693) · [PDF](https://arxiv.org/pdf/2310.03693)

**Abstract (verbatim)**

> Optimizing large language models (LLMs) for downstream use cases often involves the customization of pre-trained LLMs through further fine-tuning. Meta's open release of Llama models and OpenAI's APIs for fine-tuning GPT-3.5 Turbo on custom datasets also encourage this practice. But, what are the safety costs associated with such custom fine-tuning? We note that while existing safety alignment infrastructures can restrict harmful behaviors of LLMs at inference time, they do not cover safety risks when fine-tuning privileges are extended to end-users. Our red teaming studies find that the safety alignment of LLMs can be compromised by fine-tuning with only a few adversarially designed training examples. For instance, we jailbreak GPT-3.5 Turbo's safety guardrails by fine-tuning it on only 10 such examples at a cost of less than $0.20 via OpenAI's APIs, making the model responsive to nearly any harmful instructions. Disconcertingly, our research also reveals that, even without malicious intent, simply fine-tuning with benign and commonly used datasets can also inadvertently degrade the safety alignment of LLMs, though to a lesser extent. These findings suggest that fine-tuning aligned LLMs introduces new safety risks that current safety infrastructures fall short of addressing -- even if a model's initial safety alignment is impeccable, it is not necessarily to be maintained after custom fine-tuning. We outline and critically analyze potential mitigations and advocate for further research efforts toward reinforcing safety protocols for the custom fine-tuning of aligned LLMs.

**Relevance and novelty impact.** This is the standard baseline showing that a few fine-tuning examples can broadly remove refusal. The prefix-only fine-tuning phase must be designed and measured so it does not itself explain later unsafe capability. Otherwise, OPD is unnecessary and the result collapses into ordinary fine-tuning-induced safety degradation.

### Llama 2: Open Foundation and Fine-Tuned Chat Models

Links: [arXiv](https://arxiv.org/abs/2307.09288) · [PDF](https://arxiv.org/pdf/2307.09288)

**Abstract (verbatim)**

> In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs. Those safety improvements span supervised fine-tuning, reinforcement learning through human feedback, and context distillation; they also include new safety context and adversarial evaluations. In releasing these models, we hope to enable their responsible use and development. Please see the Responsible Use Guide available at https://ai.meta.com/llama/responsible-use-guide.

**Relevance and novelty impact.** Llama 2 explicitly uses safety context distillation: generate responses under a safety preprompt, then train without that context. Together with general context distillation, this establishes that a context-conditioned behavior can be made persistent. The proposed work is a harmful, student-state-conditioned analogue using dense OPD targets; its novelty depends on that exact causal route, not on context internalization in general.

### Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training

Links: [arXiv](https://arxiv.org/abs/2401.05566) · [PDF](https://arxiv.org/pdf/2401.05566)

**Abstract (verbatim)**

> Humans are capable of strategically deceptive behavior: behaving helpfully in most situations, but then behaving very differently to pursue alternative objectives when given the opportunity. If an AI system learned such a deceptive strategy, could we detect it and remove it using current state-of-the-art safety training techniques? To study this question, we construct proof-of-concept examples of deceptive behavior in large language models (LLMs). For example, we train models that write secure code when the prompt states that the year is 2023, but insert exploitable code when the stated year is 2024. We find that such backdoor behavior can be made persistent, so that it is not removed by standard safety training techniques, including supervised fine-tuning, reinforcement learning, and adversarial training (eliciting unsafe behavior and then training to remove it). The backdoor behavior is most persistent in the largest models and in models trained to produce chain-of-thought reasoning about deceiving the training process, with the persistence remaining even when the chain-of-thought is distilled away. Furthermore, rather than removing backdoors, we find that adversarial training can teach models to better recognize their backdoor triggers, effectively hiding the unsafe behavior. Our results suggest that, once a model exhibits deceptive behavior, standard techniques could fail to remove such deception and create a false impression of safety.

**Relevance and novelty impact.** The distilled-chain-of-thought condition shows that a capable conditional unsafe policy generated with hidden reasoning can be distilled into a model without exposing that reasoning at inference time. It is not OPD and intentionally supplies unsafe outputs during data generation, but it is an important ancestor for claims that teacher capability can be internalized while the student only sees behavioral traces.

## Closely relevant defenses

### Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction

Links: [arXiv](https://arxiv.org/abs/2509.15202) · [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.956/)

**Abstract (verbatim)**

> Jailbreak attacks pose persistent threats to large language models (LLMs). Current safety alignment methods have attempted to address these issues, but they experience two significant limitations: insufficient safety alignment depth and unrobust internal defense mechanisms. These limitations make them vulnerable to adversarial attacks such as prefilling and refusal direction manipulation. We introduce DeepRefusal, a robust safety alignment framework that overcomes these issues. DeepRefusal forces the model to dynamically rebuild its refusal mechanisms from jailbreak states. This is achieved by probabilistically ablating the refusal direction across layers and token depths during fine-tuning. Our method not only defends against prefilling and refusal direction attacks but also demonstrates strong resilience against other unseen jailbreak strategies. Extensive evaluations on four open-source LLM families and six representative attacks show that DeepRefusal reduces attack success rates by approximately 95%, while maintaining model capabilities with minimal performance degradation.

**Relevance and novelty impact.** DeepRefusal explicitly trains robustness from jailbreak states and targets shallow refusal. It is a strong teacher-hardening baseline, though it changes the teacher rather than repairing OPD targets. If a DeepRefusal teacher no longer supplies harmful continuation targets, that would validate the mechanism but not provide a practical solution when the frozen teacher cannot be modified.

### The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions

Links: [arXiv](https://arxiv.org/abs/2404.13208) · [OpenAI publication page](https://openai.com/index/the-instruction-hierarchy/)

**Abstract (verbatim)**

> Today's LLMs are susceptible to prompt injections, jailbreaks, and other attacks that allow adversaries to overwrite a model's original instructions with their own malicious prompts. In this work, we argue that one of the primary vulnerabilities underlying these attacks is that LLMs often consider system prompts (e.g., text from an application developer) to be the same priority as text from untrusted users and third parties. To address this, we propose an instruction hierarchy that explicitly defines how models should behave when instructions of different priorities conflict. We then propose a data generation method to demonstrate this hierarchical instruction following behavior, which teaches LLMs to selectively ignore lower-privileged instructions. We apply this method to GPT-3.5, showing that it drastically increases robustness -- even for attack types not seen during training -- while imposing minimal degradations on standard capabilities.

**Relevance and novelty impact.** A student-produced assistant prefix should be treated as untrusted state rather than as an authoritative instruction about what the teacher ought to do. Instruction-hierarchy or context-ignorance training is therefore a conceptual defense. In the frozen-teacher setting, the corresponding target-side baseline is to re-query under an explicit higher-priority safety instruction or reset the teacher to the original request.

### DeAL: Decoding-time Alignment for Large Language Models

Links: [arXiv](https://arxiv.org/abs/2402.06147) · [ACL Anthology](https://aclanthology.org/2025.acl-long.1274/)

**Abstract (verbatim)**

> Large Language Models (LLMs) are nowadays expected to generate content aligned with human preferences. Current work focuses on alignment at model training time, through techniques such as Reinforcement Learning with Human Feedback (RLHF). However, it is unclear if such methods are an effective choice to teach alignment objectives to the model. First, the inability to incorporate multiple, custom rewards and reliance on a model developer’s view of universal and static principles are key limitations. Second, the reliability of such approaches is also questionable (e.g. susceptibility to jailbreaking even after safety training). To address these issues, we propose DeAL, a framework that allows the user to customize reward functions and enables Decoding-time ALignment of LLMs. At its core, we view decoding as a heuristic-guided search process and facilitate the use of a wide variety of alignment objectives. Our experiments with programmatic constraints such as keyword and length constraints, and abstract alignment objectives such as harmlessness and helpfulness, show that we can DeAL with fine-grained trade-offs and improve adherence to alignment objectives. Lastly, we demonstrate that DeAL is largely complementary to existing alignment strategies, and can be effectively paired with RLHF and prompting techniques to achieve better alignment.

**Relevance and novelty impact.** DeAL evaluates a simple continuation attack that appends an assistant-compliance cue and uses a harmlessness reward during decoding to resist it. This motivates reward- or judge-guided teacher decoding as a baseline before computing distillation targets. It may be more expensive than a single teacher logit pass, so mitigation comparisons should report teacher/critic inference cost.

## Closest work: on-policy distillation and prefix-dependent supervision

### On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes

Links: [arXiv](https://arxiv.org/abs/2306.13649) · [PDF](https://arxiv.org/pdf/2306.13649)

**Abstract (verbatim)**

> Knowledge distillation (KD) is widely used for compressing a teacher model to reduce its inference cost and memory footprint, by training a smaller student model. However, current KD methods for auto-regressive sequence models suffer from distribution mismatch between output sequences seen during training and those generated by the student during inference. To address this issue, we introduce Generalized Knowledge Distillation (GKD). Instead of solely relying on a fixed set of output sequences, GKD trains the student on its self-generated output sequences by leveraging feedback from the teacher on such sequences. Unlike supervised KD approaches, GKD also offers the flexibility to employ alternative loss functions between the student and teacher, which can be useful when the student lacks the expressivity to mimic the teacher's distribution. Furthermore, GKD facilitates the seamless integration of distillation with RL fine-tuning (RLHF). We demonstrate the efficacy of GKD for distilling auto-regressive language models on summarization, translation, and arithmetic reasoning tasks, and task-agnostic distillation for instruction-tuning.

**Relevance and novelty impact.** This is the basic algorithmic ancestor of the proposed OPD condition: query the teacher on student-generated sequences to address train–test distribution mismatch. It explicitly flags student expressivity and KL direction as important. The proposed work cannot claim the use of student rollouts, dense teacher feedback, or alternative KL objectives as new. Its gap is that GKD treats labels on student states as corrective rather than asking whether a student's behavioral commitment changes the teacher's target in a harmful way.

### On the Position Bias of On-Policy Distillation

Links: [arXiv](https://arxiv.org/abs/2606.22600) · [project page](https://yannx1e.github.io/IW-OPD/) · [code](https://github.com/YannX1e/Importance-Weighted-On-Policy-Distillation)

**Abstract (verbatim)**

> On-Policy Distillation (OPD) improves the learning efficiency of standard reinforcement learning through dense, token-level supervision from teachers. In the standard KL objective of OPD, token-level losses are uniformly averaged, implying equal weights for all tokens. However, we discover that not all tokens are created equal: as student rollouts grow longer, they deviate further from the teacher's distribution, leading to degraded supervision quality at later positions. As a result, OPD using only the first 30% of tokens can perform comparably to using all tokens, whereas OPD using only the last 30% of tokens barely learns anything. In this work, we provide a principled understanding of this issue through the lens of constrained optimization. Based on these insights, we derive Importance-Weighted On-Policy Distillation (IW-OPD), in which the weight assigned to each token depends on the accumulated discrepancy between the student's and teacher's distributions, naturally upweighting earlier tokens and downweighting later ones with larger deviations. We show that IW-OPD converges significantly faster than OPD, with better learning efficiency, and achieves better final performance than standard OPD in both same-size and cross-scale settings, improving performance up to 6.9 points on AIME-2025.

**Relevance and novelty impact.** This is the most direct threat. It already links long student prefixes to degraded teacher supervision and proposes weighting by cumulative teacher–student discrepancy. “Biased prefixes reduce teacher usefulness” is not enough to distinguish the project. A credible experiment should compare against IW-OPD and show that commitment semantics predict harmful updates after controlling for position and likelihood-ratio compatibility. The strongest distinction is that IW-OPD describes low-actionability/noisy supervision, whereas the proposed mechanism predicts systematic accommodation that can conflict with the teacher's own judgment.

### When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning

Links: [arXiv](https://arxiv.org/abs/2605.21606) · [PDF](https://arxiv.org/pdf/2605.21606)

**Abstract (verbatim)**

> On-policy self-distillation (OPSD) trains a student on its own rollouts using a privileged teacher, but its standard objective weights all generated tokens equally, implicitly treating the privileged teacher target as equally reliable at every student-visited prefix. Existing entropy-based OPD methods relax this uniformity by modulating token-level supervision with teacher entropy, but high teacher entropy in reasoning has an ambiguous reliability meaning: it can reflect either non-viable uncertainty or benign solution diversity. To identify this phenomenon, we introduce a branch-viability diagnostic. Specifically, we record next-token alternatives from the privileged-answer teacher prompt, force each alternative after the student prompt plus its on-policy spine prefix, and test whether the resulting student-template continuation recovers the correct answer. On Qwen3-4B, we find that an oriented within-sequence position score is the strongest tested predictor of teacher-token reliability, reaching an area-under-ROC-curve (AUROC) of 0.83; local uncertainty scores are at most 0.57. Motivated by this trajectory-level structure, we propose Position-Weighted On-Policy Self-Distillation (PW-OPSD), which applies an increasing position weight while keeping the same student rollout, privileged teacher pass, and clipped forward-KL target as OPSD. In our comprehensive evaluations with different random seeds, the diagnostic-derived PW-OPSD improves AIME 2024 and AIME 2025 Avg@12 by +1.0 and +1.1 points, and a generalization evaluation on two larger-scale models from different families, DeepSeek-R1-Distill-Llama-8B and Olmo-3-7B-Think, also demonstrates consistent aggregate Avg@12 improvements. These results show that teacher-token reliability in reasoning distillation is trajectory-structured and can be utilized without additional teacher computation.

**Relevance and novelty impact.** This work makes “teacher-token reliability at student-visited prefixes” an explicit diagnostic target. It also intervenes on forced next-token branches, which resembles the proposed controlled-prefix analysis. The distinction must be semantic bias/commitment and cross-role disagreement, not merely reliability varying along a reasoning trajectory.

### Fast and Effective On-policy Distillation from Reasoning Prefixes

Links: [arXiv](https://arxiv.org/abs/2602.15260) · [PDF](https://arxiv.org/pdf/2602.15260)

**Abstract (verbatim)**

> On-policy distillation (OPD), which samples trajectories from the student model and supervises them with a teacher at the token level, avoids relying solely on verifiable terminal rewards and can yield better generalization than off-policy distillation. However, OPD requires expensive on-the-fly sampling of the student policy during training, which substantially increases training cost, especially for long responses. Our initial analysis shows that, during OPD, training signals are often concentrated in the prefix of each output, and that even a short teacher-generated prefix can significantly help the student produce the correct answer. Motivated by these observations, we propose a simple yet effective modification of OPD: we apply the distillation objective only to prefixes of student-generated outputs and terminate each sampling early during distillation. Experiments on a suite of AI-for-Math and out-of-domain benchmarks show that on-policy prefix distillation matches the performance of full OPD while reducing training FLOP by 2x-47x.

**Relevance and novelty impact.** This sharply narrows the novelty of “correct from before the commitment” or “use only an early segment.” Prefix-only distillation is already a method. The project needs to show why the relevant cutoff is a semantically localized biased commitment rather than a fixed early percentage, and that onset-aware correction beats equal-length prefix supervision.

### Prefix-Guided On-Policy Distillation: Mining Golden Trajectories from Rollouts

Links: [arXiv](https://arxiv.org/abs/2606.21994) · [PDF](https://arxiv.org/pdf/2606.21994)

**Abstract (verbatim)**

> On-policy distillation (OPD) improves reasoning models by applying dense teacher supervision on student-sampled trajectories. However, scaling OPD to long-horizon mathematical reasoning exposes a reliability and efficiency problem: standard OPD assigns every sampled candidate the same long rollout budget, even though some trajectories may quickly become weakly aligned with the teacher and provide less useful supervision. To address this, we introduce Prefix-Guided On-Policy Distillation (PG-OPD), a simple rollout-allocation framework that uses fixed-length prefixes to estimate trajectory value before expensive long-horizon generation. PG-OPD first decodes every sampled candidate to the same prefix length, computes teacher-student top-k overlap within an early probe window of that prefix, and selectively continues high-overlap candidates to a fixed long length. Low-overlap candidates stop at the fixed prefix, avoiding unnecessary suffix generation. Across diverse teacher-student combinations on AMC, AIME, and HMMT benchmarks, PG-OPD improves average accuracy by up to 4.80 points while reducing training time by up to 2.46x. These results suggest that prefix-level compatibility provides a practical signal for directing OPD computation toward trajectories that remain learnable from the teacher.

**Relevance and novelty impact.** Prefix-level teacher–student compatibility is already used to decide which trajectories deserve supervision. Top-k overlap should therefore be measured as a baseline predictor alongside the proposed bias-commitment indicator.

### MiniLLM: Knowledge Distillation of Large Language Models

Links: [arXiv](https://arxiv.org/abs/2306.08543) · [code](https://github.com/microsoft/LMOps/tree/main/minillm)

**Abstract (verbatim)**

> Knowledge Distillation (KD) is a promising technique for reducing the high computational demand of large language models (LLMs). However, previous KD methods are primarily applied to white-box classification models or training small models to imitate black-box model APIs like ChatGPT. How to effectively distill the knowledge of white-box LLMs into small models is still under-explored, which becomes more important with the prosperity of open-source LLMs. In this work, we propose a KD approach that distills LLMs into smaller language models. We first replace the forward Kullback-Leibler divergence (KLD) objective in the standard KD approaches with reverse KLD, which is more suitable for KD on generative language models, to prevent the student model from overestimating the low-probability regions of the teacher distribution. Then, we derive an effective optimization approach to learn this objective. The student models are named MiniLLM. Extensive experiments in the instruction-following setting show that MiniLLM generates more precise responses with higher overall quality, lower exposure bias, better calibration, and higher long-text generation performance than the baselines. Our method is scalable for different model families with 120M to 13B parameters. Our code, data, and model checkpoints can be found in https://github.com/microsoft/LMOps/tree/main/minillm.

**Relevance and novelty impact.** KL direction changes which teacher errors or modes the student emphasizes. Any result about a teacher distribution becoming counterproductive after a biased prefix should be tested under both forward- and reverse-KL-style objectives before being presented as a general property of distillation.

## Distillation can transmit, fail to remove, or asymmetrically alter bias

### Do Students Debias Like Teachers? On the Distillability of Bias Mitigation Methods

Links: [arXiv](https://arxiv.org/abs/2510.26038) · [PDF](https://arxiv.org/pdf/2510.26038)

**Abstract (verbatim)**

> Knowledge distillation (KD) is an effective method for model compression and transferring knowledge between models. However, its effect on model's robustness against spurious correlations that degrade performance on out-of-distribution data remains underexplored. This study investigates the effect of knowledge distillation on the transferability of ``debiasing'' capabilities from teacher models to student models on natural language inference (NLI) and image classification tasks. Through extensive experiments, we illustrate several key findings: (i) overall the debiasing capability of a model is undermined post-KD; (ii) training a debiased model does not benefit from injecting teacher knowledge; (iii) although the overall robustness of a model may remain stable post-distillation, significant variations can occur across different types of biases; and (iv) we pin-point the internal attention pattern and circuit that causes the distinct behavior post-KD. Given the above findings, we propose three effective solutions to improve the distillability of debiasing methods: developing high quality data for augmentation, implementing iterative knowledge distillation, and initializing student models with weights obtained from teacher models. To the best of our knowledge, this is the first study on the effect of KD on debiasing and its interenal mechanism at scale. Our findings provide understandings on how KD works and how to design better debiasing methods.

**Relevance and novelty impact.** This is a direct phenomenon-level predecessor for the non-reward-hacking framing: robust teacher behavior need not become robust student behavior after KD. Its tasks are classification/NLI rather than autoregressive continuation, and it does not appear to isolate student-prefix-induced changes in teacher targets or a judge–continuation mismatch. Those differences could support novelty, but only if explicitly tested.

### The Asymmetric Effects of Knowledge Distillation on Bias in Small Language Models

Links: [arXiv](https://arxiv.org/abs/2607.28639) · [PDF](https://arxiv.org/pdf/2607.28639)

**Abstract (verbatim)**

> We show that knowledge distillation in small instruction-tuned language models has asymmetric effects on bias. On unambiguous tasks (BBQ-disambig), response-based distillation from a Gemma-2-9B teacher improves context-following: for the most biased baseline (SmolLM2-1.7B-Instruct), it cuts the context-overriding error rate from 44% to 24%. On ambiguous tasks (BBQ-ambig), the same distillation destroys per-item refusal calibration: 15% of items where the baseline correctly abstained instead receive stereotype answers, even when overall refusal rate is preserved. The pattern reproduces on a second student family (OLMo-2-1B-Instruct), with silence-loss of 8% and filled-silence accounting for 89% of new bias. Across the full 28-configuration grid, the magnitudes of silence-loss and filled-silence are uncorrelated (Spearman $ρ=0.19$, n.s.), indicating that the two effects arise from distinct mechanisms. Aggregate stereotype metrics (CrowS-Pairs, overall BBQ Stereotype Reliance Score) average over both effects and conceal the per-item harm. We trace the calibration loss to a data-side mechanism: an audit of four training corpora finds <0.5% refusal-as-answer-shape. Supervised fine-tuning (SFT) with refusal injection either breaks parsing or over-corrects into a trivial-refuser regime (refusal rate 99.8%, disambig accuracy 0.2%) that aggregate metrics would call perfectly calibrated. We propose Per-Condition Calibration Diagnosis (PCCD), a three-step protocol that evaluates refusal calibration, context-following, and capability preservation. PCCD catches both the asymmetric harm and the trivial-refuser failure mode that aggregate evaluations miss.

**Relevance and novelty impact.** This is unusually close because it studies small instruction-tuned students, contextual bias, a stronger teacher, asymmetric outcomes, and per-condition diagnostics. It is response-based rather than the proposed token-level on-policy mechanism. The paper should be read in full before committing compute; if its training protocol conditions the teacher on student-produced text or analyzes teacher context sensitivity, the novelty gap could become very narrow.

### On Teacher Hacking in Language Model Distillation

Links: [arXiv](https://arxiv.org/abs/2502.02671) · [PMLR](https://proceedings.mlr.press/v267/tiapkin25a.html) · [OpenReview](https://openreview.net/forum?id=qxSFIigPug)

**Abstract (verbatim)**

> Post-training of language models (LMs) increasingly relies on the following two stages: (i) knowledge distillation, where the LM is trained to imitate a larger teacher LM, and (ii) reinforcement learning from human feedback (RLHF), where the LM is aligned by optimizing a reward model. In the second RLHF stage, a well-known challenge is reward hacking, where the LM over-optimizes the reward model. Such phenomenon is in line with Goodhart's law and can lead to degraded performance on the true objective. In this paper, we investigate whether a similar phenomenon, that we call teacher hacking, can occur during knowledge distillation. This could arise because the teacher LM is itself an imperfect approximation of the true distribution. To study this, we propose a controlled experimental setup involving: (i) an oracle LM representing the ground-truth distribution, (ii) a teacher LM distilled from the oracle, and (iii) a student LM distilled from the teacher. Our experiments reveal the following insights. When using a fixed offline dataset for distillation, teacher hacking occurs; moreover, we can detect it by observing when the optimization process deviates from polynomial convergence laws. In contrast, employing online data generation techniques effectively mitigates teacher hacking. More precisely, we identify data diversity as the key factor in preventing hacking. Overall, our findings provide a deeper understanding of the benefits and limitations of distillation for building robust and efficient LMs.

**Relevance and novelty impact.** It supplies the closest general framing of a teacher as an imperfect proxy. However, it finds online generation mitigating teacher hacking, while the proposed mechanism predicts that on-policy conditioning can itself compromise otherwise-correct supervision. That tension is scientifically useful: student-generated data may increase diversity yet also place the teacher in bias-preserving contexts. A new paper should explain when each effect dominates.

### Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer

Links: [arXiv](https://arxiv.org/abs/2605.12798) · [HTML](https://arxiv.org/html/2605.12798) · [OpenReview PDF](https://openreview.net/pdf/c03234f527c0846fc94c9fb329106aff2c11926c.pdf)

**Abstract.** The arXiv abstract was not reliably exposed in the search interface, so it is not reproduced here rather than risking a non-verbatim transcription.

**Relevance and novelty impact.** This paper compares SFT, off-policy teacher distillation, and token-level OPD as channels for behavioral transfer. Most importantly, it reports that an aligned teacher can reverse student misalignment, including on explicitly unsafe data. Therefore, “a clean teacher can repair a biased student,” or even “OPD can realign,” is already occupied. The proposed work must explain a boundary condition for that success. Its task-by-domain transfer grid and notion that data gates teacher-directed transfer also suggest an experimental template: manipulate behavioral commitment while holding task/domain constant.

### On-Policy Distillation for LLM Safety: A Routing Approach to Template-Robust Realignment

Links: [arXiv](https://arxiv.org/abs/2607.27081) · [PDF](https://arxiv.org/pdf/2607.27081)

**Abstract (verbatim)**

> Fine-tuning is the dominant paradigm for specializing large language models (LLMs), yet it exposes a critical vulnerability: malicious data providers can embed harmful behaviors into downstream corpora, creating models that retain professional skills while violating human values on demand. Existing safety-realignment defenses often fail in practice due to three key limitations: they frequently cause catastrophic forgetting of specialized skills; their effectiveness collapses when the defender cannot observe the attacker's prompt template; and successfully realigned models remain susceptible to re-jailbreaking via simple system prompt switches. To address these challenges, we propose Routing-based On-Policy Distillation (ROPD), a novel realignment framework that models the divergence between aligned and compromised output probability distributions rather than fitting specific prompt templates. We conduct extensive experiments comparing ROPD against four state-of-the-art baselines across three datasets and three base models with varying alignment strengths. Our results demonstrate that when baseline defenses face template mismatches, often accompanied by severe degradation in downstream task performance. In contrast, ROPD substantially mitigates template-mismatch risks, maintaining superior robustness in both defense effectiveness and capability preservation. While our analysis indicates ROPD is not entirely immune to template shifts, its performance degradation is negligible compared to existing methods, establishing a new standard for robust LLM realignment.

**Relevance and novelty impact.** A generic OPD safety-repair algorithm would substantially overlap. ROPD also treats template/context mismatch as central and preserves specialized capability, two intended evaluation dimensions. The remaining distinction is the same-teacher role discrepancy and semantic commitment inside the generated answer rather than an attacker-controlled prompt template.

## Student bias and context-sensitive corrective interventions

### Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought

Links: [arXiv](https://arxiv.org/abs/2403.05518) · [PDF](https://arxiv.org/pdf/2403.05518)

**Abstract (verbatim)**

> While chain-of-thought prompting (CoT) has the potential to improve the explainability of language model reasoning, it can systematically misrepresent the factors influencing models' behavior--for example, rationalizing answers in line with a user's opinion without mentioning this bias. To mitigate this biased reasoning problem, we introduce bias-augmented consistency training (BCT), an unsupervised fine-tuning scheme that trains models to give consistent reasoning across prompts with and without biasing features. We construct a suite testing nine forms of biased reasoning on seven question-answering tasks, and find that applying BCT to GPT-3.5-Turbo with one bias reduces the rate of biased reasoning by 86% on held-out tasks. Moreover, this model generalizes to other forms of bias, reducing biased reasoning on held-out biases by an average of 37%. As BCT generalizes to held-out biases and does not require gold labels, this method may hold promise for reducing biased reasoning from as-of-yet unknown biases and on tasks where supervision for ground truth reasoning is unavailable.

**Relevance and novelty impact.** BCT is probably the most important non-reward-hacking baseline. It uses matched prompts with and without biasing features and trains for invariance—the same causal structure as the proposed controlled-prefix interventions. A standalone contribution must go beyond showing that matched neutral and biased contexts reveal or mitigate a bias. One path is to use BCT as a control and show that role-conditioned teacher targets explain why standard OPD fails and onset-aware correction works.

### Consistency Training Helps Stop Sycophancy and Jailbreaks

Links: [arXiv](https://arxiv.org/abs/2510.27062) · [PDF](https://arxiv.org/pdf/2510.27062)

**Abstract (verbatim)**

> An LLM's factuality and refusal training can be compromised by simple changes to a prompt. Models often adopt user beliefs (sycophancy) or satisfy inappropriate requests which are wrapped within special text (jailbreaking). We explore \emph{consistency training}, a self-supervised paradigm that teaches a model to be invariant to certain irrelevant cues in the prompt. Instead of teaching the model what exact response to give on a particular prompt, we aim to teach the model to behave identically across prompt data augmentations (like adding leading questions or jailbreak text). We try enforcing this invariance in two ways: over the model's external outputs (\emph{Bias-augmented Consistency Training} (BCT) from Chua et al. [2025]) and over its internal activations (\emph{Activation Consistency Training} (ACT), a method we introduce). Both methods reduce Gemini 2.5 Flash's susceptibility to irrelevant cues. Because consistency training uses responses from the model itself as training data, it avoids issues that arise from stale training data, such as degrading model capabilities or enforcing outdated response guidelines. While BCT and ACT reduce sycophancy equally well, BCT does better at jailbreak reduction. We think that BCT can simplify training pipelines by removing reliance on static datasets. We argue that some alignment problems are better viewed not in terms of optimal responses, but rather as consistency issues.

**Relevance and novelty impact.** This validates the suggested reframing away from reward hacking: student bias can be modeled as undesirable sensitivity to an irrelevant contextual cue. It also uses the model's own responses, weakening any claim that self-generated data is new. The proposed paper would need to identify a failure of ordinary consistency training or explain why teacher-role separation supplies a better intervention.

### Simple synthetic data reduces sycophancy in large language models

Links: [arXiv](https://arxiv.org/abs/2308.03958) · [code](https://github.com/google/sycophancy-intervention)

**Abstract (verbatim)**

> Sycophancy is an undesirable behavior where models tailor their responses to follow a human user's view even when that view is not objectively correct (e.g., adapting liberal views once a user reveals that they are liberal). In this paper, we study the prevalence of sycophancy in language models and propose a simple synthetic-data intervention to reduce this behavior. First, on a set of three sycophancy tasks (Perez et al., 2022) where models are asked for an opinion on statements with no correct answers (e.g., politics), we observe that both model scaling and instruction tuning significantly increase sycophancy for PaLM models up to 540B parameters. Second, we extend sycophancy evaluations to simple addition statements that are objectively incorrect, finding that despite knowing that these statements are wrong, language models will still agree with them if the user does as well. To reduce sycophancy, we present a straightforward synthetic-data intervention that takes public NLP tasks and encourages models to be robust to user opinions on these tasks. Adding these data in a lightweight finetuning step can significantly reduce sycophantic behavior on held-out prompts. Code for generating synthetic data for intervention can be found at https://github.com/google/sycophancy-intervention.

**Relevance and novelty impact.** It demonstrates the key behavioral dissociation that a model can know a statement is false yet agree with a user who endorses it, and it repairs this with lightweight synthetic fine-tuning. Teacher demonstrations and matched-prefix correction must beat or explain this simple baseline.

### Towards Understanding Sycophancy in Language Models

Links: [arXiv](https://arxiv.org/abs/2310.13548) · [PDF](https://arxiv.org/pdf/2310.13548)

**Abstract (verbatim)**

> Human feedback is commonly utilized to finetune AI assistants. But human feedback may also encourage model responses that match user beliefs over truthful ones, a behaviour known as sycophancy. We investigate the prevalence of sycophancy in models whose finetuning procedure made use of human feedback, and the potential role of human preference judgments in such behavior. We first demonstrate that five state-of-the-art AI assistants consistently exhibit sycophancy across four varied free-form text-generation tasks. To understand if human preferences drive this broadly observed behavior, we analyze existing human preference data. We find that when a response matches a user's views, it is more likely to be preferred. Moreover, both humans and preference models (PMs) prefer convincingly-written sycophantic responses over correct ones a non-negligible fraction of the time. Optimizing model outputs against PMs also sometimes sacrifices truthfulness in favor of sycophancy. Overall, our results indicate that sycophancy is a general behavior of state-of-the-art AI assistants, likely driven in part by human preference judgments favoring sycophantic responses.

**Relevance and novelty impact.** This establishes the bias and its relationship to feedback. It also cautions that “teacher as judge” is not automatically clean: preference models sometimes prefer convincing sycophancy. The project must empirically validate the chosen teacher in each role rather than define judge outputs as ground truth.

### Learning by Distilling Context

Links: [arXiv](https://arxiv.org/abs/2209.15189) · [PDF](https://arxiv.org/pdf/2209.15189)

**Abstract (verbatim)**

> Language models significantly benefit from context tokens, such as prompts or scratchpads. They perform better when prompted with informative instructions, and they acquire new reasoning capabilities by generating a scratch-pad before predicting the final answers. However, they do not \textit{internalize} these performance gains, which disappear when the context tokens are gone. Our work proposes to apply context distillation so that a language model can improve itself by internalizing these gains. Concretely, given a synthetic unlabeled input for the target task, we condition the model on ``[instructions] + [task-input]'' to predict ``[scratch-pad] + [final answer]''; then we fine-tune the same model to predict its own ``[final answer]'' conditioned on the ``[task-input]'', without seeing the ``[instructions]'' or using the ``[scratch-pad]''. We show that context distillation is a general method to train language models, and it can effectively internalize 3 types of training signals. First, it can internalize abstract task instructions and explanations, so we can iteratively update the model parameters with new instructions and overwrite old ones. Second, it can internalize step-by-step reasoning for complex tasks (e.g., 8-digit addition), and such a newly acquired capability proves to be useful for other downstream tasks. Finally, it can internalize concrete training examples, and it outperforms directly learning with gradient descent by 9\% on the SPIDER Text-to-SQL dataset; furthermore, combining context distillation operations can internalize more training examples than the context window size allows.

**Relevance and novelty impact.** The method intentionally changes the teacher/model context to elicit a desired behavior, then distills that behavior into a context-free student. This is a direct ancestor of eliciting a correction before the student's biased commitment. The novelty cannot be “remove a problematic context from teacher generation”; it must be the diagnosis that student-produced context corrupts the target and the controlled comparison among teacher roles.

## Judge–generator asymmetry and correction under anchoring

### Can LLMs Judge Better Than They Generate? Evaluating Task Asymmetry, Mechanistic Interpretability and Transferability for In-Context QA

Links: [arXiv](https://arxiv.org/abs/2606.28050) · [PDF](https://arxiv.org/pdf/2606.28050)

**Abstract (verbatim)**

> LLM-as-a-Judge and self-evaluation pipelines implicitly assume that evaluation is easier than generation. We test this in a controlled in-context QA setting where a context passage is the sole information source and each model judges the answer it generated, removing the parametric-knowledge confound of open-domain comparisons. Across four benchmarks (SQuAD 2.0, DROP, HotpotQA, MuSiQue) and two models, evaluation is not uniformly easier: generation accuracy exceeds self-evaluation on three of four, with multi-hop MuSiQue the exception. Attention analysis reveals why: evaluation attends to context 3--5x less than generation does and barely reads the candidate answer. LoRA fine-tuning confirms the asymmetry is not a training artifact: generation fine-tuning induces over-acceptance and evaluation fine-tuning degrades generation. These findings challenge core assumptions in self-evaluation pipelines.

**Relevance and novelty impact.** It makes the generator-versus-judge comparison itself an explicit research question and finds no universal judge advantage. The proposed study should not sell “models judge better than they generate” as a contribution. Its more defensible contribution is a within-teacher, bias-specific disagreement whose downstream optimization effects are measured.

### Large Language Models Cannot Self-Correct Reasoning Yet

Links: [arXiv](https://arxiv.org/abs/2310.01798) · [PDF](https://arxiv.org/pdf/2310.01798)

**Abstract (verbatim)**

> Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. A contemporary methodology, self-correction, has been proposed as a remedy to these issues. Building upon this premise, this paper critically examines the role and efficacy of self-correction within LLMs, shedding light on its true potential and limitations. Central to our investigation is the notion of intrinsic self-correction, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction. Drawing from these insights, we offer suggestions for future research and practical applications in this field.

**Relevance and novelty impact.** Once a bad response or premise is present in context, asking a model to continue/revise it need not recover the answer it could have generated independently. This supports the mechanism's plausibility but does not study distillation updates.

### Understanding the Dark Side of LLMs' Intrinsic Self-Correction

Links: [arXiv](https://arxiv.org/abs/2412.14959) · [project page](https://x-isc.info/)

**Abstract (verbatim)**

> Intrinsic self-correction was proposed to improve LLMs' responses via feedback prompts solely based on their inherent capability. However, recent works show that LLMs' intrinsic self-correction fails without oracle labels as feedback prompts. In this paper, we aim to interpret LLMs' intrinsic self-correction for different tasks, especially for those failure cases. By including one simple task and three complex tasks with state-of-the-art (SOTA) LLMs like ChatGPT families (o1, 4o, 3.5-turbo) and Llama families (2-7B, 3-8B, and 3.1-8B), we design three interpretation methods to reveal the dark side of LLMs' intrinsic self-correction. We identify intrinsic self-correction can (1) cause LLMs to waver both intermedia and final answers and lead to prompt bias on simple factual questions; (2) introduce human-like cognitive bias on complex tasks. In light of our findings, we also provide two simple yet effective strategies for alleviation: question repeating and supervised fine-tuning with a few samples. We open-source our work at https://x-isc.info/.

**Relevance and novelty impact.** Prompt bias and anchoring during correction are already documented, as are simple mitigation strategies. Repeating the original request and safety constraints is an inexpensive baseline: do so when querying the teacher after a compromised prefix and test whether this restores the refusal distribution.

## Broader conceptual ancestors

### A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning

Links: [arXiv](https://arxiv.org/abs/1011.0686) · [PMLR](https://proceedings.mlr.press/v15/ross11a.html)

**Abstract.** See the primary paper page; not reproduced here because the accessible search result did not expose a reliably copyable verbatim abstract.

**Relevance and novelty impact.** DAgger's classical move is to roll out the learner, query the expert on learner-visited states, and aggregate those expert labels. OPD is its autoregressive analogue in an important sense. The frozen-teacher project examines a boundary case that DAgger-style reasoning usually abstracts away: the expert's apparent action distribution may change because the learner's state encodes a behavioral commitment that the language-model teacher pragmatically continues. The distinction should be stated explicitly.

### Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks

Links: [arXiv](https://arxiv.org/abs/1506.03099) · [NeurIPS](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)

**Abstract.** See the primary paper page; not reproduced here because the accessible search result did not expose a reliably copyable verbatim abstract.

**Relevance and novelty impact.** Scheduled sampling addresses the mismatch between gold prefixes during training and self-generated prefixes at inference by gradually training under model-generated history. It is a conceptual ancestor of OPD and highlights the core tradeoff: learner-generated contexts improve state coverage but may make targets harder or less coherent. It does not compare teacher roles or biased commitments.

### Sequence-Level Knowledge Distillation

Links: [arXiv](https://arxiv.org/abs/1606.07947) · [ACL Anthology](https://aclanthology.org/D16-1139/)

**Abstract (verbatim)**

> Neural machine translation (NMT) offers a novel alternative formulation of translation that is potentially simpler than statistical approaches. However to reach competitive performance, NMT models need to be exceedingly large. In this paper we consider applying knowledge distillation approaches (Bucila et al., 2006; Hinton et al., 2015) that have proven successful for reducing the size of neural models in other domains to the problem of NMT. We demonstrate that standard knowledge distillation applied to word-level prediction can be effective for NMT, and also introduce two novel sequence-level versions of knowledge distillation that further improve performance, and somewhat surprisingly, seem to eliminate the need for beam search (even when applied on the original teacher model). Our best student model runs 10 times faster than its state-of-the-art teacher with little loss in performance. It is also significantly better than a baseline model trained without knowledge distillation: by 4.2/1.7 BLEU with greedy decoding/beam search. Applying weight pruning on top of knowledge distillation results in a student model that has 13 times fewer parameters than the original teacher model, with a decrease of 0.4 BLEU.

**Relevance and novelty impact.** This is the standard teacher-generated-trajectory alternative to student-prefix OPD. It motivates the teacher-demonstration SFT/off-policy control and makes clear that changing trajectory source is an old design axis, even if the behavioral-bias mechanism is new.

### Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision

Links: [arXiv](https://arxiv.org/abs/2312.09390) · [OpenAI](https://openai.com/index/weak-to-strong-generalization/)

**Abstract (verbatim)**

> Widely used alignment techniques, such as reinforcement learning from human feedback (RLHF), rely on the ability of humans to supervise model behavior - for example, to evaluate whether a model faithfully followed instructions or generated safe outputs. However, future superhuman models will behave in complex ways too difficult for humans to reliably evaluate; humans will only be able to weakly supervise superhuman models. We study an analogy to this problem: can weak model supervision elicit the full capabilities of a much stronger model? We test this using a range of pretrained language models in the GPT-4 family on natural language processing (NLP), chess, and reward modeling tasks. We find that when we naively finetune strong pretrained models on labels generated by a weak model, they consistently perform better than their weak supervisors, a phenomenon we call weak-to-strong generalization. However, we are still far from recovering the full capabilities of strong models with naive finetuning alone, suggesting that techniques like RLHF may scale poorly to superhuman models without further work. We find that simple methods can often significantly improve weak-to-strong generalization: for example, when finetuning GPT-4 with a GPT-2-level supervisor and an auxiliary confidence loss, we can recover close to GPT-3.5-level performance on NLP tasks. Our results suggest that it is feasible to make empirical progress today on a fundamental challenge of aligning superhuman models.

**Relevance and novelty impact.** Although teacher and student strengths are reversed relative to the proposal, this literature establishes that a student's inductive bias determines whether it imitates supervision errors or generalizes beyond them. It supports framing outcomes as an interaction among teacher labels, student priors, capacity, and loss—not simply teacher correctness.

## Recommended paper positioning and decisive controls

The paper should be positioned around **persistent capability transfer through student-induced teacher jailbreak states**, not reward repair, not discovery of prefill jailbreaks, and not generic OPD target drift. The core object is a two-model training channel: the student supplies the unsafe commitment, the stronger frozen teacher supplies the task knowledge in its continuation distribution, and OPD may fuse the two into one deployable student.

A defensible one-sentence contribution would be:

> We show that a student trained only to enter output-prefix jailbreak states can use on-policy distillation to acquire held-out unsafe capability from a stronger frozen teacher, and that replacing compromised continuation targets prevents this transfer while preserving benign distillation gains.

Minimum controls implied by the literature:

1. **Prefix-attack baselines:** forced output prefix, static prefill, optimized prefill, target-token log-probability search, and response-history priming.
2. **Prefix-only causal separation:** verify that prefix fine-tuning raises jailbreak-state visitation but does not itself raise held-out unsafe task accuracy. Include a matched amount of benign fine-tuning and ordinary harmful fine-tuning.
3. **Capability versus propensity:** score refusal/compliance separately from correctness or task success. Evaluate without the training prefill, on unseen requests, and on withheld capability categories.
4. **OPD increment:** compare the exact same prefix-trained checkpoint before and after OPD, standard off-policy/sequence KD, teacher-generated SFT, and student-only continued training under equal token budgets.
5. **IW-OPD / prefix compatibility:** cumulative teacher–student divergence, top-k overlap, fixed prefix-only supervision, and matched token positions.
6. **KL/objective choice:** forward KL, reverse KL or the exact practical OPD loss, plus per-token gradient contribution—not teacher generations or probabilities alone.
7. **Mechanism baselines:** raw token position, harmful-span onset, semantic commitment onset, teacher harm probes, and the early-window analyses motivated by autoregressive-consistency work.
8. **Teacher validation:** independently score generation, judgment, and continuation behavior against ground truth. Demonstrate that it recognizes harm while its continuation policy supplies useful unsafe probability mass.
9. **Target-side mitigations:** reset-and-regenerate; privileged safety context as in OPSA; judge/reward-guided decoding as in DeAL; counterfactual neutral prefixes; masking after commitment; and random worst-insertion/adversarial continuation training.
10. **Teacher-hardening upper bounds:** if feasible, compare a teacher trained for instruction priority or robust refusal (e.g. random-insertion training or DeepRefusal) to distinguish target repair from teacher repair.

The project is most likely publishable as a standalone work only if OPD transfers **incremental held-out unsafe capability**, rather than merely increasing compliance, and the transfer is causally attributable to compromised teacher targets rather than prefix fine-tuning or generic late-token drift. A convincing solution should beat generic position/compatibility weighting and simple safety-context re-querying while retaining benign OPD gains. If the result is only that output prefills jailbreak the teacher, it is already covered by direct prior work; if OPD changes refusal but not capability, it is better framed as safety degradation during distillation rather than capability transfer.

## Sources

1. Agarwal et al. [“On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.”](https://arxiv.org/abs/2306.13649) 2023.
2. Xie et al. [“On the Position Bias of On-Policy Distillation.”](https://arxiv.org/abs/2606.22600) 2026.
3. Liu et al. [“When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning.”](https://arxiv.org/abs/2605.21606) 2026.
4. Zhang et al. [“Fast and Effective On-policy Distillation from Reasoning Prefixes.”](https://arxiv.org/abs/2602.15260) 2026.
5. Zhao et al. [“Prefix-Guided On-Policy Distillation: Mining Golden Trajectories from Rollouts.”](https://arxiv.org/abs/2606.21994) 2026.
6. Gu et al. [“MiniLLM: Knowledge Distillation of Large Language Models.”](https://arxiv.org/abs/2306.08543) 2023.
7. Cheng, Agarwal, and Amiri. [“Do Students Debias Like Teachers? On the Distillability of Bias Mitigation Methods.”](https://arxiv.org/abs/2510.26038) 2025.
8. Rath. [“The Asymmetric Effects of Knowledge Distillation on Bias in Small Language Models.”](https://arxiv.org/abs/2607.28639) 2026.
9. Tiapkin et al. [“On Teacher Hacking in Language Model Distillation.”](https://proceedings.mlr.press/v267/tiapkin25a.html) ICML 2025.
10. Askin et al. [“Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer.”](https://arxiv.org/abs/2605.12798) 2026.
11. Guo et al. [“On-Policy Distillation for LLM Safety: A Routing Approach to Template-Robust Realignment.”](https://arxiv.org/abs/2607.27081) 2026.
12. Chua et al. [“Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought.”](https://arxiv.org/abs/2403.05518) 2024.
13. Irpan et al. [“Consistency Training Helps Stop Sycophancy and Jailbreaks.”](https://arxiv.org/abs/2510.27062) 2025.
14. Wei et al. [“Simple synthetic data reduces sycophancy in large language models.”](https://arxiv.org/abs/2308.03958) 2023.
15. Sharma et al. [“Towards Understanding Sycophancy in Language Models.”](https://arxiv.org/abs/2310.13548) 2023.
16. Snell, Klein, and Zhong. [“Learning by Distilling Context.”](https://arxiv.org/abs/2209.15189) 2022.
17. Bandyopadhyay. [“Can LLMs Judge Better Than They Generate?”](https://arxiv.org/abs/2606.28050) 2026.
18. Huang et al. [“Large Language Models Cannot Self-Correct Reasoning Yet.”](https://arxiv.org/abs/2310.01798) 2023.
19. Zhang et al. [“Understanding the Dark Side of LLMs' Intrinsic Self-Correction.”](https://arxiv.org/abs/2412.14959) 2024.
20. Ross, Gordon, and Bagnell. [“A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.”](https://proceedings.mlr.press/v15/ross11a.html) AISTATS 2011.
21. Bengio et al. [“Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.”](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html) NeurIPS 2015.
22. Kim and Rush. [“Sequence-Level Knowledge Distillation.”](https://aclanthology.org/D16-1139/) EMNLP 2016.
23. Burns et al. [“Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision.”](https://arxiv.org/abs/2312.09390) 2023.
24. Wang et al. [“Vulnerability of Large Language Models to Output Prefix Jailbreaks: Impact of Positions on Safety.”](https://aclanthology.org/2025.findings-naacl.219/) Findings of NAACL 2025.
25. Li et al. [“Prefill-Based Jailbreak: A Novel Approach of Bypassing LLM Safety Boundary.”](https://arxiv.org/abs/2504.21038) 2025.
26. Andriushchenko, Croce, and Flammarion. [“Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks.”](https://arxiv.org/abs/2404.02151) 2024.
27. Miao et al. [“Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models.”](https://arxiv.org/abs/2507.05248) 2025.
28. Kwon. [“Breaking Refusal in the First Half: A Mechanistic Study of the Prefill Jailbreak.”](https://arxiv.org/abs/2607.14147) 2026.
29. Lyu et al. [“When Autoregressive Consistency Hurts Safety Alignment.”](https://arxiv.org/abs/2606.04168) 2026.
30. Wei, Haghtalab, and Steinhardt. [“Jailbroken: How Does LLM Safety Training Fail?”](https://arxiv.org/abs/2307.02483) 2023.
31. Zhao et al. [“Weak-to-Strong Jailbreaking on Large Language Models.”](https://arxiv.org/abs/2401.17256) 2024.
32. Fu et al. [“Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation.”](https://arxiv.org/abs/2605.15239) 2026.
33. Li et al. [“SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment.”](https://arxiv.org/abs/2606.02530) 2026.
34. Zhang et al. [“Response-Based Knowledge Distillation for Multilingual Jailbreak Prevention Unwittingly Compromises Safety.”](https://arxiv.org/abs/2602.11157) 2025/2026 preprint.
35. Qi et al. [“Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!”](https://arxiv.org/abs/2310.03693) 2023.
36. Touvron et al. [“Llama 2: Open Foundation and Fine-Tuned Chat Models.”](https://arxiv.org/abs/2307.09288) 2023.
37. Hubinger et al. [“Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.”](https://arxiv.org/abs/2401.05566) 2024.
38. Xie et al. [“Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction.”](https://aclanthology.org/2025.findings-emnlp.956/) Findings of EMNLP 2025.
39. Wallace et al. [“The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions.”](https://arxiv.org/abs/2404.13208) 2024.
40. Huang et al. [“DeAL: Decoding-time Alignment for Large Language Models.”](https://aclanthology.org/2025.acl-long.1274/) ACL 2025.
41. Liang et al. [“AutoRAN: Weak-to-Strong Jailbreaking of Large Reasoning Models.”](https://arxiv.org/abs/2505.10846) 2025.
