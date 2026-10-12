# Dataset provenance and licensing

The MIT license at the repository root applies to the UniPhysics evaluation code. It **does not** grant a new license to third-party benchmark questions, answers, or manuscript figures. These materials retain their upstream rights and restrictions; a unified file format does not change their ownership.

The source materials supplied for this release did not include a complete set of upstream license texts. The table below records attribution from the accompanying manuscript, **not a verified permission to redistribute every upstream source**. Maintainers should verify the applicable versions, redistribution terms, and required notices before publishing the data to a public host. Users should consult and credit the original benchmark publications and distribution pages.

| Source | Attribution in the manuscript |
| --- | --- |
| JEEBench | Arora, Singh, and Mausam (2023), *Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark for Large Language Models* |
| MMLU-Pro | Wang et al. (2024), *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark* |
| GPQA | Rein et al. (2023), *GPQA: A Graduate-Level Google-Proof Q&A Benchmark* |
| PhysicsQA | Jaiswal et al. (2024), *Improving Physics Reasoning in Large Language Models Using Mixture of Refinement Agents* |
| PHYSICS | Zheng et al. (2025), *Scaling Physical Reasoning with the PHYSICS Dataset* |
| UGPhysics | Xu et al. (2025), *UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with Large Language Models* |
| PhyBench | Qiu et al. (2025), *PhyBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models* |
| TheoremQA | Chen et al. (2023), *TheoremQA: A Theorem-driven Question Answering Dataset* |
| SciBench | Wang et al. (2023), *SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models* |
| HiPhO | Yu et al. (2025), *HiPhO: How Far Are (M)LLMs from Humans in the Latest High School Physics Olympiad Benchmark?* |

Four removed-set records retain `NBPhO_` and `PanPhO_` prefixes. Their exact upstream redistribution terms are not established by the supplied files either. They are preserved as audit records and are not counted as additional clean-set benchmark sources.

The two PNGs in `docs/assets/` are figure crops from the UniPhysics manuscript supplied by its author, with source figure/page details in `docs/assets/README.md`. They are research illustrations, not evaluation outputs produced by this refactoring.
