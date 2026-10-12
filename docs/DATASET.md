# UniPhysics data card

## Scope

UniPhysics integrates English, text-only physics questions from ten benchmarks. The clean release spans school-level through advanced physics. It includes 5,352 curated examples and a separate audit corpus of 3,751 removed examples. These files are not training/validation/test splits of a single supervised learning dataset; the clean corpus is the evaluation benchmark and the removed corpus is a quality-control comparison set.

The accompanying manuscript describes five curation stages: schema unification, n-gram deduplication, question verification, answer verification, and final manual review/annotation. This release distributes the resulting data and evaluation code, not an executable reconstruction of those cleaning stages.

## Schema

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Original source-prefixed identifier, unique within each file |
| `problem` | string | Physics question with LaTeX and options when applicable |
| `answer` | string | Reference answer; may be empty or unreliable in the removed set |
| `answer_type` | string | `MCQ`, `MCQ(multiple)`, `Boolean`, or `Open-Ended` |
| `domain` | string | Physics subject annotation; `Unknown` where unavailable |
| `question_type` | string | Calculation, symbolic derivation, knowledge/conceptual, or `Unknown` |

UTF-8 JSONL contains one JSON object per physical newline. Parse using file iteration rather than `str.splitlines()`: the removed corpus includes Unicode line-separator characters inside quoted JSON strings.

## Snapshot statistics

The clean corpus has 2,774 `Quantitative Calculation`, 1,969 `Symbolic Derivation`, and 609 `Knowledge and Conceptual` questions. Its answer formats are 3,463 `Open-Ended`, 1,785 `MCQ`, 62 `Boolean`, and 42 `MCQ(multiple)`.

The removed corpus has 3,751 `Open-Ended` labels in the original supplied file, even for some questions whose text contains choices. Those labels are preserved rather than inferred or silently corrected. Nine entries have empty gold answers. Four retained IDs begin with `NBPhO_` or `PanPhO_`; source selection with `all` includes them. A complete machine-generated snapshot, including per-source counts and SHA-256 hashes, is in `Eval/benchmark/manifest.json`.

## Intended use and limitations

Use the clean file for model evaluation and the removed file for auditing, noise sensitivity, or clean-versus-removed comparisons. The removed file intentionally contains issues such as missing context, questionable reference answers, and unsuitable problem types. Its scores measure agreement with those supplied labels, not necessarily physical correctness. Do not present its labels as verified ground truth.

Text-only English coverage excludes multimodal and multilingual physics reasoning. Automated judging can still make mistakes, and provider/model updates can affect results. This code reports unresolved cases and coverage separately. Empty references are skipped explicitly, which changes the evaluable denominator for the removed set.

Consult `DATA_LICENSE.md` and the upstream sources for ownership and redistribution terms. No new blanket dataset license is asserted by the software license.
