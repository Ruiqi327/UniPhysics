# Reproducibility and changes from the experimental scripts

## Preserved behavior

- Both supplied datasets are copied byte-for-byte; no questions or labels are rewritten.
- Task-specific generation and judging prompts are retained from the supplied evaluator.
- Closed-form questions use deterministic rules; open-ended questions use two independent judgments and a third judgment for disagreement/invalid outputs.
- Answer extraction prefers the final boxed answer with balanced LaTeX braces, then alternative formats.
- Local generation retains one independent process/model replica per selected GPU.

## Deliberate corrections

1. **Credentials:** hardcoded API keys are removed with the old launcher configurations. Keys are read from environment variables, are never written to run manifests, and are not accepted as CLI arguments. API errors retain only error class, HTTP status and attempt number; server response bodies and exception text are not recorded.
2. **Paths:** installed-package-relative dataset lookup replaces machine-specific paths; output and user-supplied input paths are interpreted relative to the current working directory. A local model's path is reduced to its basename in persisted metadata; a hash retains its identity for resume checks.
3. **API sampling:** the supplied API helper had `temperature` and `top_p` commented out, making CLI flags ineffective. This release forwards explicitly supplied values and omits unset values.
4. **Data selection:** `all` now means every record, including the four removed-set IDs with additional prefixes. The old fixed whitelist silently excluded these records.
5. **Resume:** generation verifies dataset, model, configuration and code hashes. Judge caches verify the complete input prompt, model, endpoint, configuration and code. Only successful cached rows are reused; failed rows are retried. Different API credentials do not by themselves invalidate results.
6. **Failure accounting:** failed generation does not get graded as a wrong physics answer. Missing gold answers are marked `missing_gold` without an API request. Failed arbitration remains unresolved. Reports expose those categories and both resolved-only and all-record accuracy.
7. **Closed-form parsing:** bare gold letters are accepted, and multiple-choice matching supports A-J rather than only A-D. The matching remains conservative; it does not infer intent from ambiguous prose.
8. **Concurrency and writes:** API futures are bounded, outputs are incrementally flushed, final files use atomic replacement, and a run lock prevents concurrent writers.
9. **Metrics:** micro/macro accuracy, subject breakdowns, coverage, and failure counts are emitted as JSON and text.
10. **Local options:** seed, presence penalty, repetition penalty and validation of generation/context budgets are exposed.

## Relationship to the manuscript

The supplied manuscript describes 21-model experiments and reports the figures reproduced in this README. This refactoring has not rerun those experiments and does not distribute their raw model outputs. Fresh scores depend on exact checkpoints, provider model versions, decoding settings and judging models.

The manuscript describes a last-200-token extraction fallback; the supplied code uses the last 300 characters. This release preserves the code's behavior and documents the discrepancy rather than introducing an unspecified tokenizer. The API default is four retries after the initial attempt, matching the manuscript's stated retry count, whereas the supplied helper allowed five retries. Local model sampling parameters should be set explicitly according to the checkpoint recommendations and the manuscript's Table 3.

Changing generation sampling behavior or excluding the nine empty removed-set references can change measured scores. Compare the published tables only after aligning the full protocol and denominators. The cleaned and removed datasets retain their original bytes regardless of these implementation corrections.

## Validation scope

The test suite runs deterministic scoring, nested answer extraction, dataset integrity, API error/retry handling, HTTP request construction, generation-to-judging integration, cache invalidation, resume, safe metadata, and local pipeline contracts without GPU inference. Run `python tools/check_release.py` before creating a public archive. Full provider and GPU inference tests require your own credentials and hardware.

For a new run, begin with `--limit 5`. If a model rejects `max_tokens` or other chat-completion fields, use an endpoint implementing those fields or adapt the backend to that provider; this is an OpenAI-compatible Chat Completions client, not a universal vendor adapter.
