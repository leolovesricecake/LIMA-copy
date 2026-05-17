# Phase A1 RCA: sentence_v2 Drift Incident (2026-05-17)

## Context
- Scope: `sentence` vs `sentence_v2` on `ERASER movie_reviews` (20 samples, deterministic, same GPU).
- Observation: `sentence_v2` fixed orphan punctuation chunks, but produced large explanation drift and mixed faithfulness metrics.

## What Failed
- Faithfulness drift exceeded acceptable range:
  - `log_odds` dropped significantly.
  - `sufficiency` and `aopc_sufficiency` regressed in direction.
- Explanation behavior changed broadly:
  - `selected_chunk_ids` and `chunk_ranking` changed on most samples.
- Speed gain was not large enough to justify semantic risk.

## Root Cause
- The previous `sentence_v2` implementation changed sentence boundaries with right-shift scanning over whitespace/symbols.
- This was not a local orphan fix; it re-shaped chunk semantics and `(subset text -> confidence)` paths.
- Greedy selection amplified small local boundary changes into large set-level explanation differences.

## Decision
- Keep `sentence` as mainline baseline.
- Rebuild `sentence_v2` as a safe variant:
  - preserve `sentence` boundary behavior;
  - only merge orphan punctuation-only chunks into the previous chunk.
- Add stronger diagnostics:
  - boundary jaccard/shift stats;
  - orphan merge hit counts;
  - large boundary-shift counters.

## Prevention
- Before 200-sample runs, require 20-sample deterministic checks with:
  - explanation drift summary (`selected/ranking/trace`);
  - boundary drift summary (`jaccard/shift/large-shift count`);
  - faithfulness direction checks.
