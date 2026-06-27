---
title: Redrob Candidate Ranker
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Redrob Candidate Ranker

Two-stage semantic candidate ranker for the Redrob Senior AI Engineer challenge.

- **Stage 1 — fast rules filter (100K → ~1,800):** drops honeypots (impossible
  skill-month ratios, expert-but-zero-tenure skills, non-AI titles padded with
  skills) and anything clearly off-role. Experience is a *loose* gate here
  (2–20 yrs); fine-grained fit is handled in Stage 2 so strong senior/borderline
  candidates aren't lost.
- **Stage 2 — semantic scoring:** an ONNX MiniLM (all-MiniLM-L6-v2) embeds each
  candidate as several segments (summary/headline + the most recent roles, recency
  weighted) rather than one truncated blob. Each segment is scored against six
  JD-derived facets and **max-pooled** per facet, so a strong recent role isn't
  diluted by old ones. Facet scores are fused by a **JD-weighted sum of the raw
  cosine similarities** (magnitudes preserved), then adjusted by:
  - experience fit (soft, centred on 4–10 yrs),
  - an aspirational-language penalty,
  - a recruitability re-rank (0.7–1.0 multiplier — reachability breaks ties
    *within* a fit band instead of overriding fit), and
  - a small verified-signal bonus (skill-assessment scores, education tier, GitHub).

Output `reasoning` is derived from each candidate's actual top facets, so the
explanation reflects why they ranked where they did.

## Pre-computation

Run once before ranking:
```bash
pip install -r requirements.txt
python download_model.py
```

## Ranking

```bash
python rank.py --candidates candidates.jsonl --out submission.csv
```

Full 100K run is ~3 min on CPU (most of it Stage 2 embedding).

## Evaluation

The proxy ground truth uses **pooled IR-style judgments**: it labels the union of
the ranker's top-100 and a stratified random sample. Pooling the ranked items is
what makes NDCG measurable; it is not circular, because labels are assigned
independently of rank — from role-fit heuristics plus verified platform signals
(assessment scores, GitHub) that the ranker doesn't key on, so the GT can disagree
with the ranker.

```bash
python build_gt.py --ranking submission.csv
python eval.py --rankings submission.csv
```
