---
title: Redrob Candidate Ranker
emoji: 
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Redrob Candidate Ranker

Two-stage semantic candidate ranker for the Redrob Senior AI Engineer challenge.

- Stage 1: Fast rules filter (100K → ~1,700)
- Stage 2: ONNX MiniLM embeddings + RRF fusion + recruitability gate

## Pre-computation

Run once before ranking:
```bash
pip install -r requirements.txt
python download_model.py
```

## Ranking

```bash
python rank.py --candidates candidates.jsonl --out submission.csv --rrf-k 60
```

## Evaluation

```bash
python build_gt.py --ranking submission.csv
python eval.py --rankings submission.csv
```
