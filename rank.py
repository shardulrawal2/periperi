"""
Redrob Candidate Ranker v3
- Stage 1: Tight rules filter (100K -> ~1K)
- Stage 2: ONNX semantic embeddings + RRF fusion + recruitability gate
"""
import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent

# ── Constants ──
AI_TITLES = {
    'ai engineer', 'ml engineer', 'machine learning engineer', 'deep learning engineer',
    'data scientist', 'research scientist', 'ai research engineer', 'ai specialist',
    'nlp engineer', 'llm engineer', 'senior data engineer',
    'senior ai engineer', 'senior ml engineer', 'applied scientist',
    'senior data scientist', 'senior nlp engineer', 'computer vision engineer',
    'senior machine learning engineer', 'staff machine learning engineer',
    'applied ml engineer', 'machine learning researcher',
    'recommendation systems engineer', 'search engineer', 'nlp engineer',
}

SWE_TITLES = {
    'software engineer', 'senior software engineer', 'backend engineer',
    'full stack developer', 'full-stack developer', 'senior backend engineer',
    'platform engineer', 'infrastructure engineer',
}

NON_AI_TITLES = {
    'marketing manager', 'hr manager', 'accountant', 'sales executive',
    'customer support', 'graphic designer', 'content writer',
    'business analyst', 'operations manager', 'project manager',
    'civil engineer', 'mechanical engineer', 'electrical engineer',
}

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'tech mahindra', 'hcl', 'mindtree', 'hcl technologies',
    'ltimindtree', 'mphasis', 'hexaware', 'cyient', 'persistent systems',
}

AI_ROLE_KEYWORDS = {
    'machine learning', 'ml engineer', 'ai engineer', 'deep learning',
    'data scientist', 'nlp', 'computer vision', 'llm', 'recommendation',
    'search engineer', 'applied scientist', 'research scientist',
    'applied ml', 'ai specialist', 'ai research',
}


# ── JD Facet definitions (used by both keyword and embedding scoring) ──
JD_FACETS = [
    "retrieval_ranking",
    "embeddings_vector",
    "llm_finetuning",
    "production_ml",
    "search_recommendation",
    "senior_product_ai",
]

FACET_QUERIES = {
    "retrieval_ranking": (
        "building production retrieval and ranking systems. "
        "Information retrieval, search relevance, ranking algorithms, "
        "learning to rank, NDCG, MRR, recall, precision."
    ),
    "embeddings_vector": (
        "working with embeddings and vector search. "
        "Dense embeddings, vector databases, semantic search, "
        "embedding models, sentence transformers, ANN."
    ),
    "llm_finetuning": (
        "fine-tuning and deploying large language models. "
        "LLM fine-tuning, prompt engineering, RAG, retrieval augmented generation, "
        "model deployment, model serving, LoRA."
    ),
    "production_ml": (
        "building production ML systems end to end. "
        "ML pipelines, model deployment, A/B testing, "
        "model monitoring, feature stores, model serving."
    ),
    "search_recommendation": (
        "building search and recommendation systems. "
        "Search engines, recommender systems, personalization, "
        "collaborative filtering, content-based recommendation."
    ),
    "senior_product_ai": (
        "senior AI engineer at a product company. "
        "Technical leadership, system design, owning ML systems, "
        "production engineering at scale, product companies."
    ),
}


def safe_text(val):
    if val is None:
        return ''
    return str(val).lower().strip()


def safe(val, default=0):
    return val if val is not None else default


# ── Data loading ──
def load_candidates(path):
    candidates = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Fast consistency + relevance filter
# ═══════════════════════════════════════════════════════════════════════════════
def stage1_filter(candidates):
    """
    Tight filter: 100K -> ~500-1500.
    Only AI/SWE profiles with clear AI career history pass.
    """
    shortlist = []
    cands_by_id = {}

    for c in candidates:
        p = c['profile']
        skills = c.get('skills', [])
        history = c.get('career_history', [])
        title = safe_text(p.get('current_title', ''))
        exp = safe(p.get('years_of_experience', 0))
        cid = c['candidate_id']
        cands_by_id[cid] = c

        # ── 1. Hard honeypot checks ──
        total_skill_months = sum(s.get('duration_months', 0) or 0 for s in skills)
        exp_months = max(exp * 12, 1)

        if total_skill_months > exp_months * 20 and total_skill_months > 500:
            continue

        expert_zero = sum(1 for s in skills if s.get('proficiency') == 'expert' and (s.get('duration_months', 0) or 0) == 0)
        if expert_zero >= 10:
            continue

        is_non = any(t in title for t in NON_AI_TITLES)
        if is_non and len(skills) >= 20:
            continue

        # ── 2. Experience range ──
        if exp < 3.0 or exp > 12:
            continue

        # ── 3. Title + career check ──
        is_ai = any(t in title for t in AI_TITLES)
        is_swe = any(t in title for t in SWE_TITLES)

        # AI titles auto-pass
        if is_ai:
            shortlist.append(cid)
            continue

        # SWE — pass only if AI in career history
        if is_swe:
            has_ai_role = False
            for role in history:
                rt = safe_text(role.get('title', ''))
                if any(kw in rt for kw in AI_ROLE_KEYWORDS):
                    has_ai_role = True
                    break
            if has_ai_role:
                shortlist.append(cid)
                continue

        # Other titles — pass only if 2+ AI roles in career
        ai_role_count = 0
        for role in history:
            rt = safe_text(role.get('title', ''))
            if any(kw in rt for kw in AI_ROLE_KEYWORDS):
                ai_role_count += 1
        if ai_role_count >= 2:
            shortlist.append(cid)
            continue

    return shortlist, cands_by_id


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Semantic embedding + facet scoring + RRF fusion
# ═══════════════════════════════════════════════════════════════════════════════

def build_candidate_text(c):
    """
    Build a text block for a candidate by concatenating career descriptions.
    """
    texts = []
    for role in c.get('career_history', []):
        title = safe_text(role.get('title', ''))
        desc = safe_text(role.get('description', ''))
        company = safe_text(role.get('company', ''))
        texts.append(f"{title} at {company}. {desc}")
    return " ".join(texts)


def score_facets_embedding(candidate_text, facet_queries, session, tokenizer):
    """
    Score a candidate against all JD facets using ONNX embeddings.
    Returns dict of facet_name -> score (0-10).
    """
    from embed import embed_texts

    texts = [candidate_text] + list(facet_queries.values())
    embs = embed_texts(texts, session, tokenizer, batch_size=16)
    cand_emb = embs[0]
    facet_embs = embs[1:]

    scores = {}
    for i, fname in enumerate(facet_queries.keys()):
        sim = float(np.dot(cand_emb, facet_embs[i]))
        scores[fname] = max(0, sim) * 10.0
    return scores


def score_facets_keyword(c):
    """
    Fast keyword-based facet scoring for candidates not reaching embedding stage.
    (Backup for Stage 1 shortlist if embedding fails.)
    """
    p = c['profile']
    history = c.get('career_history', [])
    skills = c.get('skills', [])
    sig = c.get('redrob_signals', {})
    title = safe_text(p.get('current_title', ''))
    exp = safe(p.get('years_of_experience', 0))

    all_text = " ".join(
        safe_text(r.get('description', '')) + " " + safe_text(r.get('title', ''))
        for r in history
    )
    summary = safe_text(p.get('summary', ''))
    combined = all_text + " " + summary

    facet_scores = {}
    facet_scores['retrieval_ranking'] = sum(1 for kw in
        ['retrieval', 'ranking', 'search relevance', 'learn to rank', 'ndcg',
         'mrr', 'recall', 'precision', 'candidate generation', 're-ranking']
        if kw in combined)
    facet_scores['embeddings_vector'] = sum(1 for kw in
        ['embedding', 'vector', 'semantic search', 'vector database',
         'ann', 'approximate nearest', 'cosine similarity', 'dense retrieval']
        if kw in combined)
    facet_scores['llm_finetuning'] = sum(1 for kw in
        ['llm', 'fine-tuning', 'fine tuning', 'rag', 'retrieval augmented',
         'prompt', 'instruct', 'rlhf', 'lora', 'quantization']
        if kw in combined)
    facet_scores['production_ml'] = sum(1 for kw in
        ['production', 'deployed', 'shipped', 'a/b test', 'ab test',
         'pipeline', 'serving', 'monitoring', 'mlops', 'ci/cd', 'feature store']
        if kw in combined)
    facet_scores['search_recommendation'] = sum(1 for kw in
        ['search', 'recommendation', 'personalization', 'collaborative filtering',
         'content-based', 'hybrid recommendation', 'candidate retrieval']
        if kw in combined)
    facet_scores['senior_product_ai'] = sum(1 for kw in
        ['senior', 'lead', 'staff', 'architect', 'tech lead', 'principal',
         'mentor', 'roadmap', 'cross-functional']
        if kw in combined)

    # Scale to ~0-10 range
    return {k: min(v * 2, 10) for k, v in facet_scores.items()}


def rrf_fusion(facet_lists, k=60):
    """Reciprocal Rank Fusion."""
    rankings = {}
    for facet_name, scored_list in facet_lists:
        if not scored_list:
            continue
        rankings[facet_name] = {cid: rank + 1 for rank, (cid, _) in enumerate(scored_list)}

    all_cids = set()
    for _, scored_list in facet_lists:
        for cid, _ in scored_list:
            all_cids.add(cid)

    fused = {}
    for cid in all_cids:
        rrf_score = 0.0
        for facet_name in rankings:
            rank = rankings[facet_name].get(cid, len(rankings[facet_name]) + 1)
            rrf_score += 1.0 / (k + rank)
        fused[cid] = rrf_score
    return fused


def detect_honeypot(c):
    """Impossibility checks. Returns penalty 0-1."""
    p = c['profile']
    skills = c.get('skills', [])
    exp = safe(p.get('years_of_experience', 0))

    total_skill_months = sum(s.get('duration_months', 0) or 0 for s in skills)
    exp_months = max(exp * 12, 1)
    if total_skill_months > exp_months * 20 and total_skill_months > 500:
        return 1.0

    expert_zero = sum(1 for s in skills if s.get('proficiency') == 'expert' and (s.get('duration_months', 0) or 0) == 0)
    if expert_zero >= 10:
        return 1.0

    title = safe_text(p.get('current_title', ''))
    is_non = any(t in title for t in NON_AI_TITLES)
    if is_non and len(skills) >= 20:
        return 1.0

    return 0.0


def detect_aspirational(c):
    """Penalty for aspirational language. Returns 0-1."""
    summary = safe_text(c['profile'].get('summary', ''))
    aspirational_kw = [
        'curious about', 'experimented with chatgpt', 'keeping up with ai',
        'interested in transitioning', 'side project', 'self-learner',
        'online course', 'at a self-learner level',
    ]
    matches = sum(1 for kw in aspirational_kw if kw in summary)
    if matches >= 2:
        return 0.4
    if matches >= 1:
        return 0.2
    return 0.0


def rank_candidates(candidates, rrf_k=60):
    t0 = time.time()

    # ── Stage 1 ──
    shortlist_ids, cand_by_id = stage1_filter(candidates)
    t1 = time.time()
    print(f"  Stage 1: {len(candidates)} -> {len(shortlist_ids)} in {t1-t0:.2f}s", flush=True)

    if not shortlist_ids:
        return []

    shortlist = [cand_by_id[cid] for cid in shortlist_ids]

    # ── Stage 2: Semantic embedding + facet scoring ──
    use_embeddings = False
    session = None
    tokenizer = None
    try:
        from embed import load_model, embed_texts, get_facet_embeddings
        session, tokenizer = load_model()
        facet_embeddings = get_facet_embeddings(session, tokenizer)
        use_embeddings = True
        print(f"  Embedding model loaded", flush=True)
    except Exception as e:
        print(f"  Embedding model unavailable ({e}), using keyword fallback", flush=True)

    facet_scores = {f: [] for f in JD_FACETS}
    per_candidate_text = {}

    if use_embeddings:
        # Batch-embed all candidates at once
        texts = [build_candidate_text(c) for c in shortlist]
        for i, c in enumerate(shortlist):
            per_candidate_text[c['candidate_id']] = texts[i]
        cand_embs = embed_texts(texts, session, tokenizer, batch_size=32)

        for i, c in enumerate(shortlist):
            cid = c['candidate_id']
            cand_emb = cand_embs[i]
            scores = {}
            for j, fname in enumerate(JD_FACETS):
                sim = float(np.dot(cand_emb, facet_embeddings[j]))
                scores[fname] = max(0, sim) * 10.0
            for fname in JD_FACETS:
                facet_scores[fname].append((cid, scores[fname]))
    else:
        for c in shortlist:
            cid = c['candidate_id']
            text = build_candidate_text(c)
            per_candidate_text[cid] = text
            scores = score_facets_keyword(c)
            for fname in JD_FACETS:
                facet_scores[fname].append((cid, scores[fname]))

    t2 = time.time()
    print(f"  Stage 2: scored {len(shortlist)} candidates in {t2-t1:.2f}s", flush=True)

    # Sort each facet list
    for fname in facet_scores:
        facet_scores[fname].sort(key=lambda x: -x[1])

    # RRF fusion
    facet_lists = [(f, facet_scores[f]) for f in JD_FACETS]
    fused = rrf_fusion(facet_lists, k=rrf_k)

    if fused:
        max_fused = max(fused.values())
        if max_fused > 0:
            fused = {cid: s / max_fused for cid, s in fused.items()}

    # ── Apply penalties and recruitability gate ──
    results = []
    for c in shortlist:
        cid = c['candidate_id']
        base_score = fused.get(cid, 0.0)

        hp = detect_honeypot(c)
        if hp > 0.5:
            continue

        ap = detect_aspirational(c)

        sig = c.get('redrob_signals', {})
        response_rate = safe(sig.get('recruiter_response_rate', 0.0))
        open_to_work = sig.get('open_to_work_flag', False)
        recent_active = False
        try:
            last = datetime.strptime(str(sig.get('last_active_date', '2024-01-01')), '%Y-%m-%d')
            if (datetime.now() - last).days < 90:
                recent_active = True
        except (ValueError, TypeError):
            pass

        rec_mult = 0.3
        if open_to_work:
            rec_mult += 0.2
        if recent_active:
            rec_mult += 0.2
        rec_mult += response_rate * 0.3
        rec_mult = min(rec_mult, 1.0)

        final_score = base_score * (1.0 - ap) * rec_mult
        results.append((cid, final_score))

    results.sort(key=lambda x: (-x[1], x[0]))

    # ── Generate reasoning (top 100 only) ──
    top100 = []
    for i, (cid, score) in enumerate(results[:100]):
        rank = i + 1
        c = cand_by_id[cid]
        p = c['profile']
        sig = c.get('redrob_signals', {})

        title = p.get('current_title', 'Professional')
        exp = p.get('years_of_experience', 0)

        ai_skill_count = 0
        for s in c.get('skills', []):
            name = safe_text(s.get('name', ''))
            for kw in ['machine learning', 'deep learning', 'nlp', 'computer vision',
                        'llm', 'rag', 'embedding', 'data science', 'ai', 'pytorch',
                        'tensorflow', 'keras', 'scikit-learn', 'transformers',
                        'recommendation', 'search', 'ranking', 'neural network',
                        'reinforcement learning', 'statistical', 'mlops']:
                if kw in name:
                    ai_skill_count += 1
                    break

        has_product = False
        for role in c.get('career_history', []):
            company = safe_text(role.get('company', ''))
            if not any(f in company for f in CONSULTING_FIRMS):
                has_product = True
                break

        resp = safe(sig.get('recruiter_response_rate', 0))
        saved = safe(sig.get('saved_by_recruiters_30d', 0))
        github = safe(sig.get('github_activity_score', -1))

        parts = [f"{title} with {exp}yrs"]
        if ai_skill_count > 0:
            parts.append(f"{ai_skill_count} AI/ML skills")

        # Facet strengths from text
        text = per_candidate_text.get(cid, '')
        text_lower = text.lower()
        strengths = []
        if any(kw in text_lower for kw in ['retrieval', 'ranking', 'search relevance', 'ndcg', 'mrr']):
            strengths.append("retrieval/ranking")
        if any(kw in text_lower for kw in ['embedding', 'vector', 'semantic search']):
            strengths.append("embeddings/vector")
        if any(kw in text_lower for kw in ['llm', 'fine-tuning', 'rag', 'prompt']):
            strengths.append("LLM/fine-tuning")
        if any(kw in text_lower for kw in ['production', 'deployed', 'a/b test', 'pipeline']):
            strengths.append("prod ML")
        if strengths:
            parts.append("; ".join(strengths[:2]))

        if has_product:
            parts.append("product co")
        if resp > 0.5:
            parts.append(f"{resp:.0%} resp rate")
        if saved > 10:
            parts.append(f"saved {saved}x")
        if github > 50:
            parts.append("active GitHub")

        reasoning = "; ".join(parts)
        top100.append((cid, rank, f"{score:.4f}", reasoning))

    # Tie-break pass
    i = 0
    while i < len(top100) - 1:
        j = i
        while j < len(top100) - 1 and float(top100[j][2]) == float(top100[j + 1][2]):
            j += 1
        if j > i:
            group = sorted(top100[i:j + 1], key=lambda x: x[0])
            for k, (cid, _, score, reason) in enumerate(group, start=i + 1):
                top100[k - 1] = (cid, k, score, reason)
        i = j + 1

    total = time.time() - t0
    print(f"  Total: {total:.1f}s", flush=True)
    return top100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--out', default='submission.csv')
    parser.add_argument('--rrf-k', type=int, default=60)
    args = parser.parse_args()

    print("Loading candidates...", flush=True)
    start = time.time()
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} in {time.time() - start:.1f}s", flush=True)

    print("Ranking...", flush=True)
    start = time.time()
    top100 = rank_candidates(candidates, rrf_k=args.rrf_k)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for row in top100:
            writer.writerow(row)

    print(f"Written {len(top100)} rows to {args.out}", flush=True)
    for cid, rank, score, reason in top100[:5]:
        print(f"  #{rank}: {cid} ({score}) - {reason}", flush=True)


if __name__ == '__main__':
    main()
