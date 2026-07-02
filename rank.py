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

# Relative importance of each facet for this JD (Senior AI Engineer, founding team
# at a recruiting-search company → retrieval/ranking & embeddings weighted highest).
FACET_WEIGHTS = {
    "retrieval_ranking": 1.3,
    "embeddings_vector": 1.2,
    "llm_finetuning": 1.1,
    "production_ml": 1.0,
    "search_recommendation": 1.1,
    "senior_product_ai": 1.0,
}

FACET_LABELS = {
    "retrieval_ranking": "retrieval/ranking",
    "embeddings_vector": "embeddings/vector search",
    "llm_finetuning": "LLM/fine-tuning",
    "production_ml": "production ML",
    "search_recommendation": "search/recommendation",
    "senior_product_ai": "senior product-AI",
}


def safe_text(val):
    if val is None:
        return ''
    return str(val).lower().strip()


def safe(val, default=0):
    return val if val is not None else default


def _parse_date(val):
    """Parse a YYYY-MM-DD string to datetime, or None if absent/malformed."""
    try:
        return datetime.strptime(str(val), '%Y-%m-%d')
    except (ValueError, TypeError):
        return None


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

        # ── 1. Hard honeypot checks (single source of truth: detect_honeypot) ──
        if detect_honeypot(c) > 0.5:
            continue

        # ── 2. Experience range (loose hard gate only — fine-grained fit is a
        #       soft factor in Stage 2 via experience_fit, so we don't drop
        #       borderline-junior or very-senior candidates here) ──
        if exp < 2.0 or exp > 20:
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


def build_candidate_segments(c):
    """
    Break a candidate into scorable text segments, recency-ordered.

    Unlike build_candidate_text (one mean-pooled, easily-truncated blob), this
    keeps the summary/headline (the richest fit signal) and the most recent roles
    as separate segments so each can be matched against a facet on its own and
    max-pooled — a strong recent AI role is no longer diluted by old roles, and
    nothing of value is lost to the 256-token truncation.

    Returns a list of (text, recency_weight) tuples.
    """
    p = c['profile']
    title = safe_text(p.get('current_title', ''))
    summary = safe_text(p.get('summary', ''))
    headline = safe_text(p.get('headline', ''))

    segments = []
    head_text = summary or headline
    if head_text:
        segments.append((f"{title}. {head_text}", 1.0))

    roles = c.get('career_history', []) or []
    roles_sorted = sorted(
        roles,
        key=lambda r: (bool(r.get('is_current')), str(r.get('start_date', ''))),
        reverse=True,
    )
    for i, role in enumerate(roles_sorted[:4]):
        rt = safe_text(role.get('title', ''))
        co = safe_text(role.get('company', ''))
        desc = safe_text(role.get('description', ''))
        recency_w = max(0.6, 1.0 - 0.12 * i)  # 1.0, .88, .76, .64
        segments.append((f"{rt} at {co}. {desc}", recency_w))

    if not segments:
        segments.append((build_candidate_text(c), 1.0))
    return segments


def experience_fit(exp):
    """Soft experience fit (0.6-1.0), centred on the 4-10yr sweet spot."""
    exp = exp or 0
    if exp < 2:
        return 0.6
    if exp < 4:
        return 0.75 + 0.25 * (exp - 2) / 2.0
    if exp <= 10:
        return 1.0
    if exp <= 20:
        return 1.0 - 0.3 * (exp - 10) / 10.0
    return 0.7


def quality_mult(c):
    """
    Mild quality multiplier (0.95-1.15) from *verified* signals the rest of the
    pipeline doesn't key on: Redrob skill-assessment scores, education tier, and
    GitHub activity. Kept small so it re-ranks within a fit band rather than
    overriding semantic fit.
    """
    mult = 1.0
    sig = c.get('redrob_signals', {})

    scores = sig.get('skill_assessment_scores', {}) or {}
    if scores:
        avg = sum(scores.values()) / len(scores)
        mult += 0.10 * (avg - 60) / 40.0  # 60→+0.0, 100→+0.10, 20→-0.10

    tiers = {e.get('tier') for e in (c.get('education', []) or [])}
    if 'tier_1' in tiers:
        mult += 0.05
    elif 'tier_2' in tiers:
        mult += 0.02

    if safe(sig.get('github_activity_score', -1)) > 70:
        mult += 0.03

    return max(0.95, min(mult, 1.15))


def reference_date(candidates):
    """
    Anchor 'recent activity' to the dataset's latest last_active_date rather than
    wall-clock time, so the recency signal is stable regardless of when we run.
    """
    latest = None
    for c in candidates:
        d = c.get('redrob_signals', {}).get('last_active_date')
        if not d:
            continue
        try:
            dt = datetime.strptime(str(d), '%Y-%m-%d')
        except (ValueError, TypeError):
            continue
        if latest is None or dt > latest:
            latest = dt
    return latest or datetime.now()


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


def detect_honeypot(c):
    """
    Impossibility (not irrelevance) checks. Each rule flags a profile whose
    self-reported facts contradict each other in a way no real candidate can —
    so it survives the relevance gate but should never reach the top of a ranking.
    Returns penalty 0-1.

    The four rules below were chosen by scanning every structured contradiction the
    schema permits and keeping only the ones that fire at *honeypot scale* (tens of
    profiles, ~0.01-0.03%). Contradictions that fire at population scale (e.g. a
    single skill's duration exceeding total career: 18%; per-skill endorsements
    summing above the platform total: 92%) are dataset artifacts, NOT traps, and are
    deliberately excluded. The rules also avoid title/role: a marketing manager with
    a long skill list is irrelevant, not impossible — the Stage 1 relevance gate
    already drops them.

    Explicitly REJECTED rule: "more 'expert' skills than years of experience" fires
    on 119 profiles, 34 of which are in our own top-100. Concurrent expertise across
    many tools is normal for strong senior engineers, so it flags breadth, not
    impossibility — the same conflation the original detector made. Do not add it.
    """
    p = c['profile']
    skills = c.get('skills', [])
    history = c.get('career_history', [])
    exp = safe(p.get('years_of_experience', 0))

    # Rule 1: claimed skill-experience-months vastly exceed possible given tenure.
    total_skill_months = sum(s.get('duration_months', 0) or 0 for s in skills)
    exp_months = max(exp * 12, 1)
    if total_skill_months > exp_months * 20 and total_skill_months > 500:
        return 1.0

    # Rule 2: "expert" proficiency claimed with zero months of backing. The spec's
    # flagship honeypot ("expert in N skills, 0 months"); the seeded population
    # clusters at 3-5 such skills (nobody legit has even one), so gate at >=3.
    expert_zero = sum(1 for s in skills if s.get('proficiency') == 'expert' and (s.get('duration_months', 0) or 0) == 0)
    if expert_zero >= 3:
        return 1.0

    # Rule 3: career-history tenure far exceeds stated years of experience. Roles
    # can overlap, so we only flag a large, unambiguous contradiction (>1.5x and a
    # >3yr absolute gap) — you cannot have worked 1.5x longer than you've existed
    # professionally.
    role_years = sum(r.get('duration_months', 0) or 0 for r in history) / 12.0
    if exp > 0 and role_years > exp * 1.5 and role_years - exp > 3:
        return 1.0

    # Rule 4: stated years-of-experience exceed the actual calendar span of the
    # career. Anchored on the candidate's own latest known date (role dates or last
    # active) vs their earliest role start; a >3yr surplus is impossible — you can't
    # have more experience than time has elapsed since your first job.
    starts = [_parse_date(r.get('start_date')) for r in history]
    starts = [d for d in starts if d]
    if exp > 0 and starts:
        known = starts + [_parse_date(r.get('end_date')) for r in history]
        known.append(_parse_date(c.get('redrob_signals', {}).get('last_active_date')))
        known = [d for d in known if d]
        career_span = (max(known) - min(starts)).days / 365.25
        if exp > career_span + 3:
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


def rank_candidates(candidates):
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

    n = len(shortlist)
    # Per-candidate, per-facet semantic scores, max-pooled over profile segments.
    facet_matrix = np.zeros((n, len(JD_FACETS)), dtype=np.float32)

    if use_embeddings:
        # Flatten every candidate's segments into one batch, embed once, then
        # max-pool similarities back per candidate/facet.
        seg_texts, seg_owner, seg_weight = [], [], []
        for ci, c in enumerate(shortlist):
            for text, w in build_candidate_segments(c):
                seg_texts.append(text)
                seg_owner.append(ci)
                seg_weight.append(w)

        seg_embs = embed_texts(seg_texts, session, tokenizer, batch_size=64)
        facet_arr = np.asarray(facet_embeddings, dtype=np.float32)   # (F, 384)
        sims = np.clip(seg_embs @ facet_arr.T, 0.0, None)            # (S, F)
        for s in range(len(seg_texts)):
            ci = seg_owner[s]
            facet_matrix[ci] = np.maximum(facet_matrix[ci], seg_weight[s] * sims[s])
    else:
        for ci, c in enumerate(shortlist):
            scores = score_facets_keyword(c)
            for j, fname in enumerate(JD_FACETS):
                facet_matrix[ci, j] = scores[fname] / 10.0

    t2 = time.time()
    print(f"  Stage 2: scored {len(shortlist)} candidates in {t2-t1:.2f}s", flush=True)

    # ── Fusion: JD-weighted sum of the raw (magnitude-preserving) facet scores.
    #    Replaces RRF, which discarded the cosine magnitudes and added little over
    #    a mean across these highly-correlated facet rankings. ──
    fweights = np.array([FACET_WEIGHTS[f] for f in JD_FACETS], dtype=np.float32)
    semantic = facet_matrix @ fweights                              # (n,)
    max_sem = float(semantic.max()) if n else 0.0
    if max_sem > 0:
        semantic = semantic / max_sem

    fused = {shortlist[i]['candidate_id']: float(semantic[i]) for i in range(n)}
    facet_by_cid = {shortlist[i]['candidate_id']: facet_matrix[i] for i in range(n)}

    # ── Apply penalties and recruitability re-rank ──
    ref_date = reference_date(candidates)
    results = []
    for c in shortlist:
        cid = c['candidate_id']
        base_score = fused.get(cid, 0.0)

        if detect_honeypot(c) > 0.5:
            continue

        ap = detect_aspirational(c)
        ef = experience_fit(safe(c['profile'].get('years_of_experience', 0)))

        sig = c.get('redrob_signals', {})
        response_rate = safe(sig.get('recruiter_response_rate', 0.0))
        open_to_work = sig.get('open_to_work_flag', False)
        recent_active = False
        try:
            last = datetime.strptime(str(sig.get('last_active_date', '')), '%Y-%m-%d')
            if (ref_date - last).days < 90:
                recent_active = True
        except (ValueError, TypeError):
            pass

        # Recruitability re-ranks WITHIN a fit band (0.7-1.0) rather than swinging
        # the score 3x, so reachability no longer overrides actual fit.
        signal = 0.0
        if open_to_work:
            signal += 0.4
        if recent_active:
            signal += 0.3
        signal += response_rate * 0.3
        rec_mult = 0.7 + 0.3 * min(signal, 1.0)

        # response_rate is already reflected in the score via rec_mult; the
        # submission spec requires equal-score ties to break on candidate_id asc.
        final_score = base_score * ef * (1.0 - ap) * rec_mult * quality_mult(c)
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

        # Faithful strengths: the facets that actually drove this candidate's score.
        fvec = facet_by_cid.get(cid)
        if fvec is not None and float(fvec.max()) > 0:
            order = np.argsort(-fvec)
            strengths = [FACET_LABELS[JD_FACETS[j]] for j in order[:2] if fvec[j] > 0]
            if strengths:
                parts.append("; ".join(strengths))

        if has_product:
            parts.append("product co")
        if resp > 0.5:
            parts.append(f"{resp:.0%} resp rate")
        if saved > 10:
            parts.append(f"saved {saved}x")
        search = safe(sig.get('search_appearance_30d', 0))
        if search > 500:
            parts.append(f"search {search}x")
        if github > 50:
            parts.append("active GitHub")

        reasoning = "; ".join(parts)
        top100.append((cid, rank, f"{score:.4f}", reasoning))

    # Two scores can differ as floats yet print identically at 4dp; the submission
    # spec requires such printed-equal ties to be ordered by candidate_id ascending.
    i = 0
    while i < len(top100) - 1:
        j = i
        while j < len(top100) - 1 and top100[j][2] == top100[j + 1][2]:
            j += 1
        if j > i:
            group = sorted(top100[i:j + 1], key=lambda x: x[0])
            for offset, (cid, _, score, reason) in enumerate(group):
                top100[i + offset] = (cid, i + offset + 1, score, reason)
        i = j + 1

    # ── Safety guard: no honeypot may reach the final top-100 ──
    # Submissions with >10% honeypots in the top-100 are auto-disqualified. Stage 1
    # and the re-rank both already drop honeypots, so this is belt-and-suspenders:
    # fail loudly rather than ship a DQ'd file if a future change regresses.
    leaked = [cid for cid, *_ in top100 if detect_honeypot(cand_by_id[cid]) > 0.5]
    if leaked:
        raise AssertionError(
            f"{len(leaked)} honeypot(s) leaked into top-100 (DQ risk): {leaked[:5]}"
        )

    total = time.time() - t0
    print(f"  Total: {total:.1f}s", flush=True)
    return top100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--out', default='submission.csv')
    args = parser.parse_args()

    print("Loading candidates...", flush=True)
    start = time.time()
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} in {time.time() - start:.1f}s", flush=True)

    print("Ranking...", flush=True)
    start = time.time()
    top100 = rank_candidates(candidates)

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
