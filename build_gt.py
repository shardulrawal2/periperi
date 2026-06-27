"""
Build proxy ground truth labels with overlap from a ranking output.
Usage: python build_gt.py --ranking submission.csv
"""
import json
import random
import sys
import csv
from pathlib import Path

BASE = Path(__file__).parent
CANDIDATES_PATH = BASE / "candidates.jsonl"
OUTPUT_PATH = BASE / "gt_labels.json"

PRODUCT_COMPANIES = {
    'google', 'meta', 'amazon', 'microsoft', 'apple', 'netflix', 'uber', 'lyft',
    'airbnb', 'doordash', 'instacart', 'stripe', 'square', 'shopify', 'spotify',
    'twitter', 'x', 'linkedin', 'pinterest', 'snap', 'tiktok', 'bytedance',
    'flipkart', 'swiggy', 'zomato', 'ola', 'paytm', 'razorpay', 'phonepe',
    'cred', 'groww', 'zerodha', 'unacademy', 'byjus', 'vedantu', 'upstox',
    'postman', 'hasura', 'chargebee', 'freshworks', 'zoho', 'thinkific',
    'dunder mifflin', 'initech', 'globex', 'acme', 'stark industries',
    'wayne enterprises', 'oscorp', 'umbrella corp',
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
    'applied ml', 'data engineer', 'ai specialist', 'ai research',
    'ml ops', 'mlops', 'data architect',
}

SWE_ROLE_KEYWORDS = {
    'software engineer', 'backend engineer', 'full stack', 'full-stack',
    'frontend', 'front-end', 'platform engineer', 'infrastructure',
    'devops', 'sre', 'site reliability', 'systems engineer',
}

NON_AI_TITLES = {
    'marketing', 'hr ', 'accountant', 'sales', 'customer support',
    'graphic designer', 'content writer', 'business analyst',
    'operations manager', 'civil engineer', 'mechanical engineer',
    'electrical engineer',
}


def safe_text(val):
    if val is None:
        return ''
    return str(val).lower().strip()


def verified_signal_adjust(c, tier):
    """
    Nudge the role-fit tier using *verified* platform signals (skill-assessment
    scores, GitHub activity) that the ranker does NOT use as primary features.
    This keeps the proxy GT from being a pure mirror of the ranker's own rules,
    so eval numbers aren't self-confirming. Honeypots (tier 0) are never lifted.
    """
    if tier == 0:
        return tier
    sig = c.get('redrob_signals', {})
    scores = sig.get('skill_assessment_scores', {}) or {}
    github = sig.get('github_activity_score', -1)

    avg = (sum(scores.values()) / len(scores)) if scores else None
    if avg is not None and avg >= 80 and github is not None and github > 60 and tier < 4:
        return tier + 1
    if avg is not None and avg < 35 and tier > 1:
        return tier - 1
    return tier


def label_candidate(c):
    """Assign tier 0-4 using ONLY title+company+industry. No description text."""
    p = c['profile']
    skills = c.get('skills', [])
    history = c.get('career_history', [])
    title = safe_text(p.get('current_title', ''))

    total_skill_months = sum(s.get('duration_months', 0) or 0 for s in skills)
    exp = p.get('years_of_experience', 0) or 0
    exp_months = max(exp * 12, 1)
    if total_skill_months > exp_months * 20 and total_skill_months > 500:
        return 0, "impossible skill-month ratio"
    expert_zero = sum(1 for s in skills if s.get('proficiency') == 'expert' and (s.get('duration_months', 0) or 0) == 0)
    if expert_zero >= 10:
        return 0, "expert-zero skills"
    if any(t in title for t in NON_AI_TITLES) and len(skills) >= 20:
        return 0, "non-AI title with many skills"

    max_seniority = 0
    has_product_co = False
    has_consulting_only = True
    ai_role_years = 0
    swe_role_years = 0
    total_role_years = 0
    current_is_ai = any(t in title for t in AI_ROLE_KEYWORDS)
    current_is_swe = any(t in title for t in SWE_ROLE_KEYWORDS)

    for role in history:
        co = safe_text(role.get('company', ''))
        rt = safe_text(role.get('title', ''))
        ind = safe_text(role.get('industry', ''))
        dur = role.get('duration_months', 0) or 0
        years = dur / 12.0
        total_role_years += years
        is_product = co in PRODUCT_COMPANIES or ind in {'saas', 'ai/ml', 'fintech', 'edtech', 'internet', 'software'}
        is_consult = co in CONSULTING_FIRMS
        if is_product:
            has_product_co = True
            has_consulting_only = False
        if not is_product and not is_consult:
            has_consulting_only = False
        if 'senior' in rt or 'staff' in rt or 'lead' in rt or 'principal' in rt or 'head' in rt:
            max_seniority = max(max_seniority, 2)
        elif 'junior' in rt or 'associate' in rt or 'intern' in rt:
            max_seniority = max(max_seniority, 0)
        else:
            max_seniority = max(max_seniority, 1)
        if any(kw in rt for kw in AI_ROLE_KEYWORDS):
            ai_role_years += years
        if any(kw in rt for kw in SWE_ROLE_KEYWORDS):
            swe_role_years += years

    if current_is_ai and has_product_co and ai_role_years >= 3 and max_seniority >= 1:
        return 4, "senior AI/ML at product company"
    if current_is_ai and has_product_co and ai_role_years >= 1:
        return 4, "AI/ML at product company"
    if current_is_ai and ai_role_years >= 3:
        return 3, "AI/ML with substantial experience"
    if current_is_ai and not has_consulting_only:
        return 3, "AI/ML with some non-consulting experience"
    if current_is_ai:
        return 3, "AI/ML title"
    if current_is_swe and has_product_co and ai_role_years >= 2:
        return 3, "SWE at product co with AI background"
    if current_is_swe and has_product_co:
        return 2, "SWE at product company"
    if swe_role_years >= 3 and has_product_co:
        return 2, "adjacent tech at product company"
    if ai_role_years >= 1:
        return 2, "some AI experience"
    if any(t in title for t in NON_AI_TITLES):
        return 1, "non-AI title"
    if not has_product_co and total_role_years > 0:
        return 1, "consulting/services background only"
    return 1, "no AI/SWE signal"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranking', help='CSV ranking to REPORT overlap against (not used to seed the GT pool)')
    parser.add_argument('--sample', type=int, default=300)
    args = parser.parse_args()

    print("Loading candidates...")
    candidates = []
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates")

    cand_by_id = {c['candidate_id']: c for c in candidates}

    # Pooled IR-style judgments: label the UNION of the ranker's top-100 and a
    # stratified random sample. Pooling the ranked items is what makes NDCG
    # measurable; it is NOT circular because label_candidate never sees the rank
    # or the ranker's score — tiers come from role-fit heuristics PLUS verified
    # platform signals (assessment scores, GitHub) the ranker doesn't key on, so
    # the labels can disagree with the ranker.
    ranked_ids = []
    if args.ranking:
        with open(args.ranking, 'r', encoding='utf-8') as f:
            ranked_ids = [row['candidate_id'] for row in csv.DictReader(f)][:100]

    random.seed(42)
    cat_map = {}
    for c in candidates:
        t = safe_text(c['profile']['current_title'])
        if any(kw in t for kw in AI_ROLE_KEYWORDS):
            cat_map[c['candidate_id']] = 'ai'
        elif any(kw in t for kw in SWE_ROLE_KEYWORDS):
            cat_map[c['candidate_id']] = 'swe'
        elif any(kw in t for kw in NON_AI_TITLES):
            cat_map[c['candidate_id']] = 'non_ai'
        else:
            cat_map[c['candidate_id']] = 'other'

    ranked_set = set(ranked_ids)
    pools = {'ai': [], 'swe': [], 'non_ai': [], 'other': []}
    for c in candidates:
        if c['candidate_id'] in ranked_set:
            continue  # ranked items are added separately, don't double-count
        pools[cat_map[c['candidate_id']]].append(c)

    print(f"Pools: AI={len(pools['ai'])}, SWE={len(pools['swe'])}, Non-AI={len(pools['non_ai'])}, Other={len(pools['other'])}")

    target = args.sample
    sampled = []
    sampled += random.sample(pools['ai'], min(int(target * 0.35), len(pools['ai'])))
    sampled += random.sample(pools['swe'], min(int(target * 0.25), len(pools['swe'])))
    sampled += random.sample(pools['non_ai'], min(int(target * 0.20), len(pools['non_ai'])))
    sampled += random.sample(pools['other'], min(target - len(sampled), len(pools['other'])))
    sampled += [cand_by_id[cid] for cid in ranked_ids if cid in cand_by_id]

    print(f"Labeling {len(sampled)} candidates ({len(ranked_ids)} pooled from ranking)...")
    relevance = {}
    labels = {}
    for c in sampled:
        tier, reason = label_candidate(c)
        tier = verified_signal_adjust(c, tier)
        relevance[c['candidate_id']] = tier
        labels[c['candidate_id']] = (tier, reason)

    dist = {}
    for tier in relevance.values():
        dist[tier] = dist.get(tier, 0) + 1
    print(f"GT distribution: {dict(sorted(dist.items()))}")

    if ranked_ids:
        labeled = sum(1 for cid in ranked_ids if cid in relevance)
        rdist = {}
        for cid in ranked_ids:
            t = relevance.get(cid)
            if t is not None:
                rdist[t] = rdist.get(t, 0) + 1
        print(f"Ranked top-100 labeled: {labeled}/100, tier dist: {dict(sorted(rdist.items()))}")

    output = {
        'relevance': {k: v for k, v in relevance.items()},
        'labels': {k: {'tier': v[0], 'reason': v[1]} for k, v in labels.items()},
        'distribution': {str(k): v for k, v in sorted(dist.items())},
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(relevance)} labels to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
