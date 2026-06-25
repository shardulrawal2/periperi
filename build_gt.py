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
    parser.add_argument('--ranking', help='CSV ranking file to ensure overlap with')
    parser.add_argument('--sample', type=int, default=120)
    args = parser.parse_args()

    print("Loading candidates...")
    candidates = []
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates")
    cand_by_id = {c['candidate_id']: c for c in candidates}

    sampled_ids = set()

    # Load ranking top 100 first to ensure overlap
    if args.ranking:
        with open(args.ranking, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sampled_ids.add(row['candidate_id'])
                if len(sampled_ids) >= 100:
                    break
        print(f"Got {len(sampled_ids)} candidates from ranking for overlap")

    # Fill remaining sample from stratified random pool
    random.seed(42)
    remaining = [c for c in candidates if c['candidate_id'] not in sampled_ids]

    # Pre-compute title categories for all remaining (faster than repeated computation)
    cat_map = {}
    for c in remaining:
        t = safe_text(c['profile']['current_title'])
        if any(kw in t for kw in AI_ROLE_KEYWORDS):
            cat_map[c['candidate_id']] = 'ai'
        elif any(kw in t for kw in SWE_ROLE_KEYWORDS):
            cat_map[c['candidate_id']] = 'swe'
        elif any(kw in t for kw in NON_AI_TITLES):
            cat_map[c['candidate_id']] = 'non_ai'
        else:
            cat_map[c['candidate_id']] = 'other'

    pools = {'ai': [], 'swe': [], 'non_ai': [], 'other': []}
    for c in remaining:
        pools[cat_map[c['candidate_id']]].append(c)

    print(f"Fill pools: AI={len(pools['ai'])}, SWE={len(pools['swe'])}, Non-AI={len(pools['non_ai'])}, Other={len(pools['other'])}")

    fill_target = args.sample - len(sampled_ids)
    fill = []
    fill += random.sample(pools['ai'], min(int(fill_target * 0.3), len(pools['ai'])))
    fill += random.sample(pools['swe'], min(int(fill_target * 0.25), len(pools['swe'])))
    fill += random.sample(pools['non_ai'], min(int(fill_target * 0.25), len(pools['non_ai'])))
    fill += random.sample(pools['other'], min(fill_target - len(fill), len(pools['other'])))
    sampled = fill + [cand_by_id[cid] for cid in sampled_ids if cid in cand_by_id]

    print(f"Labeling {len(sampled)} candidates...")
    relevance = {}
    labels = {}
    for c in sampled:
        tier, reason = label_candidate(c)
        relevance[c['candidate_id']] = tier
        labels[c['candidate_id']] = (tier, reason)

    dist = {}
    for tier in relevance.values():
        dist[tier] = dist.get(tier, 0) + 1
    print(f"GT distribution: {dict(sorted(dist.items()))}")

    overlap = sum(1 for cid in relevance if cid in sampled_ids)
    print(f"Overlap with ranking top 100 in GT: {overlap}/{len(relevance)}")

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
