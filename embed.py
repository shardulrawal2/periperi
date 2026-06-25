"""
ONNX-based sentence embedding using all-MiniLM-L6-v2.
Used in stage 2 of the ranker for semantic facet scoring.
"""
import json
import os
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models" / "minilm-onnx"


def load_model():
    """Load ONNX session and tokenizer. Returns (session, tokenizer)."""
    model_path = MODEL_DIR / "onnx" / "model.onnx"
    tokenizer_path = MODEL_DIR / "tokenizer.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run download_model.py first."
        )

    import onnxruntime
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    session = onnxruntime.InferenceSession(str(model_path))

    return session, tokenizer


def embed_texts(texts, session, tokenizer, batch_size=32):
    """
    Embed a list of texts using ONNX MiniLM.
    Returns numpy array of shape (len(texts), 384).
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])),
        }

        outputs = session.run(None, ort_inputs)
        token_embeddings = outputs[0]

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), 1e-9, None)
        embeddings = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-9, None)

        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


# ── JD Facet queries for semantic similarity ──
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
        "learning to rank, NDCG, MRR, recall, precision, "
        "candidate generation, re-ranking, feature engineering for ranking."
    ),
    "embeddings_vector": (
        "working with embeddings and vector search. "
        "Dense embeddings, vector databases, semantic search, "
        "embedding models, text embeddings, sentence transformers, "
        "approximate nearest neighbor, ANN, vector similarity, "
        "semantic matching, representation learning."
    ),
    "llm_finetuning": (
        "fine-tuning and deploying large language models. "
        "LLM fine-tuning, prompt engineering, RAG, retrieval augmented generation, "
        "model deployment, model serving, LLM evaluation, "
        "instruction tuning, RLHF, model quantization, "
        "LoRA, parameter efficient fine-tuning."
    ),
    "production_ml": (
        "building production ML systems end to end. "
        "ML pipelines, model deployment, A/B testing, "
        "model monitoring, feature stores, model serving infrastructure, "
        "scaling ML systems, production model inference, "
        "MLOps, CI/CD for ML, experiment tracking."
    ),
    "search_recommendation": (
        "building search and recommendation systems. "
        "Search engines, recommender systems, personalization, "
        "collaborative filtering, content-based recommendation, "
        "hybrid recommendation, candidate retrieval, "
        "ranking for recommendations, real-time inference."
    ),
    "senior_product_ai": (
        "senior AI engineer at a product company. "
        "Technical leadership, mentoring, system design, "
        "cross-functional collaboration, owning ML systems, "
        "driving ML roadmap, production engineering at scale, "
        "working at technology product companies."
    ),
}


def get_facet_embeddings(session, tokenizer):
    """Pre-compute facet query embeddings."""
    queries = [FACET_QUERIES[f] for f in JD_FACETS]
    return embed_texts(queries, session, tokenizer)


def score_facets(candidate_history_text, facet_embeddings, candidate_embedding, session, tokenizer):
    """
    Score a candidate against all JD facets.
    Returns dict of facet -> score.
    """
    scores = {}
    for i, facet_name in enumerate(JD_FACETS):
        sim = float(np.dot(candidate_embedding, facet_embeddings[i]))
        scores[facet_name] = max(0, sim) * 10  # scale to ~0-10
    return scores
