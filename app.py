"""
Redrob Hackathon - Streamlit Sandbox App
Run the ranker on a small candidate sample to verify it works.
"""
import streamlit as st
import json
import csv
import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="", layout="wide")

st.title("Redrob Hackathon - Candidate Ranker")
st.markdown("""
**Senior AI Engineer - Founding Team** at Redrob AI  
Upload a small candidate sample (≤100 candidates) to verify the ranking system works.
""")

st.sidebar.header("About")
st.sidebar.info(
    "This sandbox runs the same ranker used to produce the hackathon submission. "
    "Upload a .jsonl file with up to 100 candidates to see the ranked output."
)

uploaded_file = st.file_uploader(
    "Upload candidates.jsonl (max 100 lines)",
    type=["jsonl"],
    help="A JSONL file where each line is a JSON candidate object."
)

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    lines = [l.strip() for l in content.split("\n") if l.strip()]

    if len(lines) > 100:
        st.warning(f"File has {len(lines)} candidates. Only the first 100 will be used.")
        lines = lines[:100]

    candidates = []
    for line in lines:
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON line: {e}")

    st.success(f"Loaded {len(candidates)} candidates")

    if st.button("Run Ranking", type="primary"):
        with st.spinner("Ranking candidates..."):
            start = time.time()
            from rank import rank_candidates
            top100 = rank_candidates(candidates)
            elapsed = time.time() - start

        st.success(f"Ranking complete in {elapsed:.2f}s")

        st.subheader(f"Top {min(len(top100), 20)} Candidates")
        results_data = []
        for cid, rank, score, reasoning in top100[:20]:
            results_data.append({"Rank": rank, "Candidate ID": cid, "Score": f"{score}", "Reasoning": reasoning})
        st.table(results_data)

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank, score, reasoning in top100:
            writer.writerow([cid, rank, score, reasoning])

        st.download_button(
            label="Download Full Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name="submission.csv",
            mime="text/csv"
        )

        with st.expander("Show all 100 ranked candidates"):
            all_data = []
            for cid, rank, score, reasoning in top100:
                all_data.append({"Rank": rank, "Candidate ID": cid, "Score": f"{score}", "Reasoning": reasoning})
            st.table(all_data)
else:
    st.info("Upload a .jsonl file to get started.")
    st.markdown("""
    ### How to prepare a sample file
    ```bash
    head -n 50 candidates.jsonl > sample_50.jsonl
    ```
    ### Deploy on HuggingFace Spaces
    1. Create a new Space at https://huggingface.co/new-space
    2. Choose **Streamlit** as the SDK
    3. Upload `app.py`, `rank.py`, `embed.py`, and `requirements.txt`
    4. Run `python download_model.py` in the Space's terminal first
    """)

st.markdown("---")
st.caption("Redrob Hackathon - Intelligent Candidate Discovery & Ranking Challenge")
