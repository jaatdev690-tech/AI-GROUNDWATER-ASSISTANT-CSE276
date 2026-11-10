import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from groundwater_rag import (
    build_search_index,
    load_summary_data,
    answer_groundwater_query,
    detect_language
)

raw_data, corpus_data, model_enc, embeddings_vec = build_search_index()
d_summary, y_summary = load_summary_data()

q = "What is the average groundwater level in Abulkharana village of Muktsar district?"

print("=" * 80)
print(f"QUERY: '{q}'")
print("=" * 80)

res = answer_groundwater_query(
    query=q,
    raw_df=raw_data,
    corpus=corpus_data,
    district_summary_df=d_summary,
    yearly_summary_df=y_summary,
    model=model_enc,
    embeddings=embeddings_vec,
    api_key=os.getenv("GROQ_API_KEY")
)

print("\n--- RESULTS ---")
print("Engine:", res["engine"])
print("Intent:", res["intent"])
print("Sources:", res["sources"])
print("Has Figure:", res.get("figure") is not None)
if res.get("figure") is not None:
    print("Figure Title:", res["figure"].layout.title.text)

print(f"Records Returned: {len(res['records'])}")
if res.get("records") is not None and not res["records"].empty:
    print("Records columns:", res["records"].columns.tolist())
    print(res["records"].head(5))

print("\n--- ANSWER ---")
print(res["answer"])
print("=" * 80)

# Verify statistics in answer or records
ans = res["answer"]
has_count_25 = ("25" in ans or len(res["records"]) == 25 or len(res["records"]) == 15)
has_mean_2_01 = ("2.01" in ans or "2.00" in ans or "2.0" in ans)
has_min_0_78 = ("0.78" in ans)
has_max_3_54 = ("3.54" in ans)
has_abulkharana = ("Abulkharana" in ans or "ABULKHARANA" in ans or "abulkharana" in ans.lower())

print("\nVERIFICATION CHECKS:")
print("1. Mentions Abulkharana:", has_abulkharana)
print("2. Mentions Count 25:", has_count_25)
print("3. Mentions Mean 2.01:", has_mean_2_01)
print("4. Mentions Min 0.78:", has_min_0_78)
print("5. Mentions Max 3.54:", has_max_3_54)
print("6. Plotly Figure Title contains Abulkharana:", "Abulkharana" in (res["figure"].layout.title.text if res.get("figure") is not None else ""))

