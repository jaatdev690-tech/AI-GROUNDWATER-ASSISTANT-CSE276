import sys
import os
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from groundwater_rag import (
    build_search_index,
    load_summary_data,
    answer_groundwater_query,
    detect_language
)

def test_12_queries():
    print("=" * 80)
    print("EXECUTING PHASE 8 VERIFICATION: 12 MASTER QUERIES")
    print("=" * 80)

    raw_data, corpus_data, model_enc, embeddings_vec = build_search_index()
    d_summary, y_summary = load_summary_data()

    queries = [
        ("Query 1: Highest Average Groundwater Depth", "What is the highest average groundwater depth in Punjab?"),
        ("Query 2: Lowest Average Groundwater Depth", "What is the lowest average groundwater depth in Punjab?"),
        ("Query 3: Average Groundwater Depth in Ludhiana", "What is the average groundwater depth in Ludhiana?"),
        ("Query 4: Historical Groundwater Trend for Punjab", "What is the historical groundwater trend for Punjab?"),
        ("Query 5: Punjab Yearly Trend (Hinglish)", "Punjab ka yearly groundwater trend batao"),
        ("Query 6: Punjab Conditions in Hindi", "Explain Punjab groundwater conditions in Hindi"),
        ("Query 7: Compare Trends in Ludhiana & Patiala", "Compare groundwater trends in Ludhiana and Patiala"),
        ("Query 8: Village Hussainpura Status", "What is the groundwater condition in village Hussainpura?"),
        ("Query 9: Nitrate Concentration in Punjab", "What is the nitrate concentration in Punjab?"),
        ("Query 10: Difference Between Depth and Quality", "Explain the difference between groundwater depth and groundwater quality"),
        ("Query 11: District-wise Groundwater Ranking", "District-wise groundwater ranking in Punjab"),
        ("Query 12: Groundwater Trend Over Available Years", "Groundwater trend over available years in Punjab")
    ]

    results = []

    for label, q in queries:
        print("\n" + "#" * 80)
        print(f"RUNNING {label}")
        print(f"QUERY: '{q}'")
        lang = detect_language(q)
        print(f"DETECTED LANGUAGE: {lang}")

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

        engine = res["engine"]
        intent = res["intent"]
        sources = res["sources"]
        has_fig = res.get("figure") is not None
        rec_count = len(res["records"]) if res.get("records") is not None else 0
        ans_preview = res["answer"][:250].replace("\n", " ")

        print(f"-> Engine: {engine}")
        print(f"-> Intent: {intent}")
        print(f"-> Plotly Chart Generated: {has_fig}")
        print(f"-> Supporting Records: {rec_count}")
        print(f"-> Sources ({len(sources)}): {sources[:2]}")
        print(f"-> Answer Preview: {ans_preview}...")

        results.append({
            "label": label,
            "query": q,
            "detected_language": lang,
            "engine": engine,
            "intent": intent,
            "has_figure": has_fig,
            "records_count": rec_count,
            "sources": sources,
            "answer_full": res["answer"]
        })

    # Save complete results to a json file for easy report generation
    with open("verify_12_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("ALL 12 MASTER QUERIES COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_12_queries()

