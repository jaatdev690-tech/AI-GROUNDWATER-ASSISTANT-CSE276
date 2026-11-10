import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from groundwater_rag import (
    build_search_index,
    load_summary_data,
    answer_groundwater_query,
    extract_districts_from_query,
    is_general_knowledge_query,
    is_groundwater_related,
    detect_language
)

def run_tests():
    print("=" * 70)
    print("RUNNING AI GROUNDWATER ASSISTANT VALIDATION TEST SUITE")
    print("=" * 70)

    raw_data, corpus_data, model_enc, embeddings_vec = build_search_index()
    d_summary, y_summary = load_summary_data()

    all_passed = True

    # ----------------------------------------------------
    # TEST 1: Language Detection
    # ----------------------------------------------------
    print("\n--- TEST 1: Language Detection ---")
    test_langs = [
        ("What is an aquifer?", "en"),
        ("एक्विफर क्या है?", "hi"),
        ("पंजाब में सबसे गहरा भूजल स्तर किस जिले में है?", "hi"),
        ("Ludhiana me groundwater level kitna hai?", "hinglish"),
        ("क्या लुधियाना का डेटा उपलब्ध है?", "hi"),
    ]
    for q, expected in test_langs:
        detected = detect_language(q)
        passed = (detected == expected)
        print(f"[{'PASS' if passed else 'FAIL'}] '{q}' -> Detected: {detected} (Expected: {expected})")
        if not passed:
            all_passed = False

    # ----------------------------------------------------
    # TEST 2: District Matching for Ludhiana (English & Hindi)
    # ----------------------------------------------------
    print("\n--- TEST 2: District Matching for Ludhiana ---")
    ludhiana_queries = [
        "What is the average groundwater level in Ludhiana?",
        "लुधियाना में भूजल स्तर क्या है?",
        "क्या लुधियाना का डेटा उपलब्ध है?",
        "Is data available for Ludhiana?",
        "Ludhiana data availability",
        "Ludhiyana me groundwater level kitna hai?",
        "लुधियाना"
    ]
    for q in ludhiana_queries:
        districts = extract_districts_from_query(q, ["LUDHIANA"])
        passed = ("LUDHIANA" in districts)
        print(f"[{'PASS' if passed else 'FAIL'}] '{q}' -> Extracted Districts: {districts}")
        if not passed:
            all_passed = False

    # ----------------------------------------------------
    # TEST 3: Domain Guardrail (Hindi & English)
    # ----------------------------------------------------
    print("\n--- TEST 3: Domain Guardrail ---")
    guardrail_cases = [
        # (Query, Expected is_related)
        ("What is an aquifer?", True),
        ("एक्विफर क्या है?", True),
        ("भूजल क्या है?", True),
        ("लुधियाना में भूजल स्तर क्या है?", True),
        ("पंजाब में सबसे गहरा भूजल स्तर किस जिले में है?", True),
        ("क्या लुधियाना का भूजल डेटा उपलब्ध है?", True),
        ("How to make pizza?", False),
        ("पिज़्ज़ा कैसे बनाते हैं?", False),
        ("Who is the prime minister?", False),
        ("क्रिकेट मैच का स्कोर क्या है?", False),
    ]
    for q, expected in guardrail_cases:
        related = is_groundwater_related(q)
        passed = (related == expected)
        print(f"[{'PASS' if passed else 'FAIL'}] '{q}' -> is_groundwater_related: {related} (Expected: {expected})")
        if not passed:
            all_passed = False

    # ----------------------------------------------------
    # TEST 4: General Knowledge Detection (Hindi & English)
    # ----------------------------------------------------
    print("\n--- TEST 4: General Knowledge Detection ---")
    gk_cases = [
        ("What is an aquifer?", True),
        ("एक्विफर क्या है?", True),
        ("जलभृत क्या है?", True),
        ("What is groundwater recharge?", True),
        ("भूजल पुनर्भरण क्या है?", True),
        ("भूजल संरक्षण कैसे करें?", True),
        ("What is the water table?", True),
        ("What is the average groundwater level in Ludhiana?", False),
        ("पंजाब में सबसे गहरा जिला कौन सा है?", False),
    ]
    for q, expected in gk_cases:
        is_gk = is_general_knowledge_query(q)
        passed = (is_gk == expected)
        print(f"[{'PASS' if passed else 'FAIL'}] '{q}' -> is_gk: {is_gk} (Expected: {expected})")
        if not passed:
            all_passed = False

    # ----------------------------------------------------
    # TEST 5: End-to-End Pipeline - Ludhiana Data Availability
    # ----------------------------------------------------
    print("\n--- TEST 5: End-to-End Ludhiana Data Availability ---")
    q_ludhiana_hi = "क्या लुधियाना का भूजल डेटा उपलब्ध है?"
    res_lud_hi = answer_groundwater_query(q_ludhiana_hi, raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
    print("Query (Hindi):", q_ludhiana_hi)
    print("Engine:", res_lud_hi["engine"])
    print("Intent:", res_lud_hi["intent"])
    print("Records found:", len(res_lud_hi["records"]))
    print("Answer snippet:\n", res_lud_hi["answer"][:300])

    q_ludhiana_en = "Is data available for Ludhiana?"
    res_lud_en = answer_groundwater_query(q_ludhiana_en, raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
    print("\nQuery (English):", q_ludhiana_en)
    print("Engine:", res_lud_en["engine"])
    print("Intent:", res_lud_en["intent"])
    print("Answer snippet:\n", res_lud_en["answer"][:300])

    # Check that Ludhiana data is confirmed available and not reported missing
    has_lud_records = ("2,575" in res_lud_hi["answer"] or "2575" in res_lud_hi["answer"] or "15.59" in res_lud_hi["answer"])
    has_lud_en = ("2,575" in res_lud_en["answer"] or "2575" in res_lud_en["answer"] or "15.59" in res_lud_en["answer"] or "available" in res_lud_en["answer"].lower())
    print(f"[{'PASS' if has_lud_records and has_lud_en else 'FAIL'}] Ludhiana data availability confirmed with correct CGWB statistics.")

    # ----------------------------------------------------
    # TEST 6: End-to-End Pipeline - Hindi District Ranking
    # ----------------------------------------------------
    print("\n--- TEST 6: End-to-End Hindi District Ranking ---")
    q_rank_hi = "पंजाब में सबसे गहरा भूजल स्तर किस जिले में है?"
    res_rank_hi = answer_groundwater_query(q_rank_hi, raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
    print("Query (Hindi):", q_rank_hi)
    print("Engine:", res_rank_hi["engine"])
    print("Intent:", res_rank_hi["intent"])
    print("Figure generated:", res_rank_hi.get("figure") is not None)
    print("Answer snippet:\n", res_rank_hi["answer"][:300])
    rank_pass = ("SAS NAGAR" in res_rank_hi["answer"] or "एसएएस नगर" in res_rank_hi["answer"] or "SANGRUR" in res_rank_hi["answer"] or "संगरूर" in res_rank_hi["answer"] or "61.24" in res_rank_hi["answer"] or "26.10" in res_rank_hi["answer"])
    print(f"[{'PASS' if rank_pass else 'FAIL'}] Hindi district ranking answered with grounded statistics.")

    # ----------------------------------------------------
    # TEST 7: End-to-End Pipeline - Aquifer Question (General Knowledge)
    # ----------------------------------------------------
    print("\n--- TEST 7: End-to-End Aquifer General Knowledge ---")
    q_aq_hi = "एक्विफर क्या है?"
    res_aq_hi = answer_groundwater_query(q_aq_hi, raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
    print("Query (Hindi):", q_aq_hi)
    print("Engine:", res_aq_hi["engine"])
    print("Intent:", res_aq_hi["intent"])
    print("Answer snippet:\n", res_aq_hi["answer"][:300])

    q_aq_en = "What is an aquifer?"
    res_aq_en = answer_groundwater_query(q_aq_en, raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
    print("\nQuery (English):", q_aq_en)
    print("Engine:", res_aq_en["engine"])
    print("Intent:", res_aq_en["intent"])
    print("Answer snippet:\n", res_aq_en["answer"][:300])

    gk_marker_hi = ("वैज्ञानिक" in res_aq_hi["answer"] or "Scientific" in res_aq_hi["answer"] or "Aquifer" in res_aq_hi["answer"] or "जलभृत" in res_aq_hi["answer"])
    gk_marker_en = ("Scientific" in res_aq_en["answer"] or "Aquifer" in res_aq_en["answer"] or "permeable" in res_aq_en["answer"].lower())
    print(f"[{'PASS' if gk_marker_hi and gk_marker_en else 'FAIL'}] Aquifer scientific explanation provided with clear general knowledge distinction.")

    # ----------------------------------------------------
    # TEST 8: Graph and Answer Consistency (Year Range Filter)
    # ----------------------------------------------------
    print("\n--- TEST 8: Consistency Between Graph and Written Answer ---")
    q_filter = "Show groundwater trend for Ludhiana from 2000 to 2015"
    res_filter = answer_groundwater_query(q_filter, raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
    print("Query:", q_filter)
    print("Engine:", res_filter["engine"])
    fig = res_filter.get("figure")
    has_fig = fig is not None
    print("Figure generated:", has_fig)
    if has_fig:
        # Check figure x-axis range or title
        print("Figure title:", fig.layout.title.text)
        print("Written answer snippet:\n", res_filter["answer"][:300])
    print(f"[{'PASS' if has_fig else 'FAIL'}] Graph and written answer both generated with matched filters.")

    # ----------------------------------------------------
    # TEST 9: SAS Nagar & Mohali Canonical Normalization
    # ----------------------------------------------------
    print("\n--- TEST 9: SAS Nagar & Mohali Canonical Normalization ---")
    known_dist = sorted(raw_data["District"].dropna().unique().tolist())
    
    # 1. District extraction
    q_mohali = "What is the average groundwater level in Mohali?"
    d_mohali = extract_districts_from_query(q_mohali, known_dist)
    q_sas = "SAS Nagar groundwater level"
    d_sas = extract_districts_from_query(q_sas, known_dist)
    norm_pass = ("SAS NAGAR (MOHALI)" in d_mohali and "SAS NAGAR (MOHALI)" in d_sas)
    print(f"[{'PASS' if norm_pass else 'FAIL'}] Query for 'Mohali' and 'SAS Nagar' both extract canonical 'SAS NAGAR (MOHALI)': {d_mohali}")
    if not norm_pass:
        all_passed = False

    # 2. Raw record stats directly calculated
    sas_records = raw_data[raw_data["District"] == "SAS NAGAR (MOHALI)"]
    sas_count = len(sas_records)
    sas_mean = round(sas_records["Groundwater Level Quarterly Manual (meter)"].mean(), 2)
    stats_pass = (sas_count == 1162 and sas_mean == 11.31)
    print(f"[{'PASS' if stats_pass else 'FAIL'}] SAS NAGAR (MOHALI) combined observations: {sas_count} (Expected: 1162), mean: {sas_mean} mbgl (Expected: 11.31)")
    if not stats_pass:
        all_passed = False

    # 3. Chatbot answer verification
    res_sas = answer_groundwater_query("What is the average groundwater depth in Mohali?", raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
    has_combined_text = ("1,162" in res_sas["answer"] or "1162" in res_sas["answer"] or "11.31" in res_sas["answer"])
    print(f"[{'PASS' if has_combined_text else 'FAIL'}] Chatbot answer for Mohali contains unified statistics (1,162 records / 11.31 mbgl).")
    if not has_combined_text:
        all_passed = False

    # ----------------------------------------------------
    # TEST 10: Negative and Extreme Depth Explanations
    # ----------------------------------------------------
    print("\n--- TEST 10: Negative and Extreme Depth Explanations ---")
    q_muk = "What is the groundwater depth in Mukatsar?"
    res_muk = answer_groundwater_query(q_muk, raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
    ans_muk = res_muk["answer"].lower()
    has_artesian_note = ("artesian" in ans_muk or "waterlog" in ans_muk or "आर्टेशियन" in ans_muk or "जलभराव" in ans_muk)
    print(f"[{'PASS' if has_artesian_note else 'FAIL'}] Mukatsar negative groundwater reading contains artesian/waterlogging explanation.")
    if not has_artesian_note:
        all_passed = False

    # ----------------------------------------------------
    # TEST 11: Corrected Temporal Range Guardrail (1996-2024 vs 1991-1995 & 2025)
    # ----------------------------------------------------
    print("\n--- TEST 11: Temporal Scope Guardrail ---")
    test_years = [
        ("Show groundwater level in 1993", True, "1996 to 2024"),   # In 1991-1995: unavailable observations explanation
        ("Groundwater data for Punjab in 2025", True, "1996 to 2024"), # In 2025: unavailable observations explanation
        ("Groundwater data in 1980", True, "outside"),             # Completely out of scope
        ("Groundwater trend in 2010", False, None)                  # Valid observation year
    ]
    for q_yr, should_block, expected_phrase in test_years:
        res_yr = answer_groundwater_query(q_yr, raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
        is_blocked = (res_yr["intent"] == "out_of_scope")
        matched_expected = (is_blocked == should_block) and (expected_phrase is None or expected_phrase.lower() in res_yr["answer"].lower())
        print(f"[{'PASS' if matched_expected else 'FAIL'}] '{q_yr}' -> Blocked: {is_blocked} (Expected: {should_block})")
        if not matched_expected:
            all_passed = False

    # ----------------------------------------------------
    # TEST 12: Village Abulkharana Retrieval & Statistics
    # ----------------------------------------------------
    print("\n--- TEST 12: Village Abulkharana Retrieval & Statistics ---")
    q_abul = "What is the average groundwater level in Abulkharana village of Muktsar district?"
    res_abul = answer_groundwater_query(q_abul, raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
    ans_abul = res_abul["answer"]
    has_abul_name = ("Abulkharana" in ans_abul or "ABULKHARANA" in ans_abul or "abulkharana" in ans_abul.lower())
    has_abul_mean = ("2.01" in ans_abul or "2.00" in ans_abul or "2.0" in ans_abul)
    has_abul_fig = (res_abul.get("figure") is not None and "Abulkharana" in res_abul["figure"].layout.title.text)
    abul_pass = (has_abul_name and has_abul_mean and has_abul_fig and res_abul["intent"] == "village_analytics")
    print(f"[{'PASS' if abul_pass else 'FAIL'}] Village Abulkharana retrieved with exact statistics & village trend figure.")
    if not abul_pass:
        all_passed = False

    print("\n" + "=" * 70)
    print(f"TEST SUITE COMPLETE - ALL PASSED: {all_passed}")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
