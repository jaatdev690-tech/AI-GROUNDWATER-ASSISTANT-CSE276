import os
import re
import difflib
from pathlib import Path
import pandas as pd
import torch
import plotly.express as px
from sentence_transformers import SentenceTransformer, util

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
LEVEL_COL = "Groundwater Level Quarterly Manual (meter)"
DATE_COL = "Data Acquisition Time"
EMBEDDINGS_CACHE = DATA_DIR / "groundwater_embeddings.pt"
CORPUS_CACHE = DATA_DIR / "searchable_corpus.csv"

# Models supported by Groq, ordered by preference
PREFERRED_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "allam-2-7b"
]

# Non-Punjab locations to detect (English & Hindi)
KNOWN_OTHER_REGIONS = [
    # English
    "DELHI", "MUMBAI", "JAIPUR", "RAJASTHAN", "HARYANA", "UP", "UTTAR PRADESH",
    "BIHAR", "PATNA", "BENGALURU", "BANGALORE", "CHENNAI", "KOLKATA", "KERALA",
    "GUJARAT", "AHMEDABAD", "PUNE", "MAHARASHTRA", "HYDERABAD", "BHOPAL", "MP",
    "GOA", "HIMACHAL", "SHIMLA", "UTTARAKHAND", "DEHRADUN", "KASHMIR", "SRINAGAR",
    # Hindi
    "दिल्ली", "मुंबई", "जयपुर", "राजस्थान", "हरियाणा", "उत्तर प्रदेश", "यूपी",
    "बिहार", "पटना", "बेंगलुरु", "बैंगलोर", "चेन्नई", "कोलकाता", "केरल",
    "गुजरात", "अहमदाबाद", "पुणे", "महाराष्ट्र", "हैदराबाद", "भोपाल", "मध्य प्रदेश",
    "गोवा", "हिमाचल", "शिमला", "उत्तराखंड", "देहरादून", "कश्मीर", "श्रीनगर"
]

# Multilingual water domain terms
WATER_DOMAIN_TERMS = [
    # English
    "groundwater", "ground water", "water", "aquifer", "water table", "water level",
    "depth", "level", "mbgl", "well", "tubewell", "tube well", "borewell", "cgwb",
    "recharge", "deplet", "hydrolog", "salin", "extraction", "monsoon", "piezometer",
    "monitoring", "meter", "station", "measurement", "reading", "village", "villages",
    "data", "availability", "available", "record", "records", "status", "report", "trend",
    "ranking", "rank", "conservation",
    # Water quality and chemical hydrogeology terms
    "quality", "water quality", "groundwater quality", "nitrate", "fluoride", "arsenic",
    "salinity", "tds", "ph", "hardness", "uranium", "chloride", "heavy metal", "heavy metals",
    "contamination", "contaminant", "pollut", "pollution", "potable", "drinking", "concentration",
    "mg/l", "ppm", "chemical", "parameter", "parameters",
    # Hindi
    "भूजल", "जल", "पानी", "जलस्तर", "जल स्तर", "वाटर टेबल", "जलभृत", "एक्विफर",
    "गहराई", "स्तर", "कुआं", "कुएं", "ट्यूबवेल", "बोरवेल", "पुनर्भरण", "रिचार्ज",
    "दोहन", "कमी", "कटौती", "मानसून", "निगरानी", "माप", "रीडिंग", "गांव", "गाँव",
    "पिंड", "जिला", "जिले", "डेटा", "आंकड़े", "रिकॉर्ड", "उपलब्ध", "उपलब्धता",
    "रुझान", "ट्रेंड", "रैंकिंग", "रैंक", "संरक्षण", "बचाव",
    "गुणवत्ता", "जल गुणवत्ता", "पानी की गुणवत्ता", "नाइट्रेट", "फ्लोराइड", "आर्सेनिक", "लवणता",
    "टीडीएस", "पीएच", "यूरेनियम", "क्लोराइड", "प्रदूषण", "संदूषण", "रसायन", "पीने योग्य", "सांद्रता",
    # Hinglish
    "bhujal", "bhu-jal", "pani", "paani", "jal", "jalstar", "jal star", "gehrai",
    "star", "kuan", "tubewell", "borewell", "punarbharan", "dohan", "nigraani",
    "jila", "zila", "zile", "gaon", "pind", "aankde", "rujhan", "sanrakshan",
    "uplabdh", "upalabdh", "gunvatta", "jal gunvatta", "quality", "nitrate", "fluoride",
    "arsenic", "pradushan", "chemical"
]

# Multilingual general hydrological concepts
GENERAL_KNOWLEDGE_TERMS = [
    # English
    "groundwater", "ground water", "aquifer", "water table", "water-table",
    "recharge", "discharge", "permeability", "porosity", "artesian", "mbgl",
    "piezometer", "unconfined", "confined", "hydrogeology", "hydrology",
    "salinity", "over-exploitation", "borewell", "tubewell", "tube well",
    "drawdown", "cone of depression", "cgwb", "conservation", "water conservation",
    "rainwater harvesting", "depletion", "water quality", "groundwater quality",
    "nitrate", "fluoride", "arsenic", "depth vs quality", "quality vs depth",
    # Hindi
    "भूजल", "भूमिगत जल", "जलभृत", "एक्विफर", "जल स्तर", "जलस्तर", "वाटर टेबल",
    "पुनर्भरण", "रिचार्ज", "पारगम्यता", "सरंध्रता", "आर्टेशियन", "हाइड्रोजियोलॉजी",
    "भूजल विज्ञान", "लवणता", "संरक्षण", "जल संरक्षण", "वर्षा जल संचयन", "दोहन",
    "जल गुणवत्ता", "पानी की गुणवत्ता", "नाइट्रेट", "फ्लोराइड", "आर्सेनिक",
    # Hinglish
    "bhujal", "jalbhrut", "punarbharan", "sanrakshan", "water conservation", "rainwater",
    "water quality", "jal gunvatta"
]

GENERAL_QUESTION_TRIGGERS = [
    # English triggers
    r"\b(what is|what are|define|definition|explain|meaning of|how does|why is|why does|how do|importance of|causes of|concept of|types of|difference between|how to calculate|how is|how to save|how to conserve|methods of|measures for)\b",
    # Hindi triggers
    r"(क्या है|क्या होता है|किसे कहते हैं|परिभाषा|अर्थ क्या है|व्याख्या|समझाइए|समझाएं|क्यों होता है|कैसे होता है|महत्व क्या है|प्रकार|भेद|कैसे करें|संरक्षण कैसे|उपाय|कैसे बचाएं)",
    # Hinglish triggers
    r"\b(kya hai|kya hota hai|kise kehte|arth kya|samjhao|batao|kaise hota|kaise kare|kaise karein|upay|conservation kaise|kaise bachaye)\b"
]

CANONICAL_DISTRICT_MAP = {
    "MOHALI": "SAS NAGAR (MOHALI)",
    "SAS NAGAR": "SAS NAGAR (MOHALI)",
    "SAS NAGAR (MOHALI)": "SAS NAGAR (MOHALI)"
}

PUNJAB_DISTRICTS = [
    "AMRITSAR", "BARNALA", "BATHINDA", "FARIDKOT", "FATEHGARH SAHIB", "FAZILKA",
    "FEROZEPUR", "GURDASPUR", "HOSHIARPUR", "JALANDHAR", "KAPURTHALA", "LUDHIANA",
    "MANSA", "MOGA", "MUKATSAR", "PATHANKOT", "PATIALA", "ROPAR",
    "SANGRUR", "SAS NAGAR (MOHALI)", "SBS NAGAR", "TARN TARAN"
]

# Comprehensive synonyms, Hindi script, and transliterations for all 23 Punjab districts
DISTRICT_SYNONYMS = {
    "AMRITSAR": ["amritsar", "ambarsar", "अमृतसर", "ਅੰਮ੍ਰਿਤਸਰ"],
    "BARNALA": ["barnala", "बरनाला", "ਬਰਨਾਲਾ"],
    "BATHINDA": ["bathinda", "bhatinda", "बठिंडा", "भटिंडा", "ਬਠਿੰਡਾ"],
    "FARIDKOT": ["faridkot", "फरीदकोट", "ਫਰੀਦਕੋਟ"],
    "FATEHGARH SAHIB": ["fatehgarh sahib", "fatehgarh", "फतेहगढ़ साहिब", "फतेहगढ़", "ਫਤਿਹਗੜ੍ਹ ਸਾਹਿਬ"],
    "FAZILKA": ["fazilka", "फाजिल्का", "ਫਾਜ਼ਿਲਕਾ"],
    "FEROZEPUR": ["ferozepur", "firozpur", "firozepur", "फिरोजपुर", "ਫ਼ਿਰੋਜ਼ਪੁਰ"],
    "GURDASPUR": ["gurdaspur", "गुरदासपुर", "ਗੁਰਦਾਸਪੁਰ"],
    "HOSHIARPUR": ["hoshiarpur", "होशियारपुर", "ਹੁਸ਼ਿਆਰਪੁਰ"],
    "JALANDHAR": ["jalandhar", "jullundur", "जालंधर", "ਜਲੰਧਰ"],
    "KAPURTHALA": ["kapurthala", "कपूरथला", "ਕਪੂਰਥਲਾ"],
    "LUDHIANA": ["ludhiana", "ludhiyana", "ludhiane", "लुधियाना", "लुधियाने", "ਲੁਧਿਆਣਾ"],
    "MANSA": ["mansa", "मानसा", "ਮਾਨਸਾ"],
    "MOGA": ["moga", "मोगा", "ਮੋਗਾ"],
    "SAS NAGAR (MOHALI)": [
        "sas nagar (mohali)", "sas nagar", "s.a.s. nagar", "s.a.s nagar",
        "sahibzada ajit singh nagar", "mohali", "मोहाली", "एसएएस नगर", "ਮੋਹਾਲੀ",
        "sas nagar mohali", "mohali district", "sas nagar district"
    ],
    "MOHALI": ["mohali", "sas nagar", "s.a.s. nagar", "s.a.s nagar", "sahibzada ajit singh nagar", "मोहाली", "एसएएस नगर", "ਮੋਹਾਲੀ"],
    "MUKATSAR": ["mukatsar", "muktsar", "sri muktsar sahib", "मुक्तसर", "ਮੁਕਤਸਰ"],
    "PATHANKOT": ["pathankot", "पठानकोट", "ਪਠਾਨਕੋਟ"],
    "PATIALA": ["patiala", "पटियाला", "ਪਟਿਆਲਾ"],
    "ROPAR": ["ropar", "rupnagar", "roopnagar", "रोपड़", "रूपनगर", "ਰੋਪੜ"],
    "SANGRUR": ["sangrur", "संगरूर", "ਸੰਗਰੂਰ"],
    "SAS NAGAR": ["sas nagar", "s.a.s. nagar", "s.a.s nagar", "sahibzada ajit singh nagar", "mohali", "एसएएस नगर", "मोहाली"],
    "SBS NAGAR": ["sbs nagar", "s.b.s. nagar", "s.b.s nagar", "nawanshahr", "nawanshahar", "shaheed bhagat singh nagar", "एसबीएस नगर", "नवांशहर"],
    "TARN TARAN": ["tarn taran", "tarntaran", "तरन तारन", "ਤਰਨ ਤਾਰਨ"]
}

# Precompute a set of all district aliases (lowercased) to prevent village match collision
ALL_DISTRICT_ALIASES = set()
for canonical, syns in DISTRICT_SYNONYMS.items():
    ALL_DISTRICT_ALIASES.add(canonical.lower())
    for s in syns:
        ALL_DISTRICT_ALIASES.add(s.lower())
        for w in s.lower().split():
            ALL_DISTRICT_ALIASES.add(w)

STOPWORDS_SET = {
    "what", "is", "the", "groundwater", "ground", "water", "level", "levels", "depth",
    "trend", "trends", "graph", "graphs", "chart", "charts", "plot", "plots",
    "in", "at", "for", "of", "and", "or", "to", "from", "between", "with",
    "average", "mean", "min", "max", "minimum", "maximum", "shallowest", "deepest",
    "measurement", "measurements", "reading", "readings", "data", "records", "record",
    "punjab", "cgwb", "station", "stations", "monitoring", "show", "tell", "me", "about",
    "give", "details", "status", "report", "village", "villages", "tehsil", "block", "district",
    # Hindi / Hinglish stopwords
    "का", "की", "के", "में", "से", "पर", "को", "क्या", "है", "हैं", "बताओ", "दिखाओ", "कितना",
    "kya", "hai", "hain", "me", "mein", "ka", "ki", "ke", "se", "batao", "dikhao", "kitna"
}

_VERIFICATION_CACHE = {}
_VILLAGE_META_CACHE = {}


# ==============================================================================
# LANGUAGE DETECTION & MULTILINGUAL UTILITIES
# ==============================================================================

def detect_language(text):
    """
    Detect whether text is Hindi (Devanagari), Hinglish (Romanized Hindi), or English.
    Supports explicit user instructions like 'Explain in Hindi' or 'Mujhe Hindi mein samjhao'.
    Returns: 'hi', 'hinglish', or 'en'
    """
    if not text:
        return "en"

    t_low = text.lower()

    # 1. Check for explicit user language directives
    if re.search(r"\b(in hindi|hindi mein|hindi me|हिंदी में|हिन्दी में|explain.*in hindi|samjhao.*hindi|hindi bhasha)\b", t_low):
        return "hi"
    if re.search(r"\b(in hinglish|hinglish mein|hinglish me|hinglish bhasha)\b", t_low):
        return "hinglish"
    if re.search(r"\b(in english|english mein|english me|अंग्रेजी में|अंग्रेज़ी में)\b", t_low):
        return "en"

    # 2. Check for Devanagari Unicode characters (U+0900 to U+097F)
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    # 3. Check for Hinglish vocabulary markers in Roman script
    t_low = text.lower()
    hinglish_markers = {
        "kya", "kyu", "kyun", "kaun", "kaunsa", "kaunsi", "kitna", "kitni", "kitne",
        "hai", "hain", "kaha", "kahan", "kab", "kisko", "kis", "kise", "kaise", "kaisa",
        "me", "mein", "se", "ko", "ka", "ki", "ke", "par", "pe",
        "batao", "dikhao", "bataiye", "dikhaye", "samjhao", "chahiye", "hoga", "hogi",
        "paani", "pani", "jal", "bhujal", "bhu-jal", "gehra", "gehri", "gehrai",
        "sabse", "jila", "zila", "zile", "gaon", "pind", "karan", "vajah", "rujhan",
        "aankde", "peene", "kheti", "fasal", "upalabdh", "uplabdh"
    }
    tokens = re.findall(r"[a-z]+", t_low)
    match_count = sum(1 for w in tokens if w in hinglish_markers)
    distinctive_markers = {"kya", "kitna", "kitni", "kitne", "kaun", "kaunsa", "kaunsi", "bhujal", "gehrai", "batao", "dikhao", "sabse", "uplabdh"}
    has_distinctive = any(w in distinctive_markers for w in tokens)

    if match_count >= 2 or (match_count >= 1 and has_distinctive):
        return "hinglish"

    return "en"


def extract_year_range_from_query(query):
    """
    Extract specific year or year-range constraint from query.
    Examples:
    - 'from 2000 to 2015' -> (2000, 2015)
    - 'between 1995 and 2010' -> (1995, 2010)
    - '2000 se 2015' / '2000 से 2015' -> (2000, 2015)
    - 'after 2010' -> (2010, 2025)
    - 'before 2005' -> (1991, 2005)
    - 'in 2012' -> (2012, 2012)
    Returns: (start_year: int or None, end_year: int or None)
    """
    q_clean = query.lower()

    # 1. Range: '1995 to 2015', '1995-2015', '1995 se 2015', '1995 से 2015'
    range_match = re.search(r"\b(19\d\d|20\d\d)\s*(?:to|and|-|se|तक|से)\s*(19\d\d|20\d\d)\b", q_clean)
    if range_match:
        y1, y2 = int(range_match.group(1)), int(range_match.group(2))
        return (min(y1, y2), max(y1, y2))

    # 2. After / Since / Post / के बाद
    after_match = re.search(r"\b(?:after|since|post|के बाद|से बाद)\s*(19\d\d|20\d\d)\b", q_clean)
    if after_match:
        return (int(after_match.group(1)), 2024)

    # 3. Before / Prior / के पहले
    before_match = re.search(r"\b(?:before|prior to|के पहले|से पहले)\s*(19\d\d|20\d\d)\b", q_clean)
    if before_match:
        return (1996, int(before_match.group(1)))

    # 4. Single year: 'in 2012', 'for 2012', 'साल 2012', 'वर्ष 2012'
    single_match = re.search(r"\b(?:in|for|year|साल|वर्ष|में)\s*(19\d\d|20\d\d)\b", q_clean)
    if single_match:
        y = int(single_match.group(1))
        return (y, y)

    return (None, None)


# ==============================================================================
# GROQ API VERIFICATION
# ==============================================================================

def verify_groq_api(api_key):
    """
    Verify whether the provided Groq API key is valid by making a lightweight test call.
    Returns: (is_valid: bool, active_model: str or None, error_message: str or None)
    """
    if not api_key:
        return False, None, "No API key provided"

    if not GROQ_AVAILABLE:
        return False, None, "The 'groq' Python package is not installed."

    if api_key in _VERIFICATION_CACHE:
        return _VERIFICATION_CACHE[api_key]

    try:
        client = Groq(api_key=api_key)
        models_data = client.models.list().data
        available_ids = [m.id for m in models_data]

        active_model = next((m for m in PREFERRED_GROQ_MODELS if m in available_ids), None)
        if not active_model and available_ids:
            active_model = available_ids[0]

        if not active_model:
            res = (False, None, "No chat models available on this Groq account.")
        else:
            res = (True, active_model, None)

    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Invalid API Key" in err_msg:
            clean_err = "Invalid Groq API key. Please verify your key at console.groq.com."
        elif "connection" in err_msg.lower():
            clean_err = "Network connection to Groq API failed."
        else:
            clean_err = err_msg.split("\n")[0]
        res = (False, None, clean_err)

    _VERIFICATION_CACHE[api_key] = res
    return res


# ==============================================================================
# DATA LOADING & INDEXING
# ==============================================================================

def load_groundwater_data():
    """Load and clean raw groundwater quarterly CSV records with date parsing."""
    files = list(DATA_DIR.glob("*.csv"))
    files = [f for f in files if "searchable_corpus" not in f.name]

    if not files:
        raise FileNotFoundError("No raw CSV files found in Data folder.")

    frames = []
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        df["Source File"] = file.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data.columns = data.columns.str.strip()

    required = ["District", LEVEL_COL]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["District_Original"] = data["District"].astype(str).str.strip().str.upper()
    data["District"] = data["District_Original"].map(lambda d: CANONICAL_DISTRICT_MAP.get(d, d))
    data[LEVEL_COL] = pd.to_numeric(data[LEVEL_COL], errors="coerce")

    # Parse acquisition date and extract observation year
    if DATE_COL in data.columns:
        data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce", dayfirst=True)
        data["Year"] = data[DATE_COL].dt.year

    # Clean administrative fields
    for col in ["Village", "Tehsil", "Block", "Station"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    data = data.dropna(subset=["District", LEVEL_COL]).copy()
    data = data[data["District"] != ""].copy()
    data.reset_index(drop=True, inplace=True)
    return data


def load_summary_data():
    """Load precomputed district and yearly summary CSVs."""
    district_file = BASE_DIR / "district_groundwater_summary.csv"
    yearly_file = BASE_DIR / "yearly_groundwater_summary.csv"

    district_summary = None
    yearly_summary = None

    if district_file.exists():
        district_summary = pd.read_csv(district_file)
        if "District" in district_summary.columns:
            district_summary["District_Original"] = district_summary["District"].astype(str).str.upper()
            district_summary["District"] = district_summary["District_Original"].map(lambda d: CANONICAL_DISTRICT_MAP.get(d, d))

    if yearly_file.exists():
        yearly_summary = pd.read_csv(yearly_file)
        if "District" in yearly_summary.columns:
            yearly_summary["District"] = yearly_summary["District"].astype(str).str.upper()

    return district_summary, yearly_summary


def build_searchable_corpus(raw_data):
    """Build a high-signal station profile corpus."""
    if CORPUS_CACHE.exists():
        corpus = pd.read_csv(CORPUS_CACHE)
        if "District" in corpus.columns:
            if "District_Original" not in corpus.columns:
                corpus["District_Original"] = corpus["District"].astype(str).str.upper()
            corpus["District"] = corpus["District_Original"].map(lambda d: CANONICAL_DISTRICT_MAP.get(d, d))
        return corpus

    group_cols = ["District"]
    for col in ["Station", "Tehsil", "Block", "Village"]:
        if col in raw_data.columns:
            group_cols.append(col)

    station_agg = raw_data.groupby(group_cols, as_index=False).agg(
        Mean_Depth=(LEVEL_COL, "mean"),
        Min_Depth=(LEVEL_COL, "min"),
        Max_Depth=(LEVEL_COL, "max"),
        Total_Records=(LEVEL_COL, "count"),
    )
    station_agg["Mean_Depth"] = station_agg["Mean_Depth"].round(2)
    station_agg["Source File"] = "CGWB Station Monitoring Aggregate"

    def format_search_chunk(row):
        parts = [f"District: {row['District']}"]
        for col in ["Station", "Tehsil", "Block", "Village"]:
            if col in row.index and pd.notna(row[col]) and str(row[col]).strip() not in ["", "-", "nan"]:
                parts.append(f"{col}: {row[col]}")
        parts.append(f"Average Depth: {row['Mean_Depth']} mbgl")
        parts.append(f"Range: {row['Min_Depth']}m to {row['Max_Depth']}m")
        parts.append(f"Total Observations: {row['Total_Records']}")
        return " | ".join(parts)

    station_agg["search_text"] = station_agg.apply(format_search_chunk, axis=1)
    station_agg.to_csv(CORPUS_CACHE, index=False)
    return station_agg


def build_search_index():
    """Build and cache sentence-transformer embeddings."""
    raw_data = load_groundwater_data()
    corpus = build_searchable_corpus(raw_data)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if EMBEDDINGS_CACHE.exists():
        embeddings = torch.load(EMBEDDINGS_CACHE, weights_only=True)
    else:
        embeddings = model.encode(
            corpus["search_text"].tolist(),
            batch_size=128,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        torch.save(embeddings, EMBEDDINGS_CACHE)

    return raw_data, corpus, model, embeddings


def search_groundwater(question, corpus, model, embeddings, top_k=6):
    """Retrieve top-K semantic matches with similarity scores."""
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, embeddings)[0]
    k = min(top_k, len(corpus))
    best_indices = scores.topk(k=k).indices.cpu().tolist()

    results = corpus.iloc[best_indices].copy()
    results["Similarity"] = [float(scores[i].cpu()) for i in best_indices]
    return results


# ==============================================================================
# INTENT & GUARDRAILS LOGIC
# ==============================================================================

def extract_districts_from_query(query, known_districts):
    """
    Extract all Punjab district names mentioned in the query across English,
    Hindi, Hinglish, and Punjabi variations.
    """
    q_low = query.lower()
    found = []

    # 1. Match via comprehensive synonyms mapping
    for canonical, syn_list in DISTRICT_SYNONYMS.items():
        if canonical not in known_districts:
            continue
        for syn in sorted(syn_list, key=len, reverse=True):
            syn_low = syn.lower()
            # If Devanagari or Gurmukhi script
            if re.search(r"[\u0900-\u097F\u0A00-\u0A7F]", syn_low):
                if syn_low in q_low:
                    if canonical not in found:
                        found.append(canonical)
                    break
            else:
                pattern = r"\b" + re.escape(syn_low) + r"\b"
                if re.search(pattern, q_low):
                    if canonical not in found:
                        found.append(canonical)
                    break

    # 2. Direct match against known_districts
    for d in known_districts:
        pattern = r"\b" + re.escape(d.lower()) + r"\b"
        if re.search(pattern, q_low) and d not in found:
            found.append(d)

    return found


def is_general_knowledge_query(query):
    """
    Detect if the user is asking a conceptual or scientific definition question
    (e.g., about aquifers, water tables, recharge, conservation)
    rather than an empirical lookup from the Punjab dataset.
    Works for English, Hindi, and Hinglish.
    """
    q_low = query.lower().strip()

    # If query mentions a specific Punjab district, it is dataset-specific
    for d in PUNJAB_DISTRICTS:
        if re.search(r"\b" + re.escape(d.lower()) + r"\b", q_low):
            return False

    for syn_list in DISTRICT_SYNONYMS.values():
        for syn in syn_list:
            if re.search(r"[\u0900-\u097F]", syn):
                if syn in q_low:
                    return False
            elif re.search(r"\b" + re.escape(syn.lower()) + r"\b", q_low):
                return False

    if "punjab" in q_low or "पंजाब" in q_low:
        return False

    # Check if asking for dataset-specific numerical metrics or tables
    dataset_terms = [
        "record", "records", "dataset", "data", "csv", "ranking", "rank",
        "trend", "highest", "lowest", "maximum", "minimum", "station", "village",
        "depth", "measurement", "reading", "level in", "depth in", "water in",
        "डेटा", "आंकड़े", "रिकॉर्ड", "रैंकिंग", "रैंक", "रुझान", "अधिकतम", "न्यूनतम"
    ]
    if any(re.search(r"\b" + re.escape(w) + r"\b", q_low) if not re.search(r"[\u0900-\u097F]", w) else w in q_low for w in dataset_terms):
        return False

    has_concept = any(
        t in q_low if re.search(r"[\u0900-\u097F]", t) else re.search(r"\b" + re.escape(t) + r"\b", q_low)
        for t in GENERAL_KNOWLEDGE_TERMS
    )
    has_trigger = any(re.search(pat, q_low) for pat in GENERAL_QUESTION_TRIGGERS)

    return has_concept and (has_trigger or len(q_low.split()) <= 4)


def is_groundwater_related(query, max_similarity=0.0):
    """
    Strictly verify whether the query pertains to groundwater, water table,
    general hydrological concepts, or Punjab hydrogeological monitoring.
    Supports English, Hindi, and Hinglish queries.
    """
    q_low = query.lower()

    # Explicit non-groundwater triggers (multilingual)
    unrelated_keywords = [
        # English
        "pizza", "recipe", "cook", "food", "movie", "song", "cricket", "football",
        "president", "prime minister", "capital of", "actor", "game", "code", "python",
        "programming", "java", "joke", "crypto", "stock price", "car", "phone",
        # Hindi
        "पिज़्ज़ा", "पिज्जा", "रेसिपी", "खाना", "मूवी", "गाना", "क्रिकेट", "फुटबॉल",
        "राष्ट्रपति", "प्रधानमंत्री", "राजधानी", "एक्टर", "फिल्म", "मजाक", "गाड़ी", "फोन"
    ]
    for w in unrelated_keywords:
        match_unrelated = (w in q_low) if re.search(r"[\u0900-\u097F]", w) else bool(re.search(r"\b" + re.escape(w) + r"\b", q_low))
        if match_unrelated:
            # Override if legitimate water term explicitly present
            if not any(term in q_low for term in ["groundwater", "water table", "aquifer", "भूजल", "जलस्तर", "जलभृत", "वाटर टेबल"]):
                return False

    # Check for presence of water terms in English, Hindi, Hinglish
    has_water_term = any(
        w in q_low if re.search(r"[\u0900-\u097F]", w) else bool(re.search(r"\b" + re.escape(w) + r"\b", q_low))
        for w in WATER_DOMAIN_TERMS
    )

    # Check for Punjab districts mention (automatically relevant)
    mentioned_districts = extract_districts_from_query(query, PUNJAB_DISTRICTS)
    if mentioned_districts:
        return True

    has_punjab = ("punjab" in q_low) or ("पंजाब" in q_low)

    if has_water_term:
        return True
    if has_punjab and max_similarity >= 0.22:
        return True
    if max_similarity >= 0.35:
        return True

    return False


def detect_query_intent(query):
    """Detect whether query is analytical or station lookup (multilingual)."""
    q = query.lower()
    analytics_patterns = [
        # English patterns
        r"\b(highest|maximum|max|deepest|most depleted|worst)\b",
        r"\b(lowest|minimum|min|shallowest|least depleted|best)\b",
        r"\b(average|mean|overall)\b",
        r"\b(ranking|rank|top \d+|bottom \d+|compare|comparison)\b",
        r"\b(trend|over the years|change|depletion|increase|decrease)\b",
        r"\b(graph|chart|plot|curve|visualize)\b",
        r"\b(summary|overview|statistics|total records|availability|available)\b",
        # Hindi patterns
        r"(अधिकतम|सबसे गहरा|गहरा|सबसे ज्यादा|उच्चतम)",
        r"(न्यूनतम|सबसे कम|कम गहराई|उथला|सबसे अच्छा)",
        r"(औसत|कुल)",
        r"(रैंकिंग|रैंक|तुलना|स्थान)",
        r"(रुझान|ट्रेंड|बदलाव|सालाना|वर्षवार|गिरावट)",
        r"(ग्राफ|चार्ट|चित्र)",
        r"(उपलब्ध|उपलब्धता|डेटा|विवरण|आंकड़े)",
        # Hinglish patterns
        r"\b(sabse gehra|sabse jyada|sabse kam|ausat|rujhan|tulna|uplabdh|graph|chart)\b"
    ]
    for pattern in analytics_patterns:
        if re.search(pattern, q):
            return "analytics"
    return "search"


def check_out_of_scope(query, known_districts, lang="en"):
    """Check if query asks about locations outside Punjab or invalid years."""
    q_upper = query.upper()
    q_low = query.lower()

    for reg in KNOWN_OTHER_REGIONS:
        match = (reg in q_low) if re.search(r"[\u0900-\u097F]", reg) else bool(re.search(r"\b" + re.escape(reg) + r"\b", q_upper))
        if match:
            reg_display = reg.title() if not re.search(r"[\u0900-\u097F]", reg) else reg
            if lang == "hi":
                msg = (
                    f"मैं **पंजाब के लिए AI भूजल सूचना सहायक** हूँ। "
                    f"आपने **{reg_display}** के बारे में पूछा है, लेकिन उपलब्ध केंद्रीय भूजल बोर्ड (CGWB) "
                    f"निगरानी डेटासेट केवल **पंजाब राज्य (इसके 22 जिलों)** तक ही सीमित है (सत्यापित अवलोकन: 1996–2024)।\n\n"
                    f"पंजाब के बाहर के क्षेत्रों का डेटा इस प्रणाली में उपलब्ध नहीं है। कृपया पंजाब के किसी जिले या गांव के बारे में पूछें।"
                )
            elif lang == "hinglish":
                msg = (
                    f"Main **Punjab ke liye AI Groundwater Assistant** hoon. "
                    f"Aapne **{reg_display}** ke baare mein poocha hai, lekin Central Ground Water Board (CGWB) "
                    f"ka dataset sirf **Punjab ke 22 districts** ke liye hi available hai (1996–2024 observations).\n\n"
                    f"Punjab ke bahar ke locations ka data is system mein shamil nahi hai."
                )
            else:
                msg = (
                    f"I am the **AI Groundwater Information Assistant for Punjab**. "
                    f"You asked about **{reg_display}**, but the available Central Ground Water Board (CGWB) "
                    f"monitoring dataset is strictly confined to the state of **Punjab, India** (covering its 22 canonical districts, with verified observations from 1996 to 2024).\n\n"
                    f"Data for locations outside Punjab is not included in this system."
                )
            return True, msg

    years = re.findall(r"\b(18\d\d|19\d\d|20\d\d)\b", query)
    for y_str in years:
        y = int(y_str)
        # Check if completely outside CGWB series bounds
        if y < 1991 or y > 2025:
            if lang == "hi":
                msg = (
                    f"केंद्रीय भूजल बोर्ड (CGWB) पंजाब डेटासेट श्रृंखला 1991–2025 से संबंधित है "
                    f"(जिसमें वास्तविक सत्यापित अवलोकन **1996 से 2024** तक उपलब्ध हैं)।\n\n"
                    f"वर्ष **{y}** इस डेटासेट के दायरे से पूरी तरह बाहर है। "
                    f"कृपया 1996–2024 के सक्रिय अवलोकन वर्षों के बारे में प्रश्न पूछें।"
                )
            elif lang == "hinglish":
                msg = (
                    f"CGWB Punjab dataset 1991–2025 series ka hai (actual verified observation records "
                    f"**1996 se 2024** tak hain).\n\n"
                    f"Year **{y}** is dataset ke scope se bahar hai. Kripya 1996–2024 ke beech query karein."
                )
            else:
                msg = (
                    f"The Central Ground Water Board (CGWB) Punjab monitoring datasets in this project "
                    f"are cataloged under the 1991–2025 series (with verified monitoring observations spanning **1996 to 2024**).\n\n"
                    f"Year **{y}** is outside the dataset coverage. Please query within the active monitoring range of **1996 to 2024**."
                )
            return True, msg

        # Check for unavailable observation years within nominal series (1991–1995 and 2025)
        if (1991 <= y <= 1995) or y == 2025:
            if lang == "hi":
                msg = (
                    f"यद्यपि आधिकारिक केंद्रीय भूजल बोर्ड (CGWB) डेटासेट संग्रह का शीर्षक 1991–2025 है, "
                    f"लेकिन पंजाब में वास्तविक सत्यापित क्षेत्रीय निगरानी अवलोकन **1996 से 2024** तक ही उपलब्ध हैं।\n\n"
                    f"वर्ष **{y}** के लिए डेटासेट में कोई त्रैमासिक निगरानी रिकॉर्ड मौजूद नहीं है "
                    f"(1991–1995 और 2025 के आंकड़े जारी डेटासेट में अनुपस्थित हैं)। "
                    f"कृपया **1996 से 2024** के सक्रिय निगरानी वर्षों के बारे में प्रश्न पूछें।"
                )
            elif lang == "hinglish":
                msg = (
                    f"Official CGWB dataset collection ka title 1991–2025 hai, lekin Punjab mein "
                    f"actual verified field observations **1996 se 2024** tak hi available hain.\n\n"
                    f"Year **{y}** ke liye CGWB dataset mein koi observations nahi hain "
                    f"(1991–1995 aur 2025 ke records absent hain). "
                    f"Kripya **1996 se 2024** ke active window mein query karein."
                )
            else:
                msg = (
                    f"While the official Central Ground Water Board (CGWB) dataset archive is titled 1991–2025, "
                    f"actual validated field monitoring observations in Punjab are strictly available from **1996 to 2024**.\n\n"
                    f"No monitoring observations exist in the CGWB quarterly dataset for the year **{y}** "
                    f"(records for 1991–1995 and 2025 are absent from the released data). "
                    f"Please query within the active observation window of **1996 to 2024**."
                )
            return True, msg

    return False, None


# ==============================================================================
# VILLAGE-LEVEL DATA MANAGEMENT & RETRIEVAL
# ==============================================================================

def get_village_metadata(raw_df):
    """
    Extract, clean, and cache unique village metadata from raw observations.
    Returns: (village_meta: dict, sorted_villages: list, valid_v_df: DataFrame)
    """
    global _VILLAGE_META_CACHE
    df_id = id(raw_df)
    if df_id in _VILLAGE_META_CACHE:
        return _VILLAGE_META_CACHE[df_id]

    if "Village" not in raw_df.columns:
        return {}, [], pd.DataFrame()

    valid_v = raw_df.dropna(subset=["District", "Village", LEVEL_COL]).copy()
    valid_v["Village_Clean"] = valid_v["Village"].astype(str).str.strip()
    valid_v = valid_v[~valid_v["Village_Clean"].isin(["", "-", "nan", "None", "NAN"])].copy()
    valid_v["Village_Upper"] = valid_v["Village_Clean"].str.upper()

    meta = {}
    for v_upper, grp in valid_v.groupby("Village_Upper"):
        tehsils = [t for t in grp["Tehsil"].dropna().unique().tolist() if t not in ["-", "", "nan"]]
        blocks = [b for b in grp["Block"].dropna().unique().tolist() if b not in ["-", "", "nan"]]
        stations = [s for s in grp["Station"].dropna().unique().tolist() if s not in ["-", "", "nan"]]
        districts = [d for d in grp["District"].dropna().unique().tolist() if d not in ["-", "", "nan"]]

        meta[v_upper] = {
            "display_name": grp["Village_Clean"].iloc[0],
            "districts": sorted(districts),
            "tehsils": sorted(tehsils),
            "blocks": sorted(blocks),
            "stations": sorted(stations),
            "total_records": len(grp)
        }

    sorted_names = sorted(meta.keys(), key=lambda x: len(x), reverse=True)
    res = (meta, sorted_names, valid_v)
    _VILLAGE_META_CACHE[df_id] = res
    return res


def find_best_village_match(candidate, known_villages_list, cutoff=0.80):
    """
    Fuzzy, normalized, and suffix-aware matching against known village directory.
    Consistently handles spaces, case, spelling variations, and suffixes (e.g. -Pz, -PZ, -DW).
    """
    if not candidate:
        return None, 0.0

    cand_upper = str(candidate).upper().strip()
    cand_upper = re.sub(r"^(?:OF|IN|AT|FOR|ABOUT|NEAR)\s+", "", cand_upper).strip()
    cand_base = re.sub(r"[\s\-_]+(?:PZ|PIEZOMETER|DW|OW|TW|TUBEWELL|WELL)\b.*$", "", cand_upper, flags=re.IGNORECASE).strip()
    if not cand_base:
        cand_base = cand_upper

    if len(cand_base) < 3:
        return None, 0.0

    cand_norm = re.sub(r"[^A-Z0-9]", "", cand_base)

    # 1. Exact match against known villages
    for v in known_villages_list:
        v_upper = v.upper()
        if cand_base == v_upper or cand_upper == v_upper:
            return v, 1.0

    # 2. Normalized match (ignoring spaces, hyphens, punctuation)
    for v in known_villages_list:
        v_norm = re.sub(r"[^A-Z0-9]", "", v.upper())
        if cand_norm == v_norm:
            return v, 1.0

    # 3. Base village match (if a known village has -Pz suffix in raw data, or vice versa)
    for v in known_villages_list:
        v_base = re.sub(r"[\s\-_]+(?:PZ|PIEZOMETER|DW|OW|TW|TUBEWELL|WELL)\b.*$", "", v.upper(), flags=re.IGNORECASE).strip()
        v_base_norm = re.sub(r"[^A-Z0-9]", "", v_base)
        if cand_norm == v_base_norm:
            return v, 0.98

    # 4. Difflib close matches on candidate base
    matches = difflib.get_close_matches(cand_base, known_villages_list, n=1, cutoff=cutoff)
    if matches:
        ratio = difflib.SequenceMatcher(None, cand_base, matches[0].upper()).ratio()
        return matches[0], ratio

    # 5. Normalized difflib match
    norm_dict = {re.sub(r"[^A-Z0-9]", "", v): v for v in known_villages_list}
    norm_matches = difflib.get_close_matches(cand_norm, list(norm_dict.keys()), n=1, cutoff=cutoff)
    if norm_matches:
        orig = norm_dict[norm_matches[0]]
        ratio = difflib.SequenceMatcher(None, cand_norm, norm_matches[0]).ratio()
        return orig, ratio

    return None, 0.0


def detect_village_in_query(query, village_meta, sorted_villages, districts_set):
    """
    Accurately detect whether the query specifies a village, either explicitly or implicitly.
    Never matches district names or district aliases as villages.
    Supports English, Hindi, and Hinglish.
    """
    q_clean = query.strip()
    q_upper = q_clean.upper()

    all_district_tokens = set()
    for d in districts_set:
        all_district_tokens.add(d.upper())
    for alias in ALL_DISTRICT_ALIASES:
        all_district_tokens.add(alias.upper())

    village_keywords = r"(?:village|vill\.?|pind|gram|गांव|गाँव|ग्राम|पिंड)"

    # 1. SUCCEEDING VILLAGE KEYWORD: "<Name> village", "<Name> pind", "<Name> गांव"
    # Example: "Abulkharana village of Muktsar district", "Hussainpura village"
    succeeding_pat = re.compile(
        r"([a-zA-Z0-9\u0900-\u097F\s\-]+?)\s+" + village_keywords + r"\b",
        re.IGNORECASE
    )
    for m in succeeding_pat.finditer(q_clean):
        prefix_text = m.group(1).strip()
        prefix_words = prefix_text.split()
        if not prefix_words:
            continue
        last_word = prefix_words[-1].lower()
        if last_word in {"in", "at", "of", "for", "near", "to", "under", "me", "mein", "se", "ko", "par", "में", "का", "की", "के", "से"}:
            # This is "in village <Name>", not "<Name> village". Let preceding pattern handle it!
            continue

        for num_words in [min(3, len(prefix_words)), min(2, len(prefix_words)), 1]:
            if num_words <= 0:
                continue
            candidate = " ".join(prefix_words[-num_words:]).strip()
            cand_clean = re.sub(
                r"^(?:what\s+is\s+|the\s+|average\s+|groundwater\s+|level\s+|depth\s+|in\s+|at\s+|of\s+|for\s+|about\s+|me\s+|mein\s+|ka\s+|ki\s+|ke\s+|kya\s+|hai\s+)+",
                "", candidate, flags=re.IGNORECASE
            ).strip()
            if not cand_clean or len(cand_clean) < 3:
                continue
            if cand_clean.upper() in all_district_tokens:
                continue

            best_match, score = find_best_village_match(cand_clean, sorted_villages, cutoff=0.78)
            if best_match and best_match not in districts_set and best_match.lower() not in ALL_DISTRICT_ALIASES:
                return {"status": "found", "village": best_match, "score": score, "explicit": True, "raw": cand_clean}
            elif best_match is None and len(cand_clean) >= 4:
                return {"status": "not_found", "village": cand_clean, "explicit": True}

    # 2. PRECEDING VILLAGE KEYWORD: "village <Name>", "pind <Name>", "गांव <Name>"
    preceding_pat = re.compile(
        village_keywords + r"\s+([a-zA-Z0-9\u0900-\u097F\s\-]+?)(?:\s+(?:in|of|under|district|tehsil|block|station|में|का|की|के|जिला)|[,\.\?\!]|$|\b(groundwater|depth|level|trend|graph|chart|data|measurement|status|records?|भूजल|जलस्तर|गहराई|स्तर|रुझान|डेटा)\b)",
        re.IGNORECASE
    )
    m_prec = preceding_pat.search(q_clean)
    if m_prec:
        raw_cand = m_prec.group(1).strip()
        cleaned_cand = re.sub(r"^(?:of|in|at|under|near)\s+", "", raw_cand, flags=re.IGNORECASE).strip()
        if cleaned_cand.upper() not in all_district_tokens and cleaned_cand:
            best_match, score = find_best_village_match(cleaned_cand, sorted_villages, cutoff=0.78)
            if best_match and best_match not in districts_set and best_match.lower() not in ALL_DISTRICT_ALIASES:
                return {"status": "found", "village": best_match, "score": score, "explicit": True, "raw": cleaned_cand}
            elif best_match is None and len(cleaned_cand) >= 4:
                return {"status": "not_found", "village": cleaned_cand, "explicit": True}

    # 3. DIRECT / EXACT SUBSTRING MATCH AGAINST KNOWN VILLAGES
    for v in sorted_villages:
        if v in districts_set or v.lower() in ALL_DISTRICT_ALIASES:
            continue
        v_base = re.sub(r"[\s\-_]+(?:PZ|PIEZOMETER|DW)\b.*$", "", v, flags=re.IGNORECASE).strip()
        pat = r"\b" + re.escape(v) + r"\b"
        pat_base = r"\b" + re.escape(v_base) + r"\b" if len(v_base) >= 3 else None

        if re.search(pat, q_upper) or (pat_base and re.search(pat_base, q_upper)):
            match_span = re.search(pat, q_upper) or re.search(pat_base, q_upper)
            after_match = q_upper[match_span.end():]
            if re.match(r"^\s+DISTRICT\b", after_match):
                continue
            return {"status": "found", "village": v, "score": 1.0, "explicit": False, "raw": v}

    # 4. FUZZY TOKEN MATCH ACROSS QUERY WORDS (excluding stopwords & districts)
    tokens = [
        w for w in re.findall(r"[A-Za-z0-9\-]+", q_clean)
        if w.lower() not in STOPWORDS_SET and w.upper() not in districts_set and w.lower() not in ALL_DISTRICT_ALIASES
    ]

    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        best_match, score = find_best_village_match(bigram, sorted_villages, cutoff=0.82)
        if best_match and best_match not in districts_set and best_match.lower() not in ALL_DISTRICT_ALIASES:
            return {"status": "found", "village": best_match, "score": score, "explicit": False, "raw": bigram}

    for token in tokens:
        clean_tok = re.sub(r"[\s\-_]+(?:Pz|PZ|DW)\b", "", token, flags=re.IGNORECASE)
        if len(clean_tok) >= 4:
            best_match, score = find_best_village_match(clean_tok, sorted_villages, cutoff=0.84)
            if best_match and best_match not in districts_set and best_match.lower() not in ALL_DISTRICT_ALIASES:
                return {"status": "found", "village": best_match, "score": score, "explicit": False, "raw": token}

    return {"status": "none"}


def process_village_groundwater_query(query, raw_df, village_det, village_meta, valid_v_df, lang="en", api_key=None):
    """
    Compute exact statistics, handle administrative disambiguation, generate yearly line graph,
    and generate grounded response in the user's language for village-level groundwater queries.
    """
    # Case A: Explicit village query asked, but village is not present in CGWB monitoring network
    if village_det["status"] == "not_found":
        v_name = village_det["village"].title()
        if lang == "hi":
            answer = (
                f"पंजाब केंद्रीय भूजल बोर्ड (CGWB) डेटासेट में गांव **'{v_name}'** के लिए कोई भूजल निगरानी रिकॉर्ड नहीं मिला।\n\n"
                f"**महत्वपूर्ण जानकारी:**\n"
                f"- CGWB पंजाब भर में विशिष्ट गांवों और कस्बों में स्थापित स्थायी वेधशाला स्टेशनों (piezometers) के माध्यम से भूजल स्तर की निगरानी करता है।\n"
                f"- यदि किसी गांव में निर्दिष्ट CGWB स्टेशन नहीं है, तो उस व्यक्तिगत गांव के स्तर के आंकड़े डेटासेट में नहीं होते।\n"
                f"- शून्य-भ्रम (Zero-Hallucination) सिद्धांत के अनुसार, यह प्रणाली जिले के आंकड़ों को गांव का परिणाम मानकर प्रस्तुत नहीं करती।\n\n"
                f"💡 *सुझाव: आप उन गांवों के बारे में पूछ सकते हैं जहां CGWB स्टेशन सक्रिय हैं, जैसे **Hussainpura**, **Dhilwan**, **Balluana**, **Devigarh**, **Budhlada**, या **Bhatnura Khurd**.*"
            )
        elif lang == "hinglish":
            answer = (
                f"Village **'{v_name}'** ke liye CGWB Punjab dataset mein koi groundwater records nahi mile.\n\n"
                f"- CGWB Punjab mein dedicated monitoring stations ke through data collect karta hai.\n"
                f"- Agar kisi village mein designated station nahi hai, toh uska individual record available nahi hota.\n"
                f"- Galat data se bachne ke liye district stats ko village ka result nahi banaya jata.\n\n"
                f"💡 *Suggestion: Aap active stations wale villages query kar sakte hain, jaise **Hussainpura**, **Balluana**, ya **Devigarh**.*"
            )
        else:
            answer = (
                f"No groundwater monitoring records were found for village **'{v_name}'** in the Punjab Central Ground Water Board (CGWB) dataset.\n\n"
                f"**Important Context:**\n"
                f"- CGWB monitors groundwater levels across Punjab through dedicated permanent observation stations and piezometers located in specific villages and towns.\n"
                f"- If a village does not host a designated CGWB monitoring station, its individual village-level measurements are not present in this dataset.\n"
                f"- To prevent inaccurate data attribution and avoid false claims, this system **does not substitute district-level statistics** as village-level results.\n\n"
                f"💡 *Suggestion: You can query villages that have active CGWB monitoring stations, such as **Hussainpura**, **Dhilwan**, **Balluana**, **Devigarh**, **Budhlada**, or **Bhatnura Khurd**.*"
            )
        return {
            "answer": answer,
            "engine": "Zero-Hallucination Guardrail (Village Not Found)",
            "intent": "village_not_found",
            "sources": ["CGWB Punjab Monitoring Network"],
            "records": pd.DataFrame(),
            "figure": None
        }

    v_upper = village_det["village"]
    meta = village_meta[v_upper]
    v_records = valid_v_df[valid_v_df["Village_Upper"] == v_upper].copy()

    districts = meta["districts"]
    q_upper = query.upper()

    # Check if user mentioned a specific district or tehsil in the query (with alias support)
    from_known = sorted(list(raw_df["District"].dropna().unique().tolist()))
    mentioned_dists = extract_districts_from_query(query, from_known)
    selected_dist = None
    for qd in mentioned_dists:
        if qd in districts:
            selected_dist = qd
            break
    if not selected_dist:
        for d in districts:
            if re.search(r"\b" + re.escape(d) + r"\b", q_upper):
                selected_dist = d
                break

    selected_tehsil = None
    for t in meta["tehsils"]:
        if t and t not in ["-", "NAN"] and re.search(r"\b" + re.escape(t.upper()) + r"\b", q_upper):
            selected_tehsil = t
            break

    # Case B: Disambiguation required when multiple villages share the same name across districts
    if len(districts) > 1 and not selected_dist and not selected_tehsil:
        candidate_rows = []
        cand_list_text = []
        for i, d in enumerate(districts):
            sub = v_records[v_records["District"] == d]
            tehsils_str = ", ".join(sorted(sub["Tehsil"].dropna().unique().tolist())) or "N/A"
            blocks_str = ", ".join(sorted(sub["Block"].dropna().unique().tolist())) or "N/A"
            stations_str = ", ".join(sorted(sub["Station"].dropna().unique().tolist())) or "N/A"
            cnt = len(sub)
            m_depth = sub[LEVEL_COL].mean()

            candidate_rows.append({
                "Village": meta["display_name"],
                "District": d,
                "Tehsil": tehsils_str,
                "Block": blocks_str,
                "Stations": stations_str,
                "Observations": cnt,
                "Average Depth (mbgl)": round(m_depth, 2)
            })

            cand_list_text.append(
                f"{i+1}. **Village {meta['display_name']} in District {d}** (Tehsil: {tehsils_str}, Block: {blocks_str}) — "
                f"**{cnt} observations**, average depth: **{m_depth:.2f} mbgl** [Stations: `{stations_str}`]"
            )

        cand_df = pd.DataFrame(candidate_rows)
        if lang == "hi":
            answer = (
                f"पंजाब के विभिन्न जिलों में **'{meta['display_name']}'** नाम के कई गांव मिले हैं:\n\n"
                + "\n".join(cand_list_text) + "\n\n"
                f"⚠️ **कृपया स्पष्ट करें कि आप किस जिले या तहसील के बारे में पूछ रहे हैं** "
                f"(उदाहरण: *'{meta['display_name']} in {districts[0]}'* या *'{meta['display_name']} in {districts[1]}'*)। "
                f"इससे सही और स्रोत-सत्यापित आंकड़े प्रदान किए जा सकेंगे।"
            )
        else:
            answer = (
                f"Multiple villages named **'{meta['display_name']}'** were found in different administrative districts of Punjab:\n\n"
                + "\n".join(cand_list_text) + "\n\n"
                f"⚠️ **Please specify which district or tehsil you are inquiring about** "
                f"(for example: *'{meta['display_name']} in {districts[0]}'* or *'{meta['display_name']} in {districts[1]}'*). "
                f"This ensures exact source-grounded calculations without conflating data from separate geographic locations."
            )
        return {
            "answer": answer,
            "engine": "Administrative Disambiguation Guardrail",
            "intent": "village_disambiguation",
            "sources": ["CGWB Manual Quarterly Groundwater Dataset (Punjab)"],
            "records": cand_df,
            "figure": None
        }

    # Filter to specific district if matched or if only 1 district exists
    if selected_dist:
        v_records = v_records[v_records["District"] == selected_dist]
    elif len(districts) == 1:
        selected_dist = districts[0]

    if selected_tehsil:
        v_records = v_records[v_records["Tehsil"] == selected_tehsil]

    # Calculate exact statistics
    obs_count = len(v_records)
    mean_val = v_records[LEVEL_COL].mean()
    min_val = v_records[LEVEL_COL].min()
    max_val = v_records[LEVEL_COL].max()

    dist_name = selected_dist or ", ".join(districts)
    tehsils_list = sorted([t for t in v_records["Tehsil"].dropna().unique().tolist() if t not in ["-", "", "nan"]])
    blocks_list = sorted([b for b in v_records["Block"].dropna().unique().tolist() if b not in ["-", "", "nan"]])
    stations_list = sorted([s for s in v_records["Station"].dropna().unique().tolist() if s not in ["-", "", "nan"]])
    sources_list = sorted(v_records["Source File"].dropna().unique().tolist())

    facts = [
        f"GROUNDWATER MONITORING DATA FOR VILLAGE: {meta['display_name'].upper()}",
        f"- Administrative Location: District {dist_name}" + (f", Tehsil: {', '.join(tehsils_list)}" if tehsils_list else "") + (f", Block: {', '.join(blocks_list)}" if blocks_list else ""),
        f"- Total Valid Monitoring Observations: {obs_count:,}",
        f"- Average (Mean) Water Table Depth: {mean_val:.2f} meters below ground level (mbgl)",
        f"- Minimum Recorded Depth: {min_val:.2f} mbgl (shallowest observed)",
        f"- Maximum Recorded Depth: {max_val:.2f} mbgl (deepest observed)",
        f"- Monitoring Stations in this Village: {', '.join(stations_list)}"
    ]

    if min_val < 0:
        facts.append(
            f"- Hydrological Explanation of Negative Depth ({min_val:.2f} mbgl): Negative values in mbgl represent "
            f"artesian conditions (groundwater pressure head rising above ground surface) or surface waterlogged zones."
        )
    if max_val >= 100.0:
        facts.append(
            f"- Hydrological Explanation of Extreme Depth ({max_val:.2f} mbgl): Extreme depths (>100 mbgl) reflect deep "
            f"confined aquifer exploratory monitoring or localized heavy extraction depression cones."
        )

    # Historical trend & Plotly graph (using actual observations, no interpolation)
    fig = None
    if "Year" in v_records.columns and v_records["Year"].notna().any():
        yearly_v = v_records.dropna(subset=["Year"]).groupby("Year", as_index=False)[LEVEL_COL].mean().sort_values("Year")
        if not yearly_v.empty:
            min_yr = int(yearly_v["Year"].min())
            max_yr = int(yearly_v["Year"].max())
            baseline_row = yearly_v.iloc[0]
            recent_row = yearly_v.iloc[-1]
            net_change = recent_row[LEVEL_COL] - baseline_row[LEVEL_COL]

            trend_desc = "Depletion / Deepening Water Table" if net_change > 0 else ("Rising Water Table" if net_change < 0 else "Stable Water Table")
            facts.append(f"\nHISTORICAL TREND ({min_yr} to {max_yr} across {len(yearly_v)} active observation years):")
            facts.append(f"- Baseline Year ({int(baseline_row['Year'])}): Average depth was {baseline_row[LEVEL_COL]:.2f} mbgl")
            facts.append(f"- Recent Year ({int(recent_row['Year'])}): Average depth reached {recent_row[LEVEL_COL]:.2f} mbgl")
            facts.append(f"- Net Water Level Change: {net_change:+.2f} meters ({trend_desc})")

            # Generate Plotly line chart
            fig = px.line(
                yearly_v,
                x="Year",
                y=LEVEL_COL,
                markers=True,
                title=f"Yearly Groundwater Trend: Village {meta['display_name']} ({dist_name})",
                labels={"Year": "Observation Year", LEVEL_COL: "Average Depth (mbgl)"}
            )
            fig.update_traces(line=dict(color="#1f77b4", width=2.5), marker=dict(size=6, color="#0d47a1"))
            fig.update_layout(
                hovermode="x unified",
                yaxis_title="Depth to Water Table (mbgl)",
                xaxis=dict(dtick=max(1, (max_yr - min_yr) // 8))
            )

    context_text = "\n".join(facts)

    # Groq LLM Generation in the user's language
    llm_answer, used_model, err_msg = generate_grounded_answer_with_groq(query, context_text, lang=lang, api_key=api_key)
    if llm_answer:
        final_answer = llm_answer
        engine = f"Groq ({used_model}) - Village Analytics"
    else:
        final_answer = generate_targeted_fallback_answer(query, context_text, lang=lang)
        if api_key:
            engine = f"Verified Village Analytics (Groq API Error: {err_msg})"
        else:
            engine = "Verified Village Analytics Engine (Offline Mode)"

    supporting_cols = [c for c in ["District", "Tehsil", "Block", "Village", "Station", DATE_COL, LEVEL_COL, "Source File"] if c in v_records.columns]
    supporting_df = v_records[supporting_cols].head(15)

    return {
        "answer": final_answer,
        "engine": engine,
        "intent": "village_analytics",
        "sources": sources_list + ["CGWB Station Monitoring Network (Punjab)"],
        "records": supporting_df,
        "figure": fig
    }


# ==============================================================================
# DISTRICT GRAPH & TREND GENERATION
# ==============================================================================

def generate_district_trend_graph(query, raw_df, mentioned_districts, start_year=None, end_year=None):
    """
    Intelligently generate a Plotly graph for district queries:
    - Single district yearly line chart (when 1 district mentioned)
    - Multi-district comparison line chart (when >1 districts mentioned)
    - District ranking bar chart (when ranking/top-N/deepest districts requested)
    - Statewide Punjab yearly trend line chart (ONLY when yearly/historical trend requested)
    Filters to exact (start_year, end_year) if requested, guaranteeing consistency with written text.
    """
    q_low = query.lower()

    # Filter data by year range if specified
    df_filtered = raw_df.copy()
    if start_year is not None and end_year is not None and "Year" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Year"].between(start_year, end_year)]

    title_suffix = f" ({start_year}–{end_year})" if (start_year and end_year) else ""

    # Check for Top-N or Bottom-N
    top_n_match = re.search(r"\b(?:top|first|highest|deepest)\s+(\d+)\b", q_low)
    bottom_n_match = re.search(r"\b(?:bottom|lowest|shallowest)\s+(\d+)\b", q_low)

    is_explicit_yearly_trend = any(phrase in q_low for phrase in [
        "trend", "trends", "yearly", "annual", "over the years", "historical",
        "over time", "available years", "change over the years", "how has groundwater changed",
        "saal-dar-saal", "rujhan", "रुझान", "सालाना"
    ]) and not any(r in q_low for r in ["rank", "ranking", "deepest district", "shallowest district", "highest average", "lowest average"])

    is_district_ranking = any(phrase in q_low for phrase in [
        "top", "bottom", "rank", "ranking", "highest", "deepest", "shallowest",
        "lowest", "most depleted", "least depleted", "compare districts",
        "district comparison", "all districts", "all punjab districts",
        "every district", "across districts", "all district", "average groundwater depth",
        "depth by district", "रैंकिंग", "रैंक", "तुलना", "सबसे गहरा", "सबसे कम"
    ])

    # Case 1: Multi-district comparison trend
    if len(mentioned_districts) > 1:
        if "Year" in df_filtered.columns:
            sub = df_filtered[df_filtered["District"].isin(mentioned_districts)].dropna(subset=["Year"])
            yearly = sub.groupby(["District", "Year"], as_index=False)[LEVEL_COL].mean().sort_values(["District", "Year"])
            fig = px.line(
                yearly,
                x="Year",
                y=LEVEL_COL,
                color="District",
                markers=True,
                title=f"Groundwater Trend Comparison: {', '.join(mentioned_districts)}{title_suffix}",
                labels={"Year": "Observation Year", LEVEL_COL: "Average Depth (mbgl)"}
            )
            fig.update_layout(hovermode="x unified", yaxis_title="Average Depth (mbgl)")
            return fig

    # Case 2: Single district trend
    if len(mentioned_districts) == 1:
        d = mentioned_districts[0]
        if "Year" in df_filtered.columns:
            sub = df_filtered[df_filtered["District"] == d].dropna(subset=["Year"])
            yearly = sub.groupby("Year", as_index=False)[LEVEL_COL].mean().sort_values("Year")
            if not yearly.empty:
                min_yr = int(yearly["Year"].min())
                max_yr = int(yearly["Year"].max())
                fig = px.line(
                    yearly,
                    x="Year",
                    y=LEVEL_COL,
                    markers=True,
                    title=f"Yearly Groundwater Trend: District {d} ({min_yr}–{max_yr})",
                    labels={"Year": "Observation Year", LEVEL_COL: "Average Depth (mbgl)"}
                )
                fig.update_traces(line=dict(color="#1f77b4", width=2.5), marker=dict(size=6))
                fig.update_layout(hovermode="x unified", yaxis_title="Depth to Water Table (mbgl)")
                return fig

    # Case 3: District Ranking Bar Chart (Relevant chart for rankings / top-N / district comparisons)
    if is_district_ranking and not is_explicit_yearly_trend and not mentioned_districts:
        summary = df_filtered.groupby("District")[LEVEL_COL].agg(
            mean="mean", count="count", min="min", max="max"
        ).reset_index()
        summary["mean"] = summary["mean"].round(2)

        if bottom_n_match:
            n_val = int(bottom_n_match.group(1))
            chart_df = summary.sort_values("mean", ascending=True).head(n_val)
            chart_title = f"Top {n_val} Punjab Districts with Shallowest Groundwater Table{title_suffix}"
        elif top_n_match:
            n_val = int(top_n_match.group(1))
            chart_df = summary.sort_values("mean", ascending=False).head(n_val)
            chart_title = f"Top {n_val} Punjab Districts by Average Groundwater Depth (Deepest){title_suffix}"
        else:
            chart_df = summary.sort_values("mean", ascending=False)
            chart_title = f"Punjab Groundwater Depth by District (Deepest to Shallowest){title_suffix}"

        fig = px.bar(
            chart_df,
            x="District",
            y="mean",
            text="mean",
            title=chart_title,
            labels={"mean": "Average Depth (mbgl)", "District": "District"},
            hover_data={"count": True, "mean": ":.2f"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, yaxis_title="Average Depth (mbgl)")
        return fig

    # Case 4: Statewide Punjab yearly trend line chart (ONLY when user explicitly asks for yearly trend)
    if is_explicit_yearly_trend:
        if "Year" in df_filtered.columns:
            yearly = df_filtered.dropna(subset=["Year"]).groupby("Year", as_index=False)[LEVEL_COL].mean().sort_values("Year")
            if not yearly.empty:
                min_yr = int(yearly["Year"].min())
                max_yr = int(yearly["Year"].max())
                fig = px.line(
                    yearly,
                    x="Year",
                    y=LEVEL_COL,
                    markers=True,
                    title=f"Statewide Groundwater Trend: Punjab Overall ({min_yr}–{max_yr})",
                    labels={"Year": "Observation Year", LEVEL_COL: "Statewide Average Depth (mbgl)"}
                )
                fig.update_traces(line=dict(color="#2ca02c", width=2.5), marker=dict(size=6))
                fig.update_layout(hovermode="x unified", yaxis_title="Average Depth (mbgl)")
                return fig

    return None


# ==============================================================================
# ANALYTICAL COMPUTATION ENGINE
# ==============================================================================

def compute_analytical_facts(query, raw_df, district_summary_df, yearly_summary_df, start_year=None, end_year=None):
    """
    Compute mathematically reliable statistics using pandas tailored specifically
    to what the user asked. Applies time filtering if specified to ensure full consistency.
    """
    q = query.lower()
    known_districts = sorted(raw_df["District"].dropna().unique().tolist())
    mentioned_districts = extract_districts_from_query(query, known_districts)

    facts = []
    supporting_df = pd.DataFrame()

    # Apply year filter to raw data if specified
    active_df = raw_df.copy()
    if start_year is not None and end_year is not None and "Year" in active_df.columns:
        active_df = active_df[active_df["Year"].between(start_year, end_year)]
        year_filter_note = f" (Filtered to {start_year}–{end_year})"
    else:
        year_filter_note = ""

    # Raw dataset filenames present in active_df
    all_raw_files = sorted(active_df["Source File"].dropna().unique().tolist())

    # Check if this is an explicit data availability query
    is_availability_check = any(w in q for w in [
        "available", "availability", "data available", "exist", "present",
        "उपलब्ध", "डेटा है क्या", "आंकड़े", "data hai", "uplabdh"
    ])

    # Case A: User asks about a specific district or comparing multiple districts
    if mentioned_districts:
        dist_sources = sorted(active_df[active_df["District"].isin(mentioned_districts)]["Source File"].dropna().unique().tolist())
        sources = dist_sources if dist_sources else all_raw_files

        if len(mentioned_districts) == 1:
            d = mentioned_districts[0]
            dist_records = active_df[active_df["District"] == d]

            if dist_records.empty:
                facts.append(f"GROUNDWATER DATA AVAILABILITY FOR DISTRICT {d}:")
                facts.append(f"- Status: No monitoring observations found for district {d} within the specified period{year_filter_note}.")
                facts.append(f"- Note: The dataset strictly covers Central Ground Water Board (CGWB) observation records for Punjab.")
            else:
                mean_val = dist_records[LEVEL_COL].mean()
                min_val = dist_records[LEVEL_COL].min()
                max_val = dist_records[LEVEL_COL].max()
                rec_count = len(dist_records)
                d_sources_str = ", ".join(sorted(dist_records["Source File"].dropna().unique().tolist()))

                facts.append(f"GROUNDWATER DATA FOR DISTRICT {d}{year_filter_note}:")
                if is_availability_check:
                    facts.append(f"- Data Availability Confirmation: Yes, groundwater monitoring data for District {d} IS available in the Central Ground Water Board (CGWB) dataset.")
                facts.append(f"- Total Valid Monitoring Observations: {rec_count:,}")
                facts.append(f"- Average (Mean) Depth to Water Table: {mean_val:.2f} meters below ground level (mbgl)")
                facts.append(f"- Minimum Recorded Depth: {min_val:.2f} mbgl (shallowest observed)")
                facts.append(f"- Maximum Recorded Depth: {max_val:.2f} mbgl (deepest observed)")
                facts.append(f"- Raw Source Dataset(s): {d_sources_str}")

                if min_val < 0:
                    facts.append(
                        f"- Hydrological Explanation of Negative Depth ({min_val:.2f} mbgl): Negative values represent artesian conditions "
                        f"(groundwater pressure head rising above ground level) or surface waterlogged zones."
                    )
                if max_val >= 100.0:
                    facts.append(
                        f"- Hydrological Explanation of Extreme Depth ({max_val:.2f} mbgl): Extreme maximum depth values (>100 mbgl) "
                        f"reflect specialized deep confined aquifer exploratory piezometer observations or localized deep cones of depression."
                    )

                if "SAS NAGAR" in d or "MOHALI" in d:
                    orig_counts = dist_records["District_Original"].value_counts() if "District_Original" in dist_records.columns else {}
                    mohali_cnt = orig_counts.get("MOHALI", 0)
                    sas_cnt = orig_counts.get("SAS NAGAR", 0)
                    facts.append(
                        f"- Unified District Profile: SAS Nagar and Mohali are treated as the same administrative district '{d}'. "
                        f"This includes {mohali_cnt:,} standard monitoring records (1996–2024, mean 9.44 mbgl) and "
                        f"{sas_cnt:,} specialized exploratory piezometer records (2023–2024, mean 61.24 mbgl) "
                        f"monitoring deep, medium and shallow aquifer horizons at Akalgarh, Sundran, and Dera Bassi/Handesra."
                    )

                if "Year" in dist_records.columns and dist_records["Year"].notna().any():
                    dist_yearly = dist_records.dropna(subset=["Year"]).groupby("Year", as_index=False)[LEVEL_COL].mean().sort_values("Year")
                    if not dist_yearly.empty:
                        min_yr = int(dist_yearly["Year"].min())
                        max_yr = int(dist_yearly["Year"].max())
                        b_val = dist_yearly.iloc[0][LEVEL_COL]
                        r_val = dist_yearly.iloc[-1][LEVEL_COL]
                        chg = r_val - b_val
                        trend_txt = "Depletion / Deepening Water Table" if chg > 0 else ("Rising Water Table" if chg < 0 else "Stable Water Table")
                        facts.append(f"- Historical Monitoring Range: {min_yr} to {max_yr} ({len(dist_yearly)} observation years)")
                        facts.append(f"- Baseline ({min_yr}): {b_val:.2f} mbgl | Recent ({max_yr}): {r_val:.2f} mbgl (Net Change: {chg:+.2f} meters, {trend_txt})")

                # State rank
                curr_dist_stats = active_df.groupby("District")[LEVEL_COL].agg(["mean", "count"]).reset_index()
                state_mean = (curr_dist_stats["mean"] * curr_dist_stats["count"]).sum() / curr_dist_stats["count"].sum()
                rank_series = curr_dist_stats.sort_values("mean", ascending=False).reset_index(drop=True)
                rank_idx = rank_series[rank_series["District"] == d].index
                if len(rank_idx) > 0:
                    dist_rank = rank_idx[0] + 1
                    facts.append(f"- Depletion Rank: #{dist_rank} deepest out of {len(curr_dist_stats)} districts in Punjab")
                    facts.append(f"- Punjab Statewide Weighted Mean: {state_mean:.2f} mbgl")

                supporting_cols = [c for c in ["District", "Station", "Tehsil", "Block", "Village", DATE_COL, LEVEL_COL, "Source File"] if c in dist_records.columns]
                supporting_df = dist_records[supporting_cols].head(10)

        else:
            facts.append(f"COMPARISON BETWEEN DISTRICTS ({', '.join(mentioned_districts)}){year_filter_note}:")
            compare_rows = []
            trend_details = []
            for d in mentioned_districts:
                dist_records = active_df[active_df["District"] == d]
                if not dist_records.empty:
                    m = dist_records[LEVEL_COL].mean()
                    mi = dist_records[LEVEL_COL].min()
                    ma = dist_records[LEVEL_COL].max()
                    c = len(dist_records)
                    d_src = ", ".join(sorted(dist_records["Source File"].dropna().unique().tolist()))

                    trend_str = ""
                    if "Year" in dist_records.columns and dist_records["Year"].notna().any():
                        dist_yearly = dist_records.dropna(subset=["Year"]).groupby("Year", as_index=False)[LEVEL_COL].mean().sort_values("Year")
                        if not dist_yearly.empty:
                            b_yr = int(dist_yearly.iloc[0]["Year"])
                            b_val = dist_yearly.iloc[0][LEVEL_COL]
                            r_yr = int(dist_yearly.iloc[-1]["Year"])
                            r_val = dist_yearly.iloc[-1][LEVEL_COL]
                            net_chg = r_val - b_val
                            t_desc = "Depleted / Deepened" if net_chg > 0 else ("Rose" if net_chg < 0 else "Stable")
                            trend_str = f" | {b_yr}–{r_yr}: {b_val:.2f}m -> {r_val:.2f}m (Net Change: {net_chg:+.2f}m, {t_desc})"
                            trend_details.append({
                                "district": d,
                                "mean": m,
                                "b_val": b_val,
                                "r_val": r_val,
                                "net_chg": net_chg,
                                "b_yr": b_yr,
                                "r_yr": r_yr
                            })

                    facts.append(f"- {d}: Average Depth: {m:.2f} mbgl (Observations: {c:,}, Range: {mi:.2f}m to {ma:.2f}m, Sources: {d_src}{trend_str})")
                    compare_rows.append({
                        "District": d,
                        "Average Depth (mbgl)": round(m, 2),
                        "Observations": c,
                        "Min (m)": round(mi, 2),
                        "Max (m)": round(ma, 2),
                        "Source File": d_src
                    })

            if len(trend_details) >= 2:
                d1 = trend_details[0]
                d2 = trend_details[1]
                deeper = d1 if d1["mean"] > d2["mean"] else d2
                shallower = d2 if d1["mean"] > d2["mean"] else d1
                depth_diff = deeper["mean"] - shallower["mean"]
                faster = d1 if d1["net_chg"] > d2["net_chg"] else d2
                slower = d2 if d1["net_chg"] > d2["net_chg"] else d1
                chg_diff = faster["net_chg"] - slower["net_chg"]
                facts.append(f"\nCOMPARATIVE ANALYSIS ({d1['district']} vs {d2['district']}):")
                facts.append(f"- Depth Comparison: {deeper['district']} has an overall deeper average water table ({deeper['mean']:.2f} mbgl) compared to {shallower['district']} ({shallower['mean']:.2f} mbgl) — a difference of {depth_diff:.2f} meters.")
                facts.append(f"- Historical Depletion Comparison: Over the observation period, {faster['district']} experienced greater water table decline ({faster['net_chg']:+.2f} m) compared to {slower['district']} ({slower['net_chg']:+.2f} m) — a rate delta of {chg_diff:+.2f} meters.")

            if compare_rows:
                supporting_df = pd.DataFrame(compare_rows)

    # General / Multi-district queries (No specific district named)
    else:
        sources = all_raw_files

        # Compute accurate district summary directly from active_df
        district_stats = (
            active_df.groupby("District")[LEVEL_COL]
            .agg(Observations="count", Mean_Depth="mean", Min_Depth="min", Max_Depth="max")
            .reset_index()
        )
        district_stats["Average Depth (mbgl)"] = district_stats["Mean_Depth"].round(2)
        district_stats["Min (m)"] = district_stats["Min_Depth"].round(2)
        district_stats["Max (m)"] = district_stats["Max_Depth"].round(2)

        # Map actual source files per district
        district_sources = {}
        for d, grp in active_df.groupby("District"):
            district_sources[d] = ", ".join(sorted(grp["Source File"].dropna().unique().tolist()))
        district_stats["Source File"] = district_stats["District"].map(district_sources)

        # Detect Top-N or Bottom-N
        top_n_match = re.search(r"\b(?:top|first|highest|deepest)\s+(\d+)\b", q)
        bottom_n_match = re.search(r"\b(?:bottom|lowest|shallowest)\s+(\d+)\b", q)

        is_lowest_query = any(w in q for w in ["lowest", "minimum", "min", "shallowest", "least depleted", "best", "न्यूनतम", "सबसे कम", "उथला", "sabse kam"])
        is_highest_query = any(w in q for w in ["highest", "maximum", "max", "deepest", "most depleted", "worst", "अधिकतम", "सबसे गहरा", "गहरा", "sabse gehra"])
        is_ranking_query = any(w in q for w in ["ranking", "rank", "top", "bottom", "order", "रैंकिंग", "रैंक"])
        is_trend_query = any(w in q for w in [
            "trend", "trends", "yearly", "annual", "over the years", "historical",
            "over time", "available years", "change over the years", "how has groundwater changed",
            "saal-dar-saal", "rujhan", "रुझान", "सालाना"
        ]) and not any(r in q for r in ["rank", "ranking", "deepest district", "shallowest district", "highest average", "lowest average"])

        # Case B: Top-N / Highest / Deepest / Ranking query
        if (top_n_match or is_highest_query or is_ranking_query) and not bottom_n_match and not is_lowest_query and not is_trend_query:
            n_val = int(top_n_match.group(1)) if top_n_match else 5
            sorted_by_mean = district_stats.sort_values("Mean_Depth", ascending=False).reset_index(drop=True)
            sorted_by_mean["Rank"] = range(1, len(sorted_by_mean) + 1)
            display_df = sorted_by_mean.head(n_val).copy()

            facts.append(f"TOP {len(display_df)} PUNJAB DISTRICTS WITH THE HIGHEST AVERAGE GROUNDWATER DEPTH (MOST DEPLETED){year_filter_note}:")
            facts.append(f"Calculated directly from the complete Central Ground Water Board (CGWB) Punjab dataset ({len(active_df):,} total valid observations across {len(district_stats)} districts):\n")

            for _, r in display_df.iterrows():
                facts.append(
                    f"- Rank #{int(r['Rank'])}: {r['District']} — Average Depth: {r['Average Depth (mbgl)']:.2f} mbgl "
                    f"(Observations: {int(r['Observations']):,}, Range: {r['Min (m)']:.2f}m to {r['Max (m)']:.2f}m, Source: {r['Source File']})"
                )

            facts.append("\nCOMPLETE VERIFIED RANKING TABLE:")
            facts.append("| Rank | District | Average Depth (mbgl) | Observations | Min Depth (mbgl) | Max Depth (mbgl) | Source Dataset |")
            facts.append("|------|----------|----------------------|--------------|------------------|------------------|----------------|")
            for _, r in display_df.iterrows():
                facts.append(f"| {int(r['Rank'])} | {r['District']} | {r['Average Depth (mbgl)']:.2f} | {int(r['Observations']):,} | {r['Min (m)']:.2f} | {r['Max (m)']:.2f} | {r['Source File']} |")

            facts.append(
                f"\nADDITIONAL VERIFIED NOTES:\n"
                f"- Hydrological Explanation of Negative Depths: Negative minimum values (e.g., Mukatsar -1.84 mbgl, Pathankot -2.19 mbgl) represent artesian conditions (piezometric head above ground level) or surface waterlogging.\n"
                f"- Shallowest District Overall: {display_df.iloc[0]['District']} has the shallowest water table in Punjab ({display_df.iloc[0]['Average Depth (mbgl)']:.2f} mbgl)."
            )

            # Notes
            highest_max_row = district_stats.sort_values("Max_Depth", ascending=False).iloc[0]
            deepest_district = sorted_by_mean.iloc[0]
            sas_matches = district_stats[district_stats["District"].str.contains("SAS NAGAR", na=False)]
            sas_note = f"- SAS Nagar & Mohali Normalization: SAS Nagar and Mohali are unified as '{sas_matches.iloc[0]['District']}' with {int(sas_matches.iloc[0]['Observations']):,} observations and a combined mean of {sas_matches.iloc[0]['Average Depth (mbgl)']:.2f} mbgl (ranking #{int(district_stats.sort_values('Mean_Depth', ascending=False).reset_index(drop=True)[district_stats.sort_values('Mean_Depth', ascending=False).reset_index(drop=True)['District'] == sas_matches.iloc[0]['District']].index[0]) + 1} in Punjab). Deep exploratory piezometer observations (42 records averaging 61.24 mbgl) are integrated directly from raw data without distorting district rankings.\n" if not sas_matches.empty else ""
            facts.append(
                f"\nADDITIONAL VERIFIED NOTES:\n"
                f"- Deepest District Overall: {deepest_district['District']} has the deepest average water table at {deepest_district['Average Depth (mbgl)']:.2f} mbgl across {int(deepest_district['Observations']):,} valid observations.\n"
                f"- Deepest Single Reading: The deepest individual measurement recorded at any single monitoring station is {highest_max_row['Max_Depth']:.2f} mbgl in district {highest_max_row['District']}.\n"
                f"{sas_note}"
                f"- Hydrological Explanation of Negative Depths: Negative minimum values (e.g., Mukatsar -1.84 mbgl, Pathankot -2.19 mbgl, Sangrur -1.07 mbgl) indicate artesian conditions (piezometric head above ground surface) or surface waterlogging.\n"
                f"- Hydrological Explanation of Extreme Depths: Extreme depths (>100 mbgl, such as Sangrur 268.0 mbgl and Jalandhar 250.0 mbgl) reflect specialized deep exploratory piezometers and localized depletion cones, preserved in the dataset without alteration."
            )

            supporting_df = display_df[[
                "Rank", "District", "Average Depth (mbgl)", "Observations", "Min (m)", "Max (m)", "Source File"
            ]].copy()

        # Case C: Bottom-N / Lowest / Shallowest query
        elif (bottom_n_match or is_lowest_query) and not is_trend_query:
            n_val = int(bottom_n_match.group(1)) if bottom_n_match else 5
            sorted_by_mean = district_stats.sort_values("Mean_Depth", ascending=True).reset_index(drop=True)
            sorted_by_mean["Rank"] = range(1, len(sorted_by_mean) + 1)
            display_df = sorted_by_mean.head(n_val).copy()

            facts.append(f"TOP {len(display_df)} PUNJAB DISTRICTS WITH SHALLOWEST GROUNDWATER TABLE (BEST WATER TABLE){year_filter_note}:")
            facts.append(f"Calculated directly from the complete Central Ground Water Board (CGWB) Punjab dataset ({len(active_df):,} total valid observations across {len(district_stats)} districts):\n")

            for _, r in display_df.iterrows():
                facts.append(
                    f"- Rank #{int(r['Rank'])}: {r['District']} — Average Depth: {r['Average Depth (mbgl)']:.2f} mbgl "
                    f"(Observations: {int(r['Observations']):,}, Range: {r['Min (m)']:.2f}m to {r['Max (m)']:.2f}m, Source: {r['Source File']})"
                )

            facts.append("\nCOMPLETE VERIFIED RANKING TABLE:")
            facts.append("| Rank | District | Average Depth (mbgl) | Observations | Min Depth (mbgl) | Max Depth (mbgl) | Source Dataset |")
            facts.append("|------|----------|----------------------|--------------|------------------|------------------|----------------|")
            for _, r in display_df.iterrows():
                facts.append(f"| {int(r['Rank'])} | {r['District']} | {r['Average Depth (mbgl)']:.2f} | {int(r['Observations']):,} | {r['Min (m)']:.2f} | {r['Max (m)']:.2f} | {r['Source File']} |")

            facts.append(
                f"\nADDITIONAL VERIFIED NOTES:\n"
                f"- Hydrological Explanation of Negative Depths: Negative minimum values (e.g., Mukatsar -1.84 mbgl, Pathankot -2.19 mbgl) represent artesian conditions (piezometric head above ground level) or surface waterlogging.\n"
                f"- Shallowest District Overall: {display_df.iloc[0]['District']} has the shallowest water table in Punjab ({display_df.iloc[0]['Average Depth (mbgl)']:.2f} mbgl)."
            )

            supporting_df = display_df[[
                "Rank", "District", "Average Depth (mbgl)", "Observations", "Min (m)", "Max (m)", "Source File"
            ]].copy()

        # Case D: Historical Trend Query
        elif is_trend_query:
            curr_yearly = active_df.dropna(subset=["Year"])
            if not curr_yearly.empty:
                yearly_avg = curr_yearly.groupby("Year")[LEVEL_COL].agg(
                    Mean_Depth="mean", Observations="count"
                ).reset_index().sort_values("Year")
                yearly_avg["Average Depth (mbgl)"] = yearly_avg["Mean_Depth"].round(2)

                first_row = yearly_avg.iloc[0]
                last_row = yearly_avg.iloc[-1]
                net_change = last_row["Average Depth (mbgl)"] - first_row["Average Depth (mbgl)"]
                trend_txt = "Deepening / Water Table Decline" if net_change > 0 else "Rising Water Table"

                facts.append(f"HISTORICAL GROUNDWATER TREND IN PUNJAB{year_filter_note}:")
                facts.append(f"- Baseline Year ({int(first_row['Year'])}): Average depth was {first_row['Average Depth (mbgl)']:.2f} mbgl across {int(first_row['Observations']):,} observations")
                facts.append(f"- Recent Year ({int(last_row['Year'])}): Average depth reached {last_row['Average Depth (mbgl)']:.2f} mbgl across {int(last_row['Observations']):,} observations")
                facts.append(f"- Net Multi-Year Water Level Change: {net_change:+.2f} meters ({trend_txt})")
                facts.append(f"- Active Monitoring Period: {int(first_row['Year'])} to {int(last_row['Year'])} ({len(yearly_avg)} observation years)")
                facts.append(f"- Datasets: {', '.join(all_raw_files)}")

                facts.append("\nYEARLY GROUNDWATER DEPTH OBSERVATIONS (CGWB PUNJAB):")
                facts.append("| Year | Average Depth (mbgl) | Observations |")
                facts.append("|------|----------------------|--------------|")
                for _, yr_r in yearly_avg.iterrows():
                    facts.append(f"| {int(yr_r['Year'])} | {yr_r['Average Depth (mbgl)']:.2f} | {int(yr_r['Observations']):,} |")

                supporting_df = yearly_avg[["Year", "Average Depth (mbgl)", "Observations"]].copy()

        # Case E: Overall summary of all Punjab districts
        else:
            sorted_by_mean = district_stats.sort_values("Mean_Depth", ascending=False).reset_index(drop=True)
            sorted_by_mean["Rank"] = range(1, len(sorted_by_mean) + 1)
            display_df = sorted_by_mean.head(10).copy()

            state_avg = (district_stats["Mean_Depth"] * district_stats["Observations"]).sum() / district_stats["Observations"].sum()

            facts.append(f"OVERALL SUMMARY OF PUNJAB GROUNDWATER ({len(district_stats)} DISTRICTS){year_filter_note}:")
            facts.append(f"- Total Monitoring Districts: {len(district_stats)}")
            facts.append(f"- Total Valid Monitoring Observations: {len(active_df):,}")
            facts.append(f"- Statewide Weighted Average Depth: {state_avg:.2f} mbgl")
            facts.append(f"- Most Depleted District: {sorted_by_mean.iloc[0]['District']} ({sorted_by_mean.iloc[0]['Average Depth (mbgl)']:.2f} mbgl)")
            facts.append(f"- Shallowest District: {sorted_by_mean.iloc[-1]['District']} ({sorted_by_mean.iloc[-1]['Average Depth (mbgl)']:.2f} mbgl)")
            facts.append(f"- Source Datasets: {', '.join(all_raw_files)}")

            facts.append("\nTOP DISTRICTS BY AVERAGE GROUNDWATER DEPTH:")
            facts.append("| Rank | District | Average Depth (mbgl) | Observations | Min (m) | Max (m) | Source Dataset |")
            facts.append("|------|----------|----------------------|--------------|---------|---------|----------------|")
            for _, r in display_df.iterrows():
                facts.append(f"| {int(r['Rank'])} | {r['District']} | {r['Average Depth (mbgl)']:.2f} | {int(r['Observations']):,} | {r['Min (m)']:.2f} | {r['Max (m)']:.2f} | {r['Source File']} |")

            facts.append(
                f"\nADDITIONAL VERIFIED NOTES:\n"
                f"- Hydrological Explanation of Negative Depths: Negative minimum values (e.g., Mukatsar -1.84 mbgl, Pathankot -2.19 mbgl) represent artesian conditions (piezometric head above ground level) or surface waterlogging.\n"
                f"- Shallowest District Overall: {display_df.iloc[0]['District']} has the shallowest water table in Punjab ({display_df.iloc[0]['Average Depth (mbgl)']:.2f} mbgl)."
            )

            supporting_df = display_df[[
                "Rank", "District", "Average Depth (mbgl)", "Observations", "Min (m)", "Max (m)", "Source File"
            ]].copy()

    facts_text = "\n".join(facts)
    return facts_text, supporting_df, sources


# ==============================================================================
# GENERAL KNOWLEDGE EXPLANATION (LLM & DETERMINISTIC FALLBACK)
# ==============================================================================

def generate_general_knowledge_answer_with_groq(query, lang="en", api_key=None):
    """
    Generate an educational, scientifically accurate response for general
    groundwater concepts (aquifers, water table, recharge, conservation) using Groq LLM.
    Clearly distinguishes general scientific knowledge from CGWB dataset findings.
    Responds in the user's language.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or not GROQ_AVAILABLE:
        return None, None, "Groq API key not provided or groq package unavailable."

    is_valid, active_model, err = verify_groq_api(key)
    if not is_valid:
        return None, None, err

    lang_instructions = {
        "hi": (
            "LANGUAGE REQUIREMENT: The user asked in HINDI. "
            "You MUST write your entire response in clear, grammatical, natural HINDI (Devanagari script). "
            "Use standard scientific Hindi terminology (जैसे: जलभृत/एक्विफर, भूजल स्तर/वाटर टेबल, पुनर्भरण, जल संरक्षण)।"
        ),
        "hinglish": (
            "LANGUAGE REQUIREMENT: The user asked in HINGLISH. "
            "You MUST respond in natural, conversational HINGLISH (Romanized Hindi) or bilingual Hindi/English matching the user's phrasing."
        ),
        "en": (
            "LANGUAGE REQUIREMENT: The user asked in ENGLISH. Respond in clear, professional ENGLISH."
        )
    }

    system_prompt = (
        "You are the AI Groundwater Information Assistant (developed for CSE276 AI Foundations).\n"
        "The user is asking a general scientific or hydrogeological knowledge question about groundwater "
        "(e.g., aquifers, water table, groundwater recharge, or groundwater conservation).\n\n"
        f"{lang_instructions.get(lang, lang_instructions['en'])}\n\n"
        "STRICT SCIENTIFIC GUIDELINES:\n"
        "1. Provide a scientifically sound, educational, and structured explanation of the concept.\n"
        "2. CLEAR SCIENTIFIC DISTINCTION: Start your response with the marker:\n"
        "   📌 **[सामान्य वैज्ञानिक ज्ञान / General Scientific Knowledge]**\n"
        "   Explicitly clarify that this explanation is based on general hydrogeological and hydrological science, "
        "   and is distinct from the empirical CGWB dataset (which contains field observation records for Punjab's 23 districts from 1991 to 2025).\n"
        "3. Do NOT invent dataset statistics or claim this scientific explanation was extracted from the Punjab CGWB monitoring CSVs.\n"
        "4. If the question pertains to groundwater conservation or protection, explain concrete hydrological practices: "
        "   rainwater harvesting, artificial recharge structures, crop diversification away from water-intensive crops, "
        "   and micro-irrigation (drip and sprinkler systems).\n"
        "5. Keep your tone objective, professional, and academically clear."
    )

    user_prompt = f"USER QUESTION: {query}\n\nPlease provide a clear, structured scientific explanation:"

    try:
        client = Groq(api_key=key)
        models_to_try = [active_model] + [m for m in PREFERRED_GROQ_MODELS if m != active_model]
        last_error = None

        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=700
                )
                answer = response.choices[0].message.content
                return answer, model_name, None
            except Exception as e:
                last_error = str(e)
                continue

        return None, None, f"Groq call failed: {last_error}"
    except Exception as e:
        return None, None, f"Groq client error: {str(e)}"


def generate_general_knowledge_fallback(query, lang="en"):
    """
    Deterministic fallback for general groundwater concepts if Groq is offline.
    Distinguishes general knowledge from CGWB dataset and responds in user's language.
    """
    q_low = query.lower()

    if lang == "hi":
        prefix = (
            "📌 **[सामान्य वैज्ञानिक ज्ञान / General Scientific Knowledge]**\n\n"
            "*(यह एक सामान्य जल-वैज्ञानिक व्याख्या है। यह पंजाब के 23 जिलों के लिए उपलब्ध CGWB डेटासेट के अवलोकन रिकॉर्ड से अलग है)।*\n\n"
        )
        if "aquifer" in q_low or "जलभृत" in q_low or "एक्विफर" in q_low:
            return prefix + (
                "### 🌊 भूजल विज्ञान अवधारणा: जलभृत (Aquifer)\n\n"
                "**जलभृत (Aquifer)** जमीन के नीचे पारगम्य चट्टान, रेत, या बजरी की वह भूगर्भीय परत है जो पानी को संग्रहित कर सकती है और कुओं या नलकूपों के माध्यम से पानी निकालने की अनुमति देती है।\n\n"
                "- **अप्रतिबंधित जलभृत (Unconfined Aquifer):** इसकी ऊपरी सीमा भूजल स्तर (Water Table) होती है, जो वायुमंडलीय दबाव में होती है। सतह से रिसने वाला वर्षा जल सीधे इसे भरता है।\n"
                "- **प्रतिबंधित जलभृत (Confined Aquifer):** यह दो अभेद्य चट्टानी परतों के बीच फंसा होता है, जहां पानी जलस्थैतिक दबाव में रहता है (इसे आर्टेशियन स्थिति भी कहते हैं)।\n\n"
                "💡 *पंजाब के जिलों (जैसे लुधियाना या पटियाला) के वास्तविक भूजल स्तर जानने के लिए आप जिला-वार प्रश्न पूछ सकते हैं।*"
            )
        elif "water table" in q_low or "जल स्तर" in q_low or "वाटर टेबल" in q_low:
            return prefix + (
                "### 💧 भूजल विज्ञान अवधारणा: जल स्तर (Water Table)\n\n"
                "**जल स्तर (Water Table)** जमीन के नीचे असंतृप्त क्षेत्र (Vadose Zone) और संतृप्त क्षेत्र (Saturated Zone) के बीच की विभाजन रेखा है।\n\n"
                "- जल स्तर के ऊपर मिट्टी और चट्टान के कणों के बीच हवा और नमी दोनों होते हैं।\n"
                "- जल स्तर के नीचे सभी छिद्र पूरी तरह से भूजल से भरे होते हैं।\n"
                "- भारत में केंद्रीय भूजल बोर्ड (CGWB) इसे **मीटर नीचे (mbgl - meters below ground level)** में मापता है। अधिक mbgl का अर्थ है पानी अधिक गहरा है।\n\n"
                "💡 *पंजाब के किसी विशिष्ट जिले में जल स्तर जानने के लिए आप उस जिले का नाम पूछ सकते हैं।*"
            )
        elif "recharge" in q_low or "पुनर्भरण" in q_low or "रिचार्ज" in q_low:
            return prefix + (
                "### 🌧️ भूजल विज्ञान अवधारणा: भूजल पुनर्भरण (Groundwater Recharge)\n\n"
                "**भूजल पुनर्भरण (Recharge)** वह जल-वैज्ञानिक प्रक्रिया है जिसमें सतह का पानी रिसकर भूमिगत जलभृतों तक पहुंचता है और भूजल स्तर को बढ़ाता है।\n\n"
                "- **प्राकृतिक पुनर्भरण:** वर्षा, पिघलती बर्फ, और नदियों व झीलों से पानी के प्राकृतिक रिसाव द्वारा होता है।\n"
                "- **कृत्रिम पुनर्भरण (Artificial Recharge):** मानव निर्मित संरचनाएं जैसे चेक डैम, पुनर्भरण कुएं (recharge shafts), और वर्षा जल संचयन (rainwater harvesting) संरचनाएं।\n\n"
                "💡 *पंजाब के सालाना भूजल बदलाव को देखने के लिए 'पंजाब का सालाना रुझान' पूछें।*"
            )
        elif "conservation" in q_low or "संरक्षण" in q_low or "बचाव" in q_low:
            return prefix + (
                "### 🛡️ भूजल विज्ञान अवधारणा: भूजल संरक्षण (Groundwater Conservation)\n\n"
                "**भूजल संरक्षण** जलभृतों के अत्यधिक दोहन को रोकने और जल संसाधनों को टिकाऊ बनाए रखने की वैज्ञानिक और नीतिगत प्रक्रिया है।\n\n"
                "**प्रमुख वैज्ञानिक उपाय:**\n"
                "1. **वर्षा जल संचयन (Rainwater Harvesting):** छतों और खुली सतहों से बारिश के पानी को सीधे जलभृतों में रिचार्ज करना।\n"
                "2. **सूक्ष्म सिंचाई तकनीक (Micro-Irrigation):** पारंपरिक बाढ़ सिंचाई के बजाय टपक (Drip) और फव्वारा (Sprinkler) सिंचाई का उपयोग, जिससे 40-60% पानी की बचत होती है।\n"
                "3. **फसल विविधीकरण (Crop Diversification):** धान जैसी अत्यधिक पानी खपत वाली फसलों के स्थान पर मक्का, दालें और तिलहन जैसी कम पानी वाली फसलें उगाना।\n"
                "4. **कृत्रिम पुनर्भरण संरचनाएं:** नालों और नदियों पर चेक डैम और परकोलेशन टैंक बनाना।"
            )
        else:
            return prefix + (
                "### 💧 भूजल विज्ञान अवधारणा: भूजल (Groundwater)\n\n"
                "**भूजल** वह ताजा पानी है जो पृथ्वी की सतह के नीचे मिट्टी के छिद्रों, दरारों और जलभृतों में स्थित होता है।\n\n"
                "- **स्रोत:** इसका मुख्य स्रोत वर्षा जल का जमीन में रिसना है।\n"
                "- **महत्व:** यह कृषि सिंचाई और पेयजल का सबसे बड़ा स्रोत है।\n"
                "- **निगरानी:** केंद्रीय भूजल बोर्ड (CGWB) पंजाब के 23 जिलों में जल स्तर की निरंतर निगरानी करता है।"
            )

    elif lang == "hinglish":
        prefix = (
            "📌 **[General Scientific Knowledge / Samanya Vigyan]**\n\n"
            "*(Yeh ek general hydrological explanation hai, jo empirical CGWB dataset ke records se alag hai).*\n\n"
        )
        if "aquifer" in q_low or "जलभृत" in q_low:
            return prefix + (
                "### 🌊 Hydrogeology Concept: Aquifer\n\n"
                "**Aquifer** zameen ke neeche permeable rock, sand, ya gravel ki layer hoti hai jo paani store karti hai aur tubewells/wells ke zariye paani supply karti hai.\n\n"
                "- **Unconfined Aquifer:** Iski upper boundary water table hoti hai, jisme surface rainwater directly recharge hota hai.\n"
                "- **Confined Aquifer:** Yeh do impermeable layers ke beech fasa hota hai aur hydrostatic pressure ke under rehta hai."
            )
        elif "conservation" in q_low or "sanrakshan" in q_low:
            return prefix + (
                "### 🛡️ Hydrogeology Concept: Groundwater Conservation\n\n"
                "**Groundwater Conservation** aquifers ko deplete hone se bachane ka tarika hai:\n\n"
                "1. **Rainwater Harvesting:** Chhat aur runoff water ko recharge wells mein daalna.\n"
                "2. **Drip/Sprinkler Irrigation:** Paani bachane ke liye micro-irrigation ka upayog.\n"
                "3. **Crop Diversification:** Water-intensive crops (jaise paddy) ki jagah kam paani wali faslein lagana."
            )
        else:
            return prefix + (
                "### 💧 Hydrogeology Concept: Groundwater & Water Table\n\n"
                "**Groundwater** zameen ke neeche saturated rocks aur sand mein jama fresh water hota hai. Iska level **meters below ground level (mbgl)** mein measure hota hai.\n\n"
                "Punjab ke district-wise records dekhne ke liye kisi specific district (e.g. Ludhiana) ke baare mein poochein."
            )

    else:
        prefix = (
            "📌 **[General Scientific Knowledge]**\n\n"
            "*(Note: This explanation is based on general hydrogeological principles and is distinct from the empirical Punjab CGWB monitoring dataset).* \n\n"
        )
        if "aquifer" in q_low:
            return prefix + (
                "### 🌊 General Hydrogeology Concept: Aquifer\n\n"
                "An **aquifer** is an underground geological formation of permeable rock, sand, or gravel capable of storing and yielding significant quantities of groundwater.\n\n"
                "- **Unconfined Aquifer:** An aquifer whose upper boundary is the water table at atmospheric pressure. Water can infiltrate directly from the ground surface.\n"
                "- **Confined (Artesian) Aquifer:** An aquifer sandwiched between impermeable geological layers (aquitards/aquicludes) where groundwater is under hydrostatic pressure.\n\n"
                "*(For empirical monitoring records across Punjab districts, you can query specific districts like Ludhiana or Bathinda).*"
            )
        elif "water table" in q_low:
            return prefix + (
                "### 💧 General Hydrogeology Concept: Water Table\n\n"
                "The **water table** is the boundary underground between the unsaturated (vadose) zone and the saturated zone.\n\n"
                "- Above the water table, pore spaces between soil and rock contain both air and water.\n"
                "- Below the water table, all pore spaces are completely saturated with groundwater.\n"
                "- In hydrogeological surveys, the water table level is typically measured in **meters below ground level (mbgl)**.\n\n"
                "*(To check actual water table depths for Punjab districts, ask about a specific district or Punjab rankings).*"
            )
        elif "recharge" in q_low:
            return prefix + (
                "### 🌧️ General Hydrogeology Concept: Groundwater Recharge\n\n"
                "**Groundwater recharge** (or deep percolation) is the hydrologic process where water moves downward from surface water to groundwater.\n\n"
                "- **Natural Recharge:** Occurs through precipitation, snowmelt, and infiltration from rivers, lakes, and wetlands.\n"
                "- **Artificial Recharge:** Engineered interventions (such as check dams, recharge shafts, and rainwater harvesting structures) to replenish depleted aquifers."
            )
        elif "conservation" in q_low or "save" in q_low:
            return prefix + (
                "### 🛡️ General Hydrogeology Concept: Groundwater Conservation\n\n"
                "**Groundwater conservation** encompasses scientific, agricultural, and engineering strategies to prevent aquifer depletion and secure sustainable yields:\n\n"
                "1. **Rainwater Harvesting:** Capturing runoff from rooftops and paved surfaces to recharge local aquifers.\n"
                "2. **Micro-Irrigation (Drip and Sprinkler):** Applying water directly to the crop root zone, improving water application efficiency up to 80–90%.\n"
                "3. **Crop Diversification:** Shifting acreage from high water-footprint crops (e.g., summer paddy) to less water-intensive pulses, maize, or millets.\n"
                "4. **Regulated Extraction:** Limiting over-pumping of deep tube wells and constructing artificial recharge shafts."
            )
        else:
            return prefix + (
                "### 💧 General Hydrogeology Concept: Groundwater\n\n"
                "**Groundwater** is fresh water held underground in the soil or in pores, crevices, and permeable formations beneath the Earth's surface.\n\n"
                "- **Origin:** Primarily originates from atmospheric precipitation that infiltrates the soil and percolates downward under gravity.\n"
                "- **Storage:** Maintained in underground geological units known as **aquifers**.\n"
                "- **Monitoring:** Central Ground Water Board (CGWB) monitors groundwater levels as **depth to water table in meters below ground level (mbgl)**."
            )


# ==============================================================================
# WATER QUALITY & DEPTH VS QUALITY SCIENTIFIC MODULE
# ==============================================================================

def is_groundwater_quality_query(query):
    """
    Detect if the user query is asking about chemical groundwater quality parameters
    (nitrate, fluoride, arsenic, salinity, TDS, pH, contamination, potability)
    or comparing groundwater depth vs groundwater quality.
    """
    q_low = query.lower()
    quality_terms = [
        "nitrate", "nitrates", "fluoride", "arsenic", "salinity", "saline",
        "tds", "total dissolved solids", "ph level", "ph value", "heavy metal",
        "heavy metals", "chemical quality", "water quality", "chemical parameter",
        "chemical parameters", "contamination", "contaminant", "pollutant",
        "pollution", "potability", "potable", "safe to drink", "drinking water quality",
        "uranium", "lead", "iron", "नाइट्रेट", "फ्लोराइड", "आर्सेनिक", "गुणवत्ता",
        "जल गुणवत्ता", "पीने का पानी", "प्रदूषण", "रसायन"
    ]
    depth_vs_quality_terms = [
        "depth vs quality", "quality vs depth", "depth and quality",
        "depth or quality", "mbgl vs mg/l", "mbgl or mg/l",
        "difference between groundwater depth and groundwater quality",
        "difference between depth and quality", "depth and water quality",
        "gaharai aur gunvatta", "गहराई और गुणवत्ता", "गहराई बनाम गुणवत्ता"
    ]
    if any(term in q_low for term in depth_vs_quality_terms):
        return True
    return any(re.search(r"\b" + re.escape(term) + r"\b", q_low) for term in quality_terms)


def process_groundwater_quality_query(query, lang="en", api_key=None):
    """
    Provide grounded, scientifically authoritative answers for water quality and
    depth vs quality queries, explicitly clarifying dataset boundaries (the CGWB manual CSV
    measures physical depth in mbgl, NOT chemical quality in mg/L), explaining Punjab agricultural
    nitrate context and BIS limits, and offering structured comparison tables.
    """
    q_low = query.lower()
    is_depth_vs_quality = any(t in q_low for t in [
        "depth vs quality", "quality vs depth", "difference", "compare", "versus", "vs",
        "bhed", "antar", "अंतर", "तुलना", "mbgl vs mg/l", "mbgl or mg/l"
    ])

    # Side-by-side comparison data
    comparison_data = [
        {
            "Parameter / Aspect": "Core Scientific Concept",
            "Groundwater Depth (भूजल गहराई)": "Physical vertical distance from the ground surface down to the water table (unsaturated/saturated zone interface).",
            "Groundwater Quality (भूजल गुणवत्ता)": "Chemical, physical, and biological composition and purity of the water determining safety for drinking and irrigation."
        },
        {
            "Parameter / Aspect": "Standard Measurement Unit",
            "Groundwater Depth (भूजल गहराई)": "mbgl (meters below ground level) or meters (m).",
            "Groundwater Quality (भूजल गुणवत्ता)": "mg/L (milligrams per liter), ppm, pH scale, or µS/cm (Electrical Conductivity)."
        },
        {
            "Parameter / Aspect": "Measurement Instrument",
            "Groundwater Depth (भूजल गहराई)": "Electric contact water level sounder, steel measuring tape, or downhole piezometric pressure sensor.",
            "Groundwater Quality (भूजल गुणवत्ता)": "Chemical field kits, Spectrophotometer, Ion Chromatography (IC), or Inductively Coupled Plasma Mass Spectrometry (ICP-MS)."
        },
        {
            "Parameter / Aspect": "Key Questions Answered",
            "Groundwater Depth (भूजल गहराई)": "How much energy/lift is needed for tubewells? Is the regional water table rising or depleting over time?",
            "Groundwater Quality (भूजल गुणवत्ता)": "Is the water fit for human consumption? Does it exceed permissible limits for nitrate, fluoride, arsenic, or salinity?"
        },
        {
            "Parameter / Aspect": "Presence in this Application's CSV Dataset",
            "Groundwater Depth (भूजल गहराई)": "YES: 36,934 valid observations across 22 Punjab districts from CGWB manual monitoring (1996–2024 active series).",
            "Groundwater Quality (भूजल गुणवत्ता)": "NO: The manual quarterly dataset strictly monitors physical depth (mbgl). It contains zero chemical quality columns."
        },
        {
            "Parameter / Aspect": "Official Authoritative Source",
            "Groundwater Depth (भूजल गहराई)": "CGWB Manual Quarterly Groundwater Level Monitoring Network.",
            "Groundwater Quality (भूजल गुणवत्ता)": "CGWB Ground Water Quality Yearbooks & BIS: IS 10500:2012 Drinking Water Specification."
        }
    ]
    comparison_df = pd.DataFrame(comparison_data)

    if is_depth_vs_quality:
        if lang == "hi":
            text = (
                "### 🔍 भूजल गहराई (Depth) बनाम भूजल गुणवत्ता (Quality) में अंतर\n\n"
                "भूजल अध्ययन में **गहराई (Depth)** और **गुणवत्ता (Quality)** दो अलग-अलग वैज्ञानिक आयाम हैं:\n\n"
                "| पहलू / आयाम | भूजल गहराई (Groundwater Depth) | भूजल गुणवत्ता (Groundwater Quality) |\n"
                "|---|---|---|\n"
                "| **वैज्ञानिक परिभाषा** | जमीन की सतह से नीचे जल स्तर (Water Table) की ऊर्ध्वाधर दूरी। | पानी की रासायनिक, भौतिक और जैविक शुद्धता व पीने/सिंचाई हेतु सुरक्षा। |\n"
                "| **मापक इकाई** | **mbgl (मीटर नीचे - meters below ground level)** या मीटर | **mg/L (मिलीग्राम प्रति लीटर)**, ppm, pH मान, µS/cm |\n"
                "| **मापन उपकरण** | इलेक्ट्रिक साउंडर टेप, ऑटोमैटिक वाटर लेवल रिकॉर्डर (AWLR) | प्रयोगशाला रासायनिक विश्लेषण (स्पेक्ट्रोफोटोमीटर, आयन क्रोमैटोग्राफी) |\n"
                "| **प्रमुख प्रभाव** | नलकूपों की लिफ्टिंग लागत, भूजल भंडार में कमी या वृद्धि | स्वास्थ्य जोखिम, पीने की सुरक्षा, फसलों में लवणता |\n"
                "| **इस डेटासेट में स्थिति** | **उपलब्ध है:** CGWB पंजाब के 22 जिलों के 36,934 भौतिक अवलोकन रिकॉर्ड (1996–2024)। | **अनुपस्थित:** वर्तमान त्रैमासिक CSV डेटासेट में केवल भौतिक गहराई मापी गई है, रासायनिक गुणवत्ता नहीं। |\n\n"
                "📌 **महत्वपूर्ण निष्कर्ष:** गहरा भूजल आवश्यक रूप से दूषित नहीं होता, और उथला भूजल आवश्यक रूप से पीने योग्य नहीं होता (उथले जलभृतों में कृषि उर्वरकों से नाइट्रेट का रिसाव अधिक होता है)।"
            )
        elif lang == "hinglish":
            text = (
                "### 🔍 Groundwater Depth vs Groundwater Quality: Difference Explained\n\n"
                "Groundwater science mein **Depth (Gaharai)** aur **Quality (Gunvatta)** do alag dimensions hain:\n\n"
                "| Aspect | Groundwater Depth | Groundwater Quality |\n"
                "|---|---|---|\n"
                "| **Definition** | Zameen ke surface se water table tak ki vertical doori. | Paani ka chemical aur biological purity level. |\n"
                "| **Unit** | **mbgl (meters below ground level)** ya meters | **mg/L (milligrams per liter)**, ppm, pH |\n"
                "| **Measurement** | Electric water sounder tape, piezometer sensor | Chemical lab testing (Ion Chromatography, ICP-MS) |\n"
                "| **Purpose** | Pumping cost, aquifer depletion ya recharge check karna | Drinking safety, toxicity (nitrate, arsenic, fluoride) check karna |\n"
                "| **In This Dataset** | **AVAILABLE:** Punjab CGWB ke 36,934 valid depth records (1996–2024). | **NOT IN THIS CSV:** Yeh dataset sirf physical depth track karta hai, chemical parameters nahi. |\n\n"
                "💡 *Note: Authoritative chemical quality data CGWB State Quality Yearbooks mein alag se publish hota hai.*"
            )
        else:
            text = (
                "### 🔍 Difference Between Groundwater Depth and Groundwater Quality\n\n"
                "In hydrogeology and water resources engineering, **Groundwater Depth** and **Groundwater Quality** represent two distinct physical and chemical domains:\n\n"
                "| Aspect / Dimension | Groundwater Depth | Groundwater Quality |\n"
                "|---|---|---|\n"
                "| **Core Concept** | Vertical distance from the ground surface down to the water table (top of saturated zone). | Chemical, physical, and microbiological composition and potability of the water. |\n"
                "| **Standard Unit** | **mbgl** (meters below ground level) or meters ($m$). | **mg/L** (milligrams per liter), ppm, pH scale, or $\\mu S/cm$. |\n"
                "| **Measurement Method** | Electric contact water level sounder, steel graduated tape, or acoustic/pressure piezometers. | Laboratory analytical testing (Spectrophotometry, Ion Chromatography, ICP-MS). |\n"
                "| **Practical Implication** | Pumping lift requirements, energy costs, and aquifer storage depletion. | Human drinking safety, toxicity, infant health risks, and agricultural soil salinity. |\n"
                "| **Status in this Dataset** | **AVAILABLE**: 36,934 valid records across 22 canonical Punjab districts (1996–2024). | **NOT IN THIS CSV**: This manual quarterly monitoring dataset records physical depth only. |\n\n"
                "📌 **Hydrogeological Relationship:**\n"
                "- A district with a deep water table (e.g., Sangrur at 26.10 mbgl) reflects severe volumetric depletion due to over-extraction.\n"
                "- Conversely, shallow water tables (e.g., Mukatsar at 2.92 mbgl) can experience waterlogging, secondary soil salinization, or elevated surface contaminant infiltration.\n"
                "- Authoritative chemical quality data is published separately in the **CGWB Ground Water Quality Yearbooks** by the Ministry of Jal Shakti."
            )
        sources = [
            "Central Ground Water Board (CGWB) Punjab Monitoring Network",
            "Bureau of Indian Standards (BIS: IS 10500:2012 Drinking Water Specification)",
            "CGWB Hydrogeological Terminology Guidelines"
        ]
        return {
            "answer": text,
            "engine": "Hydrogeological Domain Knowledge (Depth vs Quality Framework)",
            "intent": "quality_vs_depth",
            "sources": sources,
            "records": comparison_df,
            "figure": None
        }

    else:
        # Specific chemical parameter query (e.g. Nitrate concentration in Punjab)
        nitrate_records = pd.DataFrame([
            {"Parameter": "Nitrate (as NO3)", "Standard Unit": "mg/L (milligrams per liter)", "BIS Permissible Limit": "45.0 mg/L", "Primary Punjab Source": "Synthetic nitrogen fertilizer (urea, DAP) leaching in wheat-paddy rotations", "Dataset Coverage": "Not measured in manual depth CSV (available in CGWB Quality Yearbooks)"},
            {"Parameter": "Fluoride (as F)", "Standard Unit": "mg/L", "BIS Permissible Limit": "1.0 - 1.5 mg/L", "Primary Punjab Source": "Geogenic mineral dissolution in southwestern Punjab (Bathinda, Mansa, Fazilka)", "Dataset Coverage": "Not measured in manual depth CSV"},
            {"Parameter": "Arsenic (as As)", "Standard Unit": "mg/L", "BIS Permissible Limit": "0.01 mg/L", "Primary Punjab Source": "Localized alluvial floodplains (Majha region / Ravi-Beas-Satluj belts)", "Dataset Coverage": "Not measured in manual depth CSV"},
            {"Parameter": "Physical Depth", "Standard Unit": "mbgl (meters below ground level)", "BIS Permissible Limit": "N/A (Physical dimension)", "Primary Punjab Source": "Central Ground Water Board (CGWB) manual quarterly monitoring network", "Dataset Coverage": "Fully present (36,934 valid records across 22 districts)"}
        ])

        if lang == "hi":
            text = (
                "### 🧪 पंजाब में भूजल गुणवत्ता एवं नाइट्रेट (Nitrate) सांद्रता\n\n"
                "⚠️ **डेटासेट की स्पष्ट सीमा (Dataset Boundary):**\n"
                "इस एप्लिकेशन में लोड किया गया केंद्रीय भूजल बोर्ड (CGWB) का डेटासेट पंजाब के 22 जिलों में **भूजल की भौतिक गहराई (meters below ground level - mbgl)** का रिकॉर्ड रखता है। इसमें रासायनिक जल गुणवत्ता (जैसे नाइट्रेट, आर्सेनिक, फ्लोराइड या pH) के अवलोकन शामिल नहीं हैं।\n\n"
                "**पंजाब में नाइट्रेट सांद्रता का जल-वैज्ञानिक संदर्भ (Hydrogeological Context):**\n"
                "1. **नाइट्रेट का मुख्य स्रोत:** पंजाब में गेहूं-धान के सघन फसल चक्र के कारण रासायनिक नाइट्रोजन उर्वरकों (यूरिया और डीएपी) का भारी उपयोग होता है। अतिरिक्त उर्वरक सिंचाई और वर्षा के पानी के साथ रिसकर उथले जलभृतों (shallow unconfined aquifers) में नाइट्रेट ($NO_3^-$) के रूप में जमा हो जाता है।\n"
                "2. **मानक अनुमेय सीमा (Permissible Limit):** भारतीय मानक ब्यूरो (**BIS IS 10500:2012**) और विश्व स्वास्थ्य संगठन (WHO) के अनुसार पीने के पानी में नाइट्रेट की अधिकतम अनुमेय सीमा **45 mg/L** है।\n"
                "3. **प्रभावित क्षेत्र:** CGWB की भूजल गुणवत्ता रिपोर्टों के अनुसार, पंजाब के कई केंद्रीय और दक्षिण-पश्चिमी जिलों (जैसे बठिंडा, मनसा, संगरूर, पटियाला) के उथले नलकूपों में नाइट्रेट की मात्रा 45 mg/L की सीमा से अधिक दर्ज की गई है।\n"
                "4. **स्वास्थ्य प्रभाव:** नाइट्रेट युक्त पानी पीने से शिशुओं में 'ब्लू बेबी सिंड्रोम' (Methemoglobinemia) और वयस्कों में थायराइड विकार का जोखिम होता है।\n\n"
                "📚 **प्रामाणिक स्रोत:** विस्तृत रासायनिक आंकड़ों के लिए जल शक्ति मंत्रालय द्वारा प्रकाशित **'CGWB Ground Water Quality Year Book (Punjab State)'** का संदर्भ लें।"
            )
        elif lang == "hinglish":
            text = (
                "### 🧪 Punjab Groundwater Quality & Nitrate (NO3) Concentration\n\n"
                "⚠️ **Dataset Boundary Clarification:**\n"
                "Active CGWB Punjab dataset mein **physical groundwater depth** (mbgl mein) di gayi hai. Is CSV dataset mein chemical water quality parameters (jaise Nitrate, Fluoride, Arsenic, TDS) shamil nahi hain.\n\n"
                "**Punjab mein Nitrate ka Hydrogeological Context:**\n"
                "- **Source:** Wheat-paddy intensive cropping mein chemical nitrogen fertilizers (Urea, DAP) ka heavy use shallow aquifers mein nitrate ($NO_3^-$) leach karta hai.\n"
                "- **Permissible Limit:** Bureau of Indian Standards (**BIS IS 10500:2012**) ke anusaar drinking water mein Nitrate ki safe limit **45 mg/L** hai.\n"
                "- **Affected Districts:** CGWB Quality Yearbooks ke mutabiq, southwestern aur central Punjab (Bathinda, Mansa, Sangrur, Patiala) ke shallow tubewells mein nitrate 45 mg/L se exceed hota hai.\n"
                "- **Health Risks:** High nitrate paani peene se infants mein *Methemoglobinemia* ('Blue Baby Syndrome') ho sakta hai.\n\n"
                "📚 *Authentic chemical records ke liye Ministry of Jal Shakti ki 'CGWB Ground Water Quality Year Book' refer karein.*"
            )
        else:
            text = (
                "### 🧪 Groundwater Quality & Nitrate ($NO_3^-$) Concentration in Punjab\n\n"
                "⚠️ **Dataset Scope & Boundary Clarification:**\n"
                "The active Central Ground Water Board (CGWB) manual quarterly monitoring CSV dataset powering this system records **physical depth to water table** (measured in meters below ground level - mbgl) across 22 canonical districts. It **does not contain chemical water quality observations** (such as nitrate, arsenic, fluoride, pH, or heavy metals).\n\n"
                "**Hydrogeological Context & Scientific Facts for Punjab:**\n"
                "1. **Agrochemical Leaching Origin:** High nitrate concentrations in Punjab's groundwater originate primarily from intensive agricultural practices. Punjab has among the highest per-hectare chemical fertilizer application rates in India; excess synthetic nitrogen fertilizers (urea and DAP) applied during intensive wheat-paddy crop rotations dissolve and leach through permeable alluvial soils into shallow unconfined aquifers.\n"
                "2. **Regulatory Permissible Limit:** Under the Bureau of Indian Standards drinking water specification (**BIS: IS 10500:2012**) and WHO guidelines, the maximum acceptable limit for nitrate ($NO_3^-$) in drinking water is **45 mg/L**.\n"
                "3. **Geographical Distribution in Punjab:** According to specialized CGWB Ground Water Quality Yearbooks, shallow aquifer samples exceeding the 45 mg/L BIS limit have been documented across several central and southwestern districts, including **Bathinda, Mansa, Sangrur, Barnala, and Patiala**.\n"
                "4. **Health Hazards:** Ingesting groundwater with nitrate concentrations above 45 mg/L can induce infantile methemoglobinemia (*'blue baby syndrome'*), impair thyroid function, and present chronic health hazards.\n"
                "5. **Authoritative Quality Reports:** For complete laboratory-tested water quality monitoring tables, consult the official **CGWB Ground Water Quality in Shallow Aquifers of India / Punjab State Water Quality Yearbooks** published by the Ministry of Jal Shakti."
            )

        sources = [
            "CGWB Ground Water Quality Yearbooks (Ministry of Jal Shakti, Government of India)",
            "Bureau of Indian Standards (BIS: IS 10500:2012 Drinking Water Specification)",
            "Central Ground Water Board (CGWB) Punjab Monitoring Network"
        ]
        return {
            "answer": text,
            "engine": "Hydrogeological Quality Knowledge Base (Verified BIS Standards)",
            "intent": "water_quality",
            "sources": sources,
            "records": nitrate_records,
            "figure": None
        }


# ==============================================================================
# GROUNDED LLM GENERATION & TARGETED FALLBACK
# ==============================================================================

def generate_grounded_answer_with_groq(query, context, lang="en", api_key=None):
    """
    Generate an answer using Groq LLM strictly grounded in the supplied context.
    Responds in the user's language (Hindi, Hinglish, or English).
    Returns: (answer_text, model_name, error_message)
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or not GROQ_AVAILABLE:
        return None, None, "Groq API key not provided or groq package unavailable."

    is_valid, active_model, err = verify_groq_api(key)
    if not is_valid:
        return None, None, err

    lang_instructions = {
        "hi": (
            "LANGUAGE REQUIREMENT: The user asked in HINDI. "
            "You MUST respond completely in natural, grammatical, professional HINDI (Devanagari script). "
            "Use standard Hindi words like 'मीटर नीचे (mbgl)', 'औसत भूजल स्तर', 'अवलोकन संख्या', 'निगरानी स्टेशन'।"
        ),
        "hinglish": (
            "LANGUAGE REQUIREMENT: The user asked in HINGLISH. "
            "You MUST respond in natural, conversational HINGLISH (Romanized Hindi) or bilingual Hindi/English matching the user's question."
        ),
        "en": (
            "LANGUAGE REQUIREMENT: The user asked in ENGLISH. Respond in objective, professional ENGLISH."
        )
    }

    system_prompt = (
        "You are the AI Groundwater Information Assistant for Punjab, India (developed for CSE276 AI Foundations).\n\n"
        f"{lang_instructions.get(lang, lang_instructions['en'])}\n\n"
        "STRICT GROUNDING INSTRUCTIONS:\n"
        "1. Answer the user's specific question using ONLY the provided verified context from Central Ground Water Board (CGWB) records.\n"
        "2. Do NOT invent, assume, or extrapolate statistics not stated in the context.\n"
        "3. Hydrogeological context: Groundwater level is measured in 'meters below ground level' (mbgl). "
        "A higher measurement means the water table is deeper underground (more depleted). "
        "A lower measurement means water is closer to the surface.\n"
        "4. HYDROLOGICAL EXPLANATIONS:\n"
        "   - Negative mbgl readings indicate artesian conditions (groundwater pressure head rising above ground level) or surface waterlogged areas. When negative depths appear, explain this context clearly.\n"
        "   - Extreme measurements (>100 mbgl, such as Sangrur 268m or Jalandhar 250m) represent specialized deep exploratory piezometer observations or localized deep cones of depression. Do not treat them as corrupted data.\n"
        "   - Active Monitoring Window: Observations in this dataset strictly span 1996 to 2024 (from the CGWB 1991–2025 series).\n"
        "5. Always mention the source datasets, observation count, and the specific numbers provided in the context.\n"
        "6. If the context confirms that data is available for a requested district, explicitly state that data IS available and provide the exact metrics.\n"
        "7. If the context notes data is unavailable, state clearly and transparently that it is unavailable.\n"
        "8. Keep your tone direct, objective, helpful, and academically sound."
    )

    user_prompt = (
        f"USER QUESTION: {query}\n\n"
        f"VERIFIED DATA CONTEXT FROM CGWB DATASET:\n{context}\n\n"
        f"Please provide a direct, question-specific, and source-grounded response in the user's language:"
    )

    try:
        client = Groq(api_key=key)
        models_to_try = [active_model] + [m for m in PREFERRED_GROQ_MODELS if m != active_model]
        last_error = None

        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=650
                )
                answer = response.choices[0].message.content
                return answer, model_name, None
            except Exception as e:
                last_error = str(e)
                continue

        return None, None, f"Groq model call failed: {last_error}"

    except Exception as e:
        return None, None, f"Groq client error: {str(e)}"


def generate_targeted_fallback_answer(query, context, lang="en"):
    """
    Generate a question-specific deterministic answer when Groq LLM is not available.
    Produces answers in user's language (Hindi, Hinglish, or English).
    """
    if not context or context.strip() == "":
        if lang == "hi":
            return "आपके प्रश्न के लिए CGWB डेटासेट में कोई संबंधित भूजल रिकॉर्ड या आंकड़े नहीं मिले।"
        elif lang == "hinglish":
            return "Aapki query ke liye CGWB dataset mein koi relevant groundwater records nahi mile."
        else:
            return "No corresponding groundwater records or statistics were found in the CGWB dataset for your query."

    if lang == "hi":
        return (
            f"### 💧 भूजल निष्कर्ष (CGWB पंजाब डेटासेट)\n\n"
            f"आधिकारिक केंद्रीय भूजल बोर्ड (CGWB) डेटासेट रिकॉर्ड के आधार पर आपके प्रश्न का उत्तर:\n\n"
            f"{context}\n\n"
            f"*(नोट: भूजल स्तर **मीटर नीचे (mbgl - meters below ground level)** में मापा जाता है। "
            f"अधिक मान अधिक गहराई / कमी को दर्शाता है। ऋणात्मक मान (-mbgl) आर्टेशियन स्थितियों या जलभराव को दर्शाते हैं जहां जल स्तर जमीन से ऊपर होता है। "
            f"सत्यापित अवलोकन 1996 से 2024 तक के हैं)।*"
        )
    elif lang == "hinglish":
        return (
            f"### 💧 Groundwater Findings (CGWB Punjab Dataset)\n\n"
            f"Central Ground Water Board (CGWB) records ke mutabiq aapke sawal ka verified answer:\n\n"
            f"{context}\n\n"
            f"*(Note: Groundwater level **meters below ground level (mbgl)** mein record hota hai. "
            f"Bada number deeper water table ko denote karta hai. Negative values artesian conditions ya waterlogging denote karti hain. "
            f"Verified observations 1996 se 2024 tak hain).* "
        )
    else:
        return (
            f"### 💧 Groundwater Findings (CGWB Punjab Dataset)\n\n"
            f"Based on the official Central Ground Water Board (CGWB) dataset records, here is the answer to your query:\n\n"
            f"{context}\n\n"
            f"*(Note: Groundwater levels are recorded in **meters below ground level (mbgl)**. "
            f"Higher numbers denote greater water table depth / depletion. Negative readings represent artesian conditions or surface waterlogging. "
            f"Verified observation records span 1996 to 2024).* "
        )


# ==============================================================================
# MAIN QUERY ORCHESTRATOR
# ==============================================================================

def answer_groundwater_query(query, raw_df, corpus, district_summary_df, yearly_summary_df, model, embeddings, api_key=None):
    """
    End-to-End Orchestrator:
    1. Detect query language (Hindi, Hinglish, English).
    2. Scope Guardrail: Checks for non-Punjab regions or invalid years.
    3. Village-Level Analysis: Exact village statistics, disambiguation, zero-hallucination guardrail, and yearly trend graph.
    4. General Hydrological Knowledge: Concept definitions (aquifers, water table, recharge, conservation).
    5. Domain Guardrail: Politeness decline for unrelated queries (e.g., recipe, sports).
    6. District-Level Analysis & Graphs: Mathematical statistics, rankings, multi-district comparisons, and Plotly graphs.
    7. Grounded LLM Generation: Strictly grounded synthesis using Groq in user's language.
    """
    lang = detect_language(query)
    known_districts = sorted(raw_df["District"].dropna().unique().tolist())
    districts_set = set(known_districts)

    # Extract time range filter if specified in query
    start_year, end_year = extract_year_range_from_query(query)

    # 1. Geographic & Temporal Scope Check
    is_out, out_msg = check_out_of_scope(query, known_districts, lang=lang)
    if is_out:
        return {
            "answer": out_msg,
            "engine": "Scope Guardrail (Out of Scope Query)",
            "intent": "out_of_scope",
            "sources": ["CGWB Dataset Scope: Punjab (1996-2024 Valid Observations)"],
            "records": pd.DataFrame(),
            "figure": None
        }

    # 2. Check for Village-Level Query (Explicit or Implicit)
    village_meta, sorted_villages, valid_v_df = get_village_metadata(raw_df)
    village_det = detect_village_in_query(query, village_meta, sorted_villages, districts_set)

    if village_det["status"] in ["found", "not_found"]:
        return process_village_groundwater_query(
            query=query,
            raw_df=raw_df,
            village_det=village_det,
            village_meta=village_meta,
            valid_v_df=valid_v_df,
            lang=lang,
            api_key=api_key
        )

    # 2.5 Check for Groundwater Quality or Depth vs Quality Query
    if is_groundwater_quality_query(query):
        return process_groundwater_quality_query(query, lang=lang, api_key=api_key)

    # 3. Check for General Scientific Groundwater Knowledge (Aquifer, Water Table, Recharge, Conservation)
    if is_general_knowledge_query(query):
        llm_answer, used_model, err_msg = generate_general_knowledge_answer_with_groq(query, lang=lang, api_key=api_key)
        if llm_answer:
            final_answer = llm_answer
            engine = f"Groq ({used_model}) - General Hydrological Knowledge"
        else:
            final_answer = generate_general_knowledge_fallback(query, lang=lang)
            if api_key:
                engine = f"General Hydrological Knowledge Base (Groq API Error: {err_msg})"
            else:
                engine = "General Hydrological Knowledge Base (Offline Mode)"

        return {
            "answer": final_answer,
            "engine": engine,
            "intent": "general_knowledge",
            "sources": ["General Hydrogeological Science & CGWB Concepts"],
            "records": pd.DataFrame(),
            "figure": None
        }

    # 4. Check Semantic Relevance to Groundwater Domain (Multilingual Guardrail)
    search_matches = search_groundwater(query, corpus, model, embeddings, top_k=6)
    max_sim = search_matches["Similarity"].max() if not search_matches.empty else 0.0

    if not is_groundwater_related(query, max_similarity=max_sim):
        if lang == "hi":
            decline_msg = (
                "मैं **पंजाब के लिए AI भूजल सूचना सहायक** हूँ। मेरा कार्य पंजाब में भूजल जलविज्ञान और "
                "केंद्रीय भूजल बोर्ड (CGWB) के निगरानी रिकॉर्ड से जुड़े प्रश्नों के उत्तर देना है।\n\n"
                "आपका प्रश्न भूजल या पंजाब जलविज्ञान के दायरे से बाहर प्रतीत होता है। "
                "कृपया भूजल से संबंधित प्रश्न पूछें (जैसे: *'भूजल क्या है?'*, "
                "*'लुधियाना में औसत भूजल स्तर क्या है?'*, या *'पंजाब में सबसे गहरा जिला कौन सा है?'*)।"
            )
        elif lang == "hinglish":
            decline_msg = (
                "Main **Punjab ke liye AI Groundwater Information Assistant** hoon. "
                "Aapka sawaal groundwater hydrology ya CGWB dataset se related nahi lagta.\n\n"
                "Kripya groundwater se juda sawaal poochein (jaise: *'Ludhiana me groundwater level kitna hai?'*, "
                "*'Aquifer kya hota hai?'*, ya *'Punjab ka yearly trend dikhao'*)."
            )
        else:
            decline_msg = (
                "I am the **AI Groundwater Information Assistant for Punjab**, designed to answer "
                "questions strictly about groundwater hydrology and Central Ground Water Board (CGWB) "
                "monitoring records in Punjab.\n\n"
                "Your question appears to be outside the scope of groundwater hydrology. "
                "Please ask a groundwater-related question (for example: *'What is groundwater?'*, "
                "*'What is the average groundwater level in Ludhiana?'*, or *'Which district has the highest groundwater measurement?'*)."
            )
        return {
            "answer": decline_msg,
            "engine": "Domain Guardrail (Unrelated Query)",
            "intent": "unrelated",
            "sources": [],
            "records": pd.DataFrame(),
            "figure": None
        }

    # 5. Intent Routing: Analytics / District Trend Graphs vs Station Lookup
    intent = detect_query_intent(query)
    sources = []
    supporting_df = pd.DataFrame()
    context_text = ""
    mentioned_districts = extract_districts_from_query(query, known_districts)

    # Generate Plotly graph for district queries with exact time range matching
    fig = generate_district_trend_graph(query, raw_df, mentioned_districts, start_year=start_year, end_year=end_year)

    if intent == "analytics" or bool(mentioned_districts):
        context_text, supporting_df, sources = compute_analytical_facts(
            query, raw_df, district_summary_df, yearly_summary_df, start_year=start_year, end_year=end_year
        )
    else:
        # Factual station record search
        if max_sim < 0.22:
            if lang == "hi":
                no_match_msg = (
                    "आपके प्रश्न के लिए पंजाब CGWB डेटासेट में कोई प्रासंगिक निगरानी स्टेशन या रिकॉर्ड नहीं मिला।\n\n"
                    "डेटासेट में **1996 से 2024** के बीच पंजाब के **22 जिलों** के त्रैमासिक अवलोकन शामिल हैं।"
                )
            elif lang == "hinglish":
                no_match_msg = (
                    "Aapki query ke liye CGWB dataset mein koi relevant monitoring stations nahi mile.\n\n"
                    "Dataset mein Punjab ke **22 districts** ke records shamil hain (1996–2024)."
                )
            else:
                no_match_msg = (
                    "No relevant groundwater monitoring stations or records were found in the Punjab CGWB dataset for your query.\n\n"
                    "The dataset contains quarterly observations for **22 canonical districts of Punjab** between **1996 and 2024**."
                )
            return {
                "answer": no_match_msg,
                "engine": "Zero-Hallucination Guardrail (No Matching Records)",
                "intent": "no_match",
                "sources": ["CGWB Monitoring Records (Punjab: 1996-2024)"],
                "records": pd.DataFrame(),
                "figure": None
            }

        supporting_df = search_matches
        context_lines = [f"- {row['search_text']}" for _, row in search_matches.iterrows()]
        context_text = "RELEVANT MONITORING STATIONS:\n" + "\n".join(context_lines)
        sources = [
            "CGWB Manual Quarterly Groundwater Dataset (Punjab: 1996-2024 observations)",
            "CGWB Station Monitoring Network (Punjab)"
        ]

    # 6. Generate Answer via Groq LLM in user's language
    llm_answer, used_model, err_msg = generate_grounded_answer_with_groq(query, context_text, lang=lang, api_key=api_key)

    if llm_answer:
        final_answer = llm_answer
        engine = f"Groq ({used_model}) - Grounded Generation"
    else:
        final_answer = generate_targeted_fallback_answer(query, context_text, lang=lang)
        if api_key:
            engine = f"Verified Local Analytics (Groq API Error: {err_msg})"
        else:
            engine = "Verified Local Analytics Engine (Offline Mode)"

    return {
        "answer": final_answer,
        "engine": engine,
        "intent": intent,
        "sources": list(dict.fromkeys(sources)),
        "records": supporting_df,
        "figure": fig
    }


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Testing Complete AI Groundwater Assistant Pipeline...")
    raw_data, corpus_data, model_enc, embeddings_vec = build_search_index()
    d_summary, y_summary = load_summary_data()

    tests = [
        # Hindi district ranking
        "पंजाब में सबसे गहरा भूजल स्तर किस जिले में है?",
        # Ludhiana data availability
        "क्या लुधियाना का भूजल डेटा उपलब्ध है?",
        # English Ludhiana query
        "Is data available for Ludhiana?",
        # General knowledge: Aquifer (Hindi & English)
        "What is an aquifer?",
        "एक्विफर क्या है?",
        # General knowledge: Conservation
        "भूजल संरक्षण कैसे करें?",
        # Village query in Hindi
        "Balluana गांव में भूजल स्तर क्या है?",
        # Trend query in Hindi
        "पंजाब का सालाना भूजल रुझान दिखाओ",
        # Unrelated query in Hindi & English
        "पिज़्ज़ा कैसे बनाते हैं?",
        "How to make pizza?"
    ]

    for t in tests:
        print(f"\n========================================================")
        print(f"QUERY: {t}")
        res = answer_groundwater_query(t, raw_data, corpus_data, d_summary, y_summary, model_enc, embeddings_vec)
        print("Engine:", res["engine"])
        print("Intent:", res["intent"])
        print("Sources:", res["sources"])
        print("Figure Generated:", res.get("figure") is not None)
        print("Records Count:", len(res["records"]))
        print("Answer Snippet:\n", res["answer"][:300] + "...")
