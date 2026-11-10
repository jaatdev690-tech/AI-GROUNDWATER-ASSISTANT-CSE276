# 💧 AI Groundwater Information Assistant

**CSE276 – AI Foundations | Project P16 | Lovely Professional University**

The AI Groundwater Information Assistant is a source-grounded Retrieval-Augmented Generation (RAG) chatbot that helps users access groundwater-related information about Punjab, India.

The project uses groundwater monitoring datasets provided by the Central Ground Water Board (CGWB), along with a groundwater-related PDF document. It combines semantic search, statistical analysis, an AI-powered chatbot, and an interactive dashboard to make groundwater information easier to access and understand.

---

## 1. Features

| Feature | Description |
|---|---|
| **Interactive Dashboard** | Provides year and district filters, summary metrics, district rankings, and downloadable data. |
| **Year-wise Trend Chart** | Displays statewide groundwater trends over time. |
| **District Comparison** | Compares average groundwater measurements across 22 Punjab districts. |
| **RAG Chatbot** | Retrieves relevant groundwater information and generates natural-language answers. |
| **Village-Level Queries** | Supports village-level groundwater queries where matching monitoring records are available. |
| **District Trend Graphs** | Provides groundwater trend visualisations for individual and multiple districts. |
| **General Groundwater Knowledge** | Answers questions about groundwater, aquifers, water tables, and conservation. |
| **Query Guardrails** | Handles unrelated questions, unsupported locations, and unavailable records. |
| **Groq LLM Integration** | Uses the Groq API for AI-based answer generation. |
| **Local Fallback** | Provides supported statistical and rule-based responses when the Groq API is unavailable. |

---

## 2. Project Structure

```text
AI GROUNDWATER ASSISTANT/
│
├── app.py
├── groundwater_rag.py
├── read_data.py
├── visualize_data.py
├── verify_groundwater.py
│
├── test_suite.py
├── verify_12_queries.py
├── verify_12_results.json
├── verify_abulkharana_fix.py
│
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── district_groundwater_summary.csv
├── yearly_groundwater_summary.csv
│
├── district_groundwater_graph.png
├── groundwater_yearly_trend.png
│
└── Data/
    ├── gwl_manual_quarterly_cgwb_pb_1991_2020 (1).csv
    ├── gwl_manual_quarterly_cgwb_pb_2021_2025.csv
    ├── searchable_corpus.csv
    ├── groundwater_embeddings.pt
    │
    └── reports/
        └── Punjab_Groundwater_Yearbook_2021_2022.pdf.pdf
```

**Note:** The project includes original datasets and preprocessed files. Some processed files can be regenerated using the available Python scripts.

---

## 3. Requirements

The following software and tools are required to run the project:

- Python 3.10 or a compatible version.
- Visual Studio Code or another Python editor.
- Required Python packages listed in `requirements.txt`.
- Internet connection for downloading model files and using the Groq API.

---

## 4. Quick Start

### Step 1: Extract and Open the Project

1. Extract the downloaded project ZIP file.
2. Open the `AI GROUNDWATER ASSISTANT` folder.
3. Open this folder in Visual Studio Code.
4. Open the terminal inside VS Code.

### Step 2: Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

**Note:** The project uses PyTorch for embedding operations. CPU execution is sufficient for this application. For GPU-specific installation, refer to the official [PyTorch website](https://pytorch.org/get-started/locally/).

### Step 4: Configure the Groq API Key

The project uses the Groq API for LLM-based answer generation.

An `.env.example` file is provided as a template.

1. Create a copy of `.env.example`.
2. Rename the copied file to `.env`.
3. Open the `.env` file.
4. Add your own Groq API key:

```text
GROQ_API_KEY=your_groq_api_key_here
```

You can obtain an API key from the [Groq Console](https://console.groq.com/keys).

**Important:** Never share your API key or include your private `.env` file in the project submission ZIP. Only `.env.example` should be included.

The application supports local statistical and rule-based functionality without a Groq API key. However, AI-generated responses require a valid API key.

### Step 5: Generate District Summary (If Required)

The project includes a preprocessed district summary file.

If you need to regenerate it from the original groundwater CSV files, run:

```bash
python read_data.py
```

The script processes the available CSV data and generates district-wise groundwater statistics.

The script generates `district_groundwater_summary.csv`. The yearly summary file is separately included in the project.

### Step 6: Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application normally opens in the browser at:

```text
http://localhost:8501
```

**First Run:** The Sentence Transformer model may need to download its pretrained model files if they are not already available locally. The time required depends on the internet connection and system performance.

---

## 5. Usage Examples

Users can enter groundwater-related questions in natural language through the chatbot.

### District-Level Queries

Examples:

- What is the average groundwater level in Ludhiana?
- Which district has the highest groundwater measurement?
- Compare groundwater trends in Ludhiana and Patiala.
- Show the groundwater graph for Punjab districts.

### Village-Level Queries

Examples:

- What is the groundwater depth in village Hussainpura?
- Show the groundwater trend for Balluana in Fazilka.
- What groundwater information is available for Devigarh?

The system provides village-level information when matching records are available in the dataset.

### General Groundwater Questions

Examples:

- What is groundwater?
- What is an aquifer?
- What is the water table?
- How can groundwater be conserved?

### Historical Trend Queries

Examples:

- Show the yearly groundwater trend for Punjab.
- Compare groundwater trends between different districts.

The responses depend on the information available in the datasets and the configured chatbot functionality.

---

## 6. Dataset Details

The project uses groundwater monitoring data provided by the Central Ground Water Board (CGWB), Government of India.

### Original CSV Datasets

| File Name | Source | Original Records |
|---|---|---:|
| `gwl_manual_quarterly_cgwb_pb_1991_2020 (1).csv` | CGWB | 32,478 |
| `gwl_manual_quarterly_cgwb_pb_2021_2025.csv` | CGWB | 4,463 |

### Dataset Statistics

| Parameter | Value |
|---|---:|
| Number of original CSV files | 2 |
| Total raw records | 36,941 |
| Valid records after cleaning | 36,934 |
| Records removed during cleaning | 7 |
| Number of districts | 22 |
| Original columns | 21 |
| Observation period | 1996–2024 |
| Searchable corpus records | 1,613 |
| Searchable corpus columns | 11 |
| Groundwater PDF documents | 1 |

### Dataset Coverage

- Geographical coverage: 22 canonical districts of Punjab.
- Observation period: 1996–2024.
- The dataset includes quarterly manual groundwater-level observations.
- District names referring to Mohali and SAS Nagar are standardised during preprocessing.

### Important Dataset Feature

`Groundwater Level Quarterly Manual (meter)`

This column contains groundwater-level measurements in metres below ground level (mbgl).

The original monitoring records should be consulted when interpreting unusual or extreme measurements.

---

## 7. Processed Dataset Files

The project includes additional files used for statistical analysis and semantic retrieval.

| File Name | Purpose |
|---|---|
| `district_groundwater_summary.csv` | Contains district-wise observation counts and groundwater statistics. |
| `yearly_groundwater_summary.csv` | Contains yearly groundwater summary information. |
| `searchable_corpus.csv` | Contains processed records prepared for semantic retrieval. |
| `groundwater_embeddings.pt` | Stores numerical vector representations used for semantic search. |

The searchable corpus contains 1,613 processed records with 11 columns.

The embedding model used in the project is `all-MiniLM-L6-v2` from the Sentence Transformers library.

---

## 8. How the RAG Pipeline Works

The project follows a Hybrid Retrieval-Augmented Generation approach.

The basic workflow is:

```text
                   User Query
                       |
                       v
                Query Processing
                       |
                       v
                Scope Guardrails
                       |
                       v
                Query Identification
                       |
          +------------+-------------+
          |            |             |
          v            v             v
      Village       General       Dataset
      Queries       Knowledge     Queries
          |            |             |
          v            v             v
      Village        Groq LLM     Statistical
      Analytics      Processing   Processing
          |            |             |
          +------------+-------------+
                       |
                       v
                Relevant Context
                       |
                       v
                    Groq LLM
                       |
                       v
                  Final Answer
```

### Main Steps

1. **User Query:** The user enters a groundwater-related question.
2. **Query Processing:** The system identifies the type of question.
3. **Guardrails:** The system checks unsupported locations, unavailable records, and unrelated questions.
4. **Information Retrieval:** Relevant groundwater information is retrieved using semantic search and statistical processing.
5. **Context Preparation:** Relevant information is prepared for answer generation.
6. **Answer Generation:** The Groq-based LLM generates a natural-language response using the available context.
7. **Response Display:** The final answer is displayed through the chatbot interface.

---

## 9. Technical Stack

| Component | Technology |
|---|---|
| Programming Language | [Python](https://docs.python.org/3/) |
| Web Application | [Streamlit](https://docs.streamlit.io/) |
| Data Processing | [Pandas](https://pandas.pydata.org/docs/) |
| Interactive Visualisation | [Plotly](https://plotly.com/python/) |
| Semantic Search | [Sentence Transformers](https://www.sbert.net/) |
| Embedding Model | `all-MiniLM-L6-v2` |
| Tensor Operations | [PyTorch](https://pytorch.org/) |
| Large Language Model API | [Groq](https://console.groq.com/docs) |
| Environment Configuration | [python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## 10. Utility Scripts

The project contains additional Python scripts for data processing, visualisation, and testing.

| Script | Purpose |
|---|---|
| `app.py` | Runs the Streamlit dashboard and chatbot interface. |
| `groundwater_rag.py` | Contains the RAG pipeline, retrieval logic, analytics, and chatbot processing. |
| `read_data.py` | Processes original CSV files and generates district-wise summary statistics. |
| `visualize_data.py` | Generates groundwater data visualisations. |
| `verify_groundwater.py` | Performs groundwater dataset verification. |
| `test_suite.py` | Contains project testing functionality. |
| `verify_12_queries.py` | Runs a set of groundwater-related query checks. |
| `verify_12_results.json` | Stores the corresponding query verification results. |
| `verify_abulkharana_fix.py` | Contains a targeted verification script. |

---

## 11. Limitations

The current project has certain limitations:

1. The structured groundwater dataset mainly covers 22 districts of Punjab.
2. The chatbot depends on the available CSV datasets and groundwater PDF document.
3. The system does not directly connect to live groundwater monitoring stations or sensors.
4. AI-generated answers depend on the availability of the Groq API.
5. The accuracy of statistical results depends on the quality of the original groundwater records.
6. More extensive testing and manually verified answers are required for quantitative evaluation.

---

## 12. Future Scope

The project can be improved in the future through the following developments:

1. Expansion of groundwater datasets to more Indian states and districts.
2. Integration of additional groundwater reports and research documents.
3. Integration of real-time groundwater monitoring data.
4. Improvement of the RAG system and document retrieval process.
5. Addition of machine learning models for groundwater-level prediction.
6. Support for additional Indian languages such as Hindi and Punjabi.
7. Development of a mobile application.
8. Addition of more advanced dashboard visualisations and geographical maps.

---

## 13. Team Members

**Project:** P16 – AI Groundwater Information Assistant  
**Course:** CSE276 – AI Foundations  
**University:** Lovely Professional University  
**Faculty:** Mr. Ravneet Singh

| Team Member | Registration Number |
|---|---|
| Devsanwariya Jat | 12513644 |
| Harshdeep Singh | 12505927 |

---

## 14. References

1. [Central Ground Water Board (CGWB)](https://cgwb.gov.in/)
2. [Sentence Transformers Documentation](https://www.sbert.net/)
3. [Groq API Documentation](https://console.groq.com/docs)
4. [Streamlit Documentation](https://docs.streamlit.io/)
5. [Pandas Documentation](https://pandas.pydata.org/docs/)
6. [Plotly Documentation](https://plotly.com/python/)
7. [Python Documentation](https://docs.python.org/3/)
8. Lewis et al. (2020), [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).

---

## 15. Academic Use

This project has been developed for academic purposes as part of the CSE276 – AI Foundations course at Lovely Professional University.

The groundwater datasets and related reports are attributed to their respective original sources, including the Central Ground Water Board (CGWB), Government of India.