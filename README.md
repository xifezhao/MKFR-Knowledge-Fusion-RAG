# MKFR: A Framework for Multi-faceted Knowledge Fusion in Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and experimental setup for the paper "MKFR: A Framework for Multi-faceted Knowledge Fusion in Retrieval-Augmented Generation." MKFR is a novel architecture designed to enhance Large Language Models (LLMs) by systematically integrating and fusing knowledge from diverse sources: vector databases, relational databases, and graph databases.

## 📜 Abstract (from the paper)

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding them in external knowledge, mitigating issues like hallucinations and outdated information. However, most RAG systems rely primarily on semantic retrieval from unstructured text, underutilizing structured relational data and interconnected knowledge graphs. This paper presents the Multi-faceted Knowledge Fusion Retrieval (MKFR) framework, a novel architecture that systematically integrates knowledge from three distinct sources: vector databases for semantic understanding, relational databases for structured facts and metadata, and graph databases for explicit relationships and multi-hop reasoning. MKFR combines query intent analysis, parallel multi-source retrieval, and a weighted Reciprocal Rank Fusion (RRF) strategy to assemble comprehensive contexts for LLMs. Experiments on a simulated MS MARCO-style dataset compare MKFR against single-source baselines. Results show that while vector-based semantic retrieval remains strong, MKFR—especially under vector-dominant RRF weighting—achieves similar or identical retrieval quality with significantly reduced latency.

## ✨ Key Features & Contributions

*   **Multi-Source Integration:** Systematically combines vector, relational, and graph databases for comprehensive knowledge retrieval.
*   **Heterogeneous Querying:** Strategies for querying diverse database types within a unified RAG pipeline.
*   **Intelligent Fusion:** Employs Reciprocal Rank Fusion (RRF) with dynamic weighting and source selection based on query intent.
*   **Experimental Evaluation:** Demonstrates MKFR's effectiveness on a simulated MS MARCO-style dataset, analyzing retrieval quality (NDCG, MRR) and latency.
*   **Flexible Architecture:** Provides a modular foundation for building more robust and informed RAG systems.

## 🏗️ System Architecture

The MKFR framework processes an input query through several stages:

1.  **Query Input and Preprocessing & Intent Analysis:**
    *   The raw query is cleaned, normalized.
    *   Named Entity Recognition (NER) and Linking (NEL) identify key entities.
    *   Query Intent Classification determines the likely information need and suggests appropriate sources/strategies (e.g., factual, relational, semantic).
    *   The query is embedded for semantic search.
2.  **Parallel (or Adaptive) Multi-Source Retrieval:**
    *   Based on the intent and configuration, retrieval requests are dispatched to:
        *   **Vector Database:** For semantic matching using query embeddings.
        *   **Relational Database:** For structured data lookups, metadata filtering, or keyword-based searches (e.g., BM25).
        *   **Graph Database:** For retrieving information based on entity relationships and path traversals.
    *   Each source returns a ranked list of candidate documents/snippets with scores.
3.  **Candidate Set Fusion:**
    *   The candidate lists from different sources are fused using **Reciprocal Rank Fusion (RRF)**.
    *   Source weights in RRF can be dynamically adjusted based on query intent.
4.  **Multi-Dimensional Re-ranking (Optional):**
    *   An optional stage to re-rank the fused list using richer features (e.g., document freshness, source authority, textual coherence). *This is primarily a future enhancement in the current codebase.*
5.  **Context Construction and Output:**
    *   The top-N ranked knowledge snippets are selected and formatted to create a comprehensive context for the LLM.

## 📂 Directory Structure

```
.
├── documents.jsonl         # Simulated document corpus
├── queries.train.tsv       # Simulated queries
├── qrels.train.tsv         # Simulated query-relevance judgments
├── relations.jsonl         # Simulated explicit graph relations
├── dataset.py              # Handles data loading and preprocessing
├── sim_databases.py        # Simulates Vector, Relational, and Graph DBs
├── mkfr_pipeline.py        # Core MKFR retrieval pipeline logic (simplified in current Run.py)
├── evaluation.py           # Evaluation metrics (P@K, R@K, NDCG@K, MRR)
├── Run.py                  # Main script to run benchmark experiments
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## ⚙️ Requirements

*   Python 3.8+
*   Libraries listed in `requirements.txt`:
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `sentence-transformers`
    *   `rank_bm25`
    *   `networkx`

## 🚀 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/MKFR.git # Replace with your actual repo URL
    cd MKFR
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Dataset (Simulated `Dsim`)

The experiments use a small, programmatically generated dataset (`Dsim`) designed to simulate aspects of the MS MARCO document ranking task and to test different retrieval modalities.

*   **`documents.jsonl`**: Contains 5 unique documents, each with `doc_id`, `text`, and `metadata` (including `category`, `year`, and `entities`).
*   **`queries.train.tsv`**: Contains 6 unique queries, including 2 "graph-friendly" queries designed for relational path traversal.
*   **`qrels.train.tsv`**: Provides 9 positive query-document relevance judgments.
*   **`relations.jsonl`**: Defines 12 explicit graph relations (entity-entity and document-entity).

**Note:** If these data files are not present when `dataset.py` is first run, dummy versions will be automatically created to allow the code to execute.

## ධ Running the Experiments

The main script to run the benchmark experiments is `Run.py`.

```bash
python Run.py
```

This script will:
1.  Initialize the `Dataset` object (loading or creating dummy data).
2.  Set up simulated Vector, Relational, and Graph databases.
3.  Define several pipeline configurations (e.g., VectorOnly, RelationalOnly, various MKFR fusion strategies).
4.  Run each configuration against all queries in the dataset.
5.  For each query and configuration:
    *   Retrieve relevant documents from active sources.
    *   Fuse results using RRF (if multiple sources are active).
    *   Calculate evaluation metrics (P@K, R@K, F1@K, NDCG@K for K={1,3,5,10}, and MRR).
    *   Measure average query processing latency.
6.  Output a summary of average metrics for all configurations to the console and save it to `benchmark_summary_msmarco.csv`.
7.  Perform a basic failure case analysis between a baseline (e.g., `VectorOnly_TopN`) and a hybrid configuration (e.g., `VecRel_RRF_TopN`), saving results to a CSV file.

## 🧩 Core Code Components

*   **`dataset.py`**:
    *   Loads documents, queries, qrels, and explicit relations.
    *   Handles document text embedding using `SentenceTransformer`.
    *   Creates dummy data files if they don't exist on first run.
*   **`sim_databases.py`**:
    *   `SimulatedVectorDB`: Performs k-NN search on document embeddings using cosine similarity.
    *   `SimulatedRelationalDB`: Simulates keyword search using `BM25Okapi` and metadata filtering on a Pandas DataFrame.
    *   `SimulatedGraphDB`: Builds an in-memory graph using `NetworkX`, populating it from document metadata and explicit relations. Supports basic entity mention and path-based queries.
*   **`mkfr_pipeline.py`**:
    *   Defines the `MKFRPipeline` class, originally intended to orchestrate the full multi-source retrieval and fusion.
    *   In the current `Run.py` setup, this class is used more as a wrapper for single-source retrieval when `Run.py` orchestrates the multi-source calls and RRF itself.
*   **`evaluation.py`**:
    *   Contains functions to calculate standard IR metrics: Precision@K, Recall@K, F1-Score@K, NDCG@K, and Mean Reciprocal Rank (MRR).
*   **`Run.py`**:
    *   The main experiment runner.
    *   Initializes the dataset and simulated databases.
    *   Defines various experimental configurations.
    *   Iterates through configurations and queries, performing retrieval.
    *   Manages fetching results from individual sources.
    *   Implements Reciprocal Rank Fusion (RRF) to combine results.
    *   Calculates and aggregates performance metrics.
    *   Outputs results and performs failure analysis.

## 📈 Results Summary (from the paper)

*   **Vector-based semantic retrieval (`VectorOnly_TopN`) is a strong baseline** on the simulated dataset, achieving high retrieval quality (e.g., NDCG@5 of 0.9599).
*   **MKFR with vector-dominant RRF weighting (`MKFR_RRF_TopN_VecDom_Intent`) can match the retrieval quality of the best single source (`VectorOnly_TopN`) while significantly reducing average query latency** (e.g., from 0.056s to 0.027s on the simulated setup).
*   The effectiveness of RRF depends critically on **appropriate source weighting**. Balanced weights can dilute strong signals if weaker sources are included.
*   The **simulated graph database shows potential for targeted queries** (e.g., multi-hop relational questions) when guided by query intent, but its general contribution is limited by the KG's sparsity and simple query mapping in the current setup.
*   Individual non-vector baselines (RelationalOnly, GraphOnly with general strategy) performed significantly worse than vector-based retrieval for general queries.

## ⚠️ Limitations (from the paper & current codebase)

*   **Simulated Environment:** Uses in-memory simulated databases and a very small dummy dataset, limiting direct extrapolation to production systems.
*   **Dataset Scale & Complexity:** The `Dsim` dataset is small, which affects statistical significance and may not fully showcase the benefits of complex relational/graph queries.
*   **Query Intent Classification:** The current implementation in `Run.py` uses a placeholder (`classify_query_intent`) which is heuristic and primarily for "graph-friendly" queries. A robust, scalable solution is needed.
*   **Evaluation Metrics:** Focuses on IR metrics (P@K, NDCG@K, MRR). End-to-end evaluation of LLM-generated responses based on MKFR's context is not included.
*   **Fusion Strategy:** Primarily explores RRF with static or heuristically determined weights. More advanced fusion or learning-to-rank (LTR) models are not implemented.

## 🚀 Future Work (from the paper)

*   **Enhance Individual DB Components:** Transition to production-grade databases, improve KG construction, and implement advanced graph querying.
*   **Advanced Fusion & Re-ranking:** Explore LTR models and sophisticated score normalization.
*   **Sophisticated Query Intent & Dynamic Adaptation:** Leverage LLMs for few-shot intent classification and train ML models for dynamic policy learning.
*   **End-to-End LLM Evaluation:** Assess the impact of MKFR's context on the quality of LLM-generated responses.
*   **Scaling & Benchmarking:** Evaluate on larger, standard benchmark datasets (MS MARCO, KILT) and real-world database systems.
*   **Diverse Knowledge Types:** Integrate temporal KBs, commonsense KGs, multi-modal sources, and fact-level triple fusion.

## 📄 Citation

If you use this work, please cite the original paper (details to be added once published):

```bibtex
@article{Zhao20XXMKFR,
  title={MKFR: A Framework for Multi-faceted Knowledge Fusion in Retrieval-Augmented Generation},
  author={Xiaofei Zhao and Amin Wang and Li Ga and Chi Yan and Ha Wang and Fanglin Guo},
  year={20XX}, % Replace with actual year
  journal={Manuscript submitted to ACM}, % Replace with actual venue
  % Add volume, pages, doi etc. once available
}
```

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (you'll need to create a LICENSE file with the MIT license text).
