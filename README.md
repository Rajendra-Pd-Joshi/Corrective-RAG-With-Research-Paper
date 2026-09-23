# Corrective RAG with Research Paper

A notebook-based implementation of a **Corrective Retrieval-Augmented Generation (C-RAG)** workflow, developed incrementally from a basic RAG pipeline into a retrieval-aware system that evaluates retrieved evidence, refines context, falls back to web search when local knowledge is insufficient, rewrites queries for web search, and explicitly handles ambiguous retrieval.

The repository also includes the associated **research paper** and the documents used by the notebooks.

## Overview

Traditional RAG pipelines usually follow a simple pattern:

```text
Question → Retrieve → Generate
```

This project explores a corrective workflow in which the retrieved evidence is first evaluated before generation:

```text
Question
   ↓
Retrieve relevant chunks
   ↓
Evaluate retrieved evidence
   ↓
 ┌───────────────┬────────────────┬────────────────┐
 │ CORRECT       │ INCORRECT      │ AMBIGUOUS      │
 │               │                │                │
 │ refine local  │ rewrite query  │ combine /      │
 │ evidence      │ + web search   │ reconcile       │
 └───────────────┴────────────────┴────────────────┘
   ↓
Refined context
   ↓
Generate answer
```

The implementation is built with **LangGraph** so the retrieval, evaluation, correction, search, refinement, and generation steps are represented as explicit graph nodes and conditional routes.

## What the notebooks cover

| Notebook | Focus | Main contribution |
|---|---|---|
| [1_basic_rag.ipynb](./1_basic_rag.ipynb) | Basic RAG | Loads PDF documents, splits them into chunks, creates embeddings, stores them in FAISS, retrieves the top documents, and generates an answer with an LLM. |
| [2_retrieval_refinement.ipynb](./2_retrieval_refinement.ipynb) | Retrieval refinement | Adds sentence-level filtering so only evidence that directly helps answer the question is retained before generation. |
| [3_retrieval_evaluator.ipynb](./3_retrieval_evaluator.ipynb) | Retrieval evaluation | Scores retrieved chunks and classifies retrieval as **CORRECT**, **INCORRECT**, or **AMBIGUOUS** using configurable thresholds. |
| [4_web_search_refinement.ipynb](./4_web_search_refinement.ipynb) | Web fallback | Adds Tavily web search when internal retrieval is insufficient, then sends the web results through the refinement and generation stages. |
| [5_query_rewrite.ipynb](./5_query_rewrite.ipynb) | Query rewriting | Introduces an LLM-based query rewriting step that converts the original question into a keyword-oriented web query, including recency constraints when appropriate. |
| [6_ambiguous.ipynb](./6_ambiguous.ipynb) | Ambiguous retrieval | Extends correction so ambiguous cases can use both retrieved internal knowledge and web evidence instead of treating uncertainty as a simple retrieval failure. |

## Notebook progression

### 1. Basic RAG

The first notebook establishes the baseline pipeline.

It:

- Loads PDF files with `PyPDFLoader`
- Splits documents with `RecursiveCharacterTextSplitter`
- Uses OpenAI embeddings
- Builds a **FAISS** vector store
- Creates a similarity retriever with `k=4`
- Uses `ChatOpenAI` for answer generation
- Connects retrieval and generation with a simple **LangGraph** workflow

The notebook uses `text-embedding-3-small` for embeddings and `gpt-4o-mini` for generation in the baseline example.

### 2. Retrieval Refinement

The second notebook improves the baseline by filtering retrieved content before generation.

Instead of passing complete retrieved chunks directly to the answer model, the workflow examines individual sentences and keeps a sentence only when it directly contributes to answering the question.

Conceptually:

```text
Retrieve → Sentence filtering → Refined context → Generate
```

This reduces irrelevant retrieved text and gives the generator a more focused context.

### 3. Retrieval Evaluator

The third notebook introduces explicit retrieval quality assessment.

Each retrieved chunk is evaluated with an LLM-based structured evaluator. The implementation uses two thresholds:

- `UPPER_TH = 0.7`
- `LOWER_TH = 0.3`

The retrieval state is classified as:

- **CORRECT** — at least one retrieved chunk scores above the upper threshold.
- **INCORRECT** — all retrieved chunks score below the lower threshold.
- **AMBIGUOUS** — the scores fall between those two conditions.

This classification becomes the control signal for the corrective graph.

### 4. Web Search Refinement

The fourth notebook introduces an external knowledge fallback using **Tavily**.

When internal retrieval is classified as insufficient, the graph can branch to web search, convert search results into LangChain `Document` objects, refine those results, and then generate an answer from the refined context.

The workflow therefore moves beyond a closed document collection:

```text
Internal retrieval
      ↓
Evidence evaluation
      ↓
Insufficient evidence
      ↓
Web search
      ↓
Refinement
      ↓
Generation
```

The notebook uses `gpt-4o-mini` with temperature 0 and `text-embedding-3-large` for the retrieval stage.

### 5. Query Rewrite

The fifth notebook improves the web-search fallback by rewriting the user's question before searching.

A structured `WebQuery` output is generated by the LLM. The rewrite prompt is designed to:

- convert the question into search-oriented keywords,
- preserve the user's information need,
- add a recency constraint when the question implies freshness,
- avoid answering the question during the rewrite step.

The resulting query is then passed to Tavily.

This creates a more deliberate corrective path:

```text
Poor internal retrieval
        ↓
Query rewrite
        ↓
Web search
        ↓
Refine
        ↓
Generate
```

### 6. Ambiguous Retrieval

The final notebook treats ambiguity as a first-class case.

Rather than viewing retrieval as only correct or incorrect, the workflow distinguishes uncertainty and allows ambiguous cases to use a broader evidence set.

The refinement logic is organized as:

```text
CORRECT
  → refine internal evidence

INCORRECT
  → rewrite query
  → web search
  → refine web evidence

AMBIGUOUS
  → rewrite query
  → web search
  → refine with internal + web evidence
```

The final graph therefore makes the corrective behavior explicit instead of blindly generating from the first retrieved results.

## Core technologies

- **Python 3.11+**
- **LangChain**
- **LangGraph**
- **OpenAI Chat Models**
- **OpenAI Embeddings**
- **FAISS**
- **Tavily Search**
- **Pydantic structured outputs**
- **PyPDFLoader**
- **RecursiveCharacterTextSplitter**
- **Jupyter Notebook**
- **python-dotenv**

## Repository structure

```text
Corrective-RAG-With-Research-Paper/
│
├── 1_basic_rag.ipynb
├── 2_retrieval_refinement.ipynb
├── 3_retrieval_evaluator.ipynb
├── 4_web_search_refinement.ipynb
├── 5_query_rewrite.ipynb
├── 6_ambiguous.ipynb
│
├── documents/
│   ├── book1.pdf
│   ├── book2.pdf
│   └── book3.pdf
│
├── Research Paper.pdf
├── LICENSE
├── .gitignore
└── README.md
```

## End-to-end architecture

The final implementation can be understood as the following LangGraph:

```text
                         ┌──────────────┐
                         │   Question   │
                         └──────┬───────┘
                                ↓
                         ┌──────────────┐
                         │   Retrieve   │
                         └──────┬───────┘
                                ↓
                      ┌────────────────────┐
                      │ Evaluate each doc  │
                      └─────────┬──────────┘
                                ↓
                   ┌────────────┼────────────┐
                   ↓            ↓            ↓
                CORRECT      INCORRECT    AMBIGUOUS
                   ↓            ↓            ↓
                Refine     Rewrite query  Rewrite query
                   ↓            ↓            ↓
                   │        Web search   Web search
                   │            ↓            ↓
                   │          Refine       Refine
                   │            │        internal + web
                   └────────────┴────────────┘
                                ↓
                         ┌──────────────┐
                         │   Generate   │
                         └──────┬───────┘
                                ↓
                             Answer
```

## Setup

Clone the repository:

```bash
git clone https://github.com/Rajendra-Pd-Joshi/Corrective-RAG-With-Research-Paper.git
cd Corrective-RAG-With-Research-Paper
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages used throughout the notebooks:

```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters langgraph faiss-cpu pypdf python-dotenv pydantic tavily-python
```

Some notebook code uses the older `langchain_community.tools.tavily_search.TavilySearchResults` interface. Depending on your installed LangChain version, you may need the newer `langchain-tavily` package/API.

## Environment variables

Create a `.env` file in the project root and provide the credentials required by the notebooks:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

The notebooks load environment variables with `python-dotenv`.

## Running the notebooks

Open the project in Jupyter Notebook or VS Code and run the notebooks in numerical order.

The recommended learning path is:

```text
1_basic_rag.ipynb
        ↓
2_retrieval_refinement.ipynb
        ↓
3_retrieval_evaluator.ipynb
        ↓
4_web_search_refinement.ipynb
        ↓
5_query_rewrite.ipynb
        ↓
6_ambiguous.ipynb
```

Running them in order makes the evolution of the corrective RAG architecture easier to follow.

## Key ideas demonstrated

### Retrieval is evaluated before generation

The project does not assume that the top-`k` retrieved chunks are automatically useful. Retrieved evidence is explicitly judged against the question.

### Corrective routing

The retrieval verdict controls the next graph node. This makes the workflow adaptive rather than linear.

### Context refinement

Retrieved chunks are decomposed and filtered so the generation model receives evidence that is more directly related to the user's question.

### External knowledge fallback

When the local document collection cannot adequately answer the question, the graph can obtain additional evidence through web search.

### Query-aware web search

The web-search branch can rewrite the user's question into a search-oriented query and preserve recency requirements when the question calls for recent information.

### Ambiguity handling

The final notebook distinguishes between confidently supported retrieval, clearly insufficient retrieval, and uncertain/mixed retrieval.

## Research Paper

The repository contains the associated research paper:

**[Research Paper.pdf](./Research%20Paper.pdf)**

The paper should be read together with the notebooks to connect the implementation with the underlying Corrective RAG research.

## Notes

This repository is organized as an educational, incremental notebook implementation. The notebooks intentionally expose the intermediate stages of the architecture so that each corrective mechanism can be studied independently.

API/model names and LangChain integrations can change over time. When reproducing the notebooks with newer package versions, check the corresponding package documentation if an integration has been deprecated or renamed.

## License

This project is released under the license included in [LICENSE](./LICENSE).
