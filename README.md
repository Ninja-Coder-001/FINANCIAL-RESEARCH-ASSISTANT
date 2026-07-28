# Financial Research Assistant

An AI tool that reads annual reports and answers questions about them.

Upload a PDF, get an executive summary, key financial metrics, risk analysis and an assessment of management sentiment — then ask follow-up questions directly of the document.

**Live app:** https://financial-research-assistant-3ksjyzsphsgchikeymi7d9.streamlit.app/

---

## The problem

Annual reports run 300–600 pages. Analysts covering a company spend hours pulling out the same handful of things every time: how the business performed, what management is worried about, what the headline numbers look like, and whether the tone has shifted from last year.

Most of that first pass is mechanical. This tool automates it.

---

## What it does

**Four automated analysis modules**, generated on upload:

| Module | Output |
|---|---|
| Executive Summary | Business overview, headline results, management outlook |
| Financial Highlights | Key metrics extracted and grouped, sector-adaptive |
| Risk Factors | Credit, interest rate, liquidity and operational risk with mitigation approach |
| Management Sentiment | Tone assessment with supporting themes and forward guidance |

**Conversational Q&A** over the full document, with context carried across turns — so follow-ups like "what about liabilities?" resolve correctly against the previous answer.

**Auto-detected page numbers** for the KPI and MD&A sections, with manual override when the heuristic gets it wrong.

---

## Architecture

The system deliberately uses **two different retrieval strategies** depending on what is being asked.

### Hybrid retrieval — for narrative content

Semantic search alone consistently failed on this document type. Terms like "chairman" or "management" appear across governance reports, board committee listings and remuneration tables, and those keyword-dense sections outscored the actual commentary being searched for.

The fix was an ensemble of semantic and lexical search:

```
BM25 (keyword)  ─┐
                 ├─→ EnsembleRetriever (0.5 / 0.5) ─→ LLM
Chroma (semantic)─┘
```

BM25 catches exact phrase matches that embeddings miss. Embeddings catch paraphrases that BM25 misses. Merging both rankings fixed an entire class of previously failing queries.

### Direct page injection — for structured data

For KPI tables and MD&A sentiment, retrieval was the wrong tool entirely.

Requests for AUM or ROE consistently retrieved Basel III liquidity tables from deep in the financial statements rather than the one-page KPI summary near the front. Ninety pages of dense financial vocabulary will always outscore a single page on similarity alone.

Since KPI and MD&A sections sit in predictable locations, the app locates the page and passes its contents straight to the model — no retrieval step.

**The underlying principle:** retrieval is for finding things you cannot locate. It is not for fetching things whose address you already know.

### History-aware query rewriting

Follow-up questions containing "this", "that" or "it" have no meaningful embedding on their own. Before retrieval, an LLM call rewrites the question into a standalone form using conversation history.

---

## Tech stack

| Component | Choice |
|---|---|
| Orchestration | LangChain |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| Lexical search | BM25 (`rank_bm25`) |
| PDF parsing | PyPDF |
| Interface | Streamlit |

---

## Tested on

Validated across sectors to check that page detection and sector-agnostic extraction generalise:

- Housing finance (NBFC)
- Steel and mining
- Chemicals
- Stock exchange
- Industrial chemicals (international filing)

---

## Running locally

```bash
git clone https://github.com/Ninja-Coder-001/FINANCIAL-RESEARCH-ASSISTANT.git
cd FINANCIAL-RESEARCH-ASSISTANT

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

Then:

```bash
streamlit run app.py
```

---

## Known limitations

These are documented rather than hidden — each one is a real constraint of the approach.

**Tables split across chunks.** Character-based chunking cuts financial statements mid-structure. Balance sheets often return one section completely and flag the rest as missing. Solving this properly requires table-aware parsing rather than character splitting.

**Image-based pages are invisible.** Designed spreads — a chairman's photograph with the title and pull quote rendered as graphics — extract as nothing. In one report the MD interview cover page yielded a single stray line; the actual content was on the following three pages. OCR would be required.

**Questions about people underperform questions about topics.** Asking what a named executive said tends to retrieve governance text describing their role. Asking about the subject they discussed works reliably.

**Compound questions degrade.** Three questions in one message produce a blended embedding that matches no single chunk well. Query decomposition would address this.

**Page detection is heuristic.** Keyword-density scoring with a position weighting toward the front of the document. It is right most of the time and wrong occasionally, which is why the manual override exists.

---

## What would come next

- **Reranking** — retrieve broadly, then score with a cross-encoder. Highest expected accuracy gain.
- **Table-aware extraction** — `unstructured.io` or Camelot for financial statements.
- **Query decomposition** — split compound questions, retrieve independently, synthesise.
- **Evaluation harness** — RAGAS or LangSmith, to measure retrieval quality rather than assess it by eye.

Add README
