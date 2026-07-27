import os
import uuid
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Research Assistant",
    page_icon="📊",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {font-weight: 700;}
    .stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("analysis_done", False),
    ("summary_output", ""),
    ("kpi_output", ""),
    ("risk_output", ""),
    ("sentiment_output", ""),
    ("chain", None),
    ("chunks", None),
    ("llm", None),
    ("num_pages", 0),
    ("processed_file", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("How to use")
    st.markdown("""
    1. Upload an annual report PDF
    2. Confirm the auto-detected page numbers
    3. Click **Generate Analysis**
    4. Review the four analysis tabs
    5. Ask follow-up questions in the chat
    """)
    st.divider()
    st.caption("Tip: ask one focused question at a time for best results.")
    st.divider()
    st.caption("Built with LangChain, OpenAI GPT-4o-mini, ChromaDB and Streamlit.")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Financial Research Assistant")
st.markdown(
    "<p style='color:#64748B; font-size:1.05rem; margin-top:-0.5rem;'>"
    "AI-powered annual report analysis — summary, KPIs, risks and management "
    "sentiment in under a minute.</p>",
    unsafe_allow_html=True
)
st.divider()

# ─── File uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Annual Report (PDF)",
    type="pdf",
    help="Upload any company's annual report in PDF format"
)

# ─── Page detection helpers ───────────────────────────────────────────────────
def detect_kpi_page(chunks):
    kpi_kwrds = [
          # cross-sector
        "revenue", "ebitda", "profit after tax", "pat", "net profit",
        "return on equity", "return on assets", "roce", "net worth",
        "earnings per share", "eps", "net debt", "free cash flow",
        "operating margin", "total income", "financial highlights",
        "key performance", "performance highlights",
        # lending-specific
        "gross aum", "net stage-3", "capital adequacy ratio", "disbursement",
    ]
    page_scores = {}
    for chunk in chunks:
        count = 0
        page_no = chunk.metadata.get("page", 0)
        content = chunk.page_content.lower()
        for kwrd in kpi_kwrds:
            if kwrd in content:
                count += 1
        page_scores[page_no] = page_scores.get(page_no, 0) + count
    return max(page_scores, key=page_scores.get) if page_scores else 0


def detect_mda_page(chunks):
    mda_keywords = [
        "management discussion", "operating performance",
        "financial performance", "outlook", "revenue",
        "profitability", "results of operations", "business overview"
    ]
    page_scores = {}
    for chunk in chunks:
        count = 0
        page_no = chunk.metadata.get("page", 0)
        content = chunk.page_content.lower()
        for kwrd in mda_keywords:
            if kwrd in content:
                count += 1
        page_scores[page_no] = page_scores.get(page_no, 0) + count
    return max(page_scores, key=page_scores.get) if page_scores else 0

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a Senior Equity Research Analyst specialising in financial document analysis. "
    "Answer the question based strictly on the following context from the annual report. "
    "If the user asks multiple questions in one message, answer each one separately under "
    "its own heading, using whatever relevant information you can find for each. "
    "If the information is not in the context, say: This information was not found in the "
    "retrieved sections. "
    "Be specific, detailed, and use actual numbers where available. "
    "{context}"
)

# ─── Main logic ───────────────────────────────────────────────────────────────
if uploaded_file is not None:

    # Heavy processing runs ONLY when a new file is uploaded
    if st.session_state.processed_file != uploaded_file.name:

        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("Loading and processing your document..."):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)


        with st.spinner("Building knowledge base — this runs once per document..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        collection_name=f"doc_{uuid.uuid4().hex[:12]}"
         )
        retriever = vectordb.as_retriever(search_kwargs={"k": 10})

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}")
        ])
        rag_chain = create_stuff_documents_chain(llm, prompt_template)

        # Persist everything needed on later reruns
        st.session_state.chain = create_retrieval_chain(retriever, rag_chain)
        st.session_state.chunks = chunks
        st.session_state.llm = llm
        st.session_state.num_pages = len(docs)
        st.session_state.processed_file = uploaded_file.name

        # New document — clear previous results
        st.session_state.analysis_done = False
        st.session_state.messages = []

    # Read back from session state on every rerun (fast)
    chunks = st.session_state.chunks
    chain = st.session_state.chain
    llm = st.session_state.llm
    num_pages = st.session_state.num_pages

    # Document stats
    m1, m2, m3 = st.columns(3)
    m1.metric("Pages", num_pages)
    m2.metric("Chunks", len(chunks))
    m3.metric("Status", "Ready")

    st.divider()

    # Auto-detect page numbers
    detected_kpi_page = detect_kpi_page(chunks)
    detected_mda_page = detect_mda_page(chunks)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 KPI page auto-detected: Page {detected_kpi_page + 1}")
        kpi_page = st.number_input(
            "KPI Page Number (change if incorrect)",
            min_value=1, max_value=num_pages,
            value=min(detected_kpi_page + 1, num_pages),
            help="Check the report's table of contents if the detected page is wrong"
        )
    with col2:
        st.info(f"📝 MD&A page auto-detected: Page {detected_mda_page + 1}")
        mda_start_page = st.number_input(
            "MD&A Start Page (change if incorrect)",
            min_value=1, max_value=num_pages,
            value=min(detected_mda_page + 1, num_pages),
            help="The Management Discussion and Analysis section start page"
        )

    kpi_page_index = int(kpi_page) - 1
    mda_page_index = int(mda_start_page) - 1

    st.divider()

    # ─── Generate Analysis ────────────────────────────────────────────────────
    if st.button("🚀 Generate Analysis", type="primary", use_container_width=True):

        progress = st.progress(0, text="Starting analysis...")

        # Module 1 — Executive Summary
        progress.progress(10, text="Generating executive summary...")
        summary_result = chain.invoke({
            "input": "Provide a concise executive summary of this annual report covering business overview ,key financial performance highlights, and management outlook in 5-6 lines."
        })
        st.session_state.summary_output = summary_result["answer"]

        # Module 2 — Financial Highlights (direct injection)
        progress.progress(35, text="Extracting financial highlights...")
        kpi_content = ""
        for chunk in chunks:
            page = chunk.metadata.get("page", 0)
            if page in (kpi_page_index, kpi_page_index + 1):
                kpi_content += chunk.page_content

        kpi_prompt = f"""You are a senior financial analyst.
The following text contains key performance indicators from a company's annual report.
The data may appear jumbled due to multi-column PDF extraction.

First identify what sector this company operates in based on the metrics present.
Then extract every financial and operational metric you can find, matching each
metric name to its correct value, and present them as a clean structured list
grouped by category.

Only report metrics that actually appear in the text. Do not report a metric as
"Not Provided" — simply omit anything that is not present.

{kpi_content}"""
        st.session_state.kpi_output = llm.invoke(kpi_prompt).content

        # Module 3 — Risk Factors
        progress.progress(60, text="Analysing risk factors...")
        risk_result = chain.invoke({
            "input": "What is the company's risk management framework? What are the key "
                     "risks including credit risk, interest rate risk, liquidity risk and "
                     "operational risk? How does management monitor and mitigate each risk?"
        })
        st.session_state.risk_output = risk_result["answer"]

        # Module 4 — Management Sentiment (direct injection)
        progress.progress(85, text="Assessing management sentiment...")
        sentiment_content = ""
        for chunk in chunks:
            page = chunk.metadata.get("page", 0)
            if mda_page_index <= page <= mda_page_index + 5:
                sentiment_content += chunk.page_content

        sentiment_prompt = f"""You are a senior Equity Research Analyst.
The following text contains the MD letter to shareholders and the Management
Discussion and Analysis section.
Assess the overall management sentiment — Positive, Cautiously Optimistic,
Neutral, or Concerned.
Identify 4 specific themes supporting your assessment.
Note any specific guidance or targets for the coming year.
Keep the response under 300 words.

{sentiment_content}"""
        st.session_state.sentiment_output = llm.invoke(sentiment_prompt).content

        progress.progress(100, text="Analysis complete")
        st.session_state.analysis_done = True
        progress.empty()

# ─── Results — outside every conditional so they survive reruns ───────────────
if st.session_state.analysis_done:
    st.divider()
    st.subheader("📋 Analysis Results")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Executive Summary",
        "💰 Financial Highlights",
        "⚠️ Risk Factors",
        "🎯 Management Sentiment"
    ])
    with tab1:
        st.markdown(st.session_state.summary_output)
    with tab2:
        st.markdown(st.session_state.kpi_output)
    with tab3:
        st.markdown(st.session_state.risk_output)
    with tab4:
        st.markdown(st.session_state.sentiment_output)

    # ─── Chat ─────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Ask anything about this report")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask a question about the annual report..."):
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
            st.markdown(user_question)

        answer = ""
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                response = st.session_state.chain.invoke({"input": user_question})
                answer = response["answer"]
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )