from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import regex as re


# Path
PDF_PATH = "books/AWSではじめる生成AI.pdf"
PERSIST_DIR = "vector_store"

# Parameters
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 300

# Load OPENAI_API_KEY
load_dotenv()

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Normalize Japanese texts
def normalize_text(t: str) -> str:
    if not t:
        return ""

    t = t.replace("\u00A0", " ")                 # NBSP → 半角スペース
    t = re.sub(r"[ \t]+", " ", t)                # 連続スペース圧縮
    t = re.sub(r"\s*\n\s*", "\n", t)             # 行頭末スペース除去
    t = re.sub(r"\n{3,}", "\n\n", t)             # 3連改行→2

    return t.strip()


# === Load PDF ===
pdf = Path(PDF_PATH)
assert pdf.exists(), f"PDF not found: {pdf.resolve()}"
# loader = PyPDFLoader(str(pdf))
loader = PyMuPDFLoader(str(pdf))
docs = loader.load()  # page_content + metadata: {"page": n, "source": path}

# === Add metadata ===
for d in docs:
    d.metadata["source_name"] = pdf.name
    d.page_content = normalize_text(d.page_content)

# === Create chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                          separators=[
                                                "\n\n", "\n", "。", "！", "？", "．", "…",
                                                "、", "，", "：", "；", "—", "・", " ", ""
                                          ])
chunks = splitter.split_documents(docs)

# ==== Build vector store ====
vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
print(f"✅ Indexed {len(chunks)} chunks from {pdf.name} into {PERSIST_DIR}")
