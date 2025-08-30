from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Path
PERSIST_DIR = "vector_Store"

# Load OPENAI_API_KEY
load_dotenv()

# ==== Embeddings & LLM ====
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7})

SYSTEM = """You are a precise book analyst. Answer using ONLY the provided context.
If the answer is not in the context, say you don't know.
Cite sources as (page X, source_name) when possible."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question:\n{question}\n\nContext:\n{context}")
])


def format_context(docs):
    out = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_name", d.metadata.get("source", ""))
        page = d.metadata.get("page")
        head = f"[{i}] ({src}, page {page+1 if isinstance(page, int) else page})"
        text = d.page_content.strip()
        out.append(f"{head}\n{text}")
    return "\n\n".join(out)


def answer(question: str):
    docs = retriever.invoke(question)
    ctx = format_context(docs)
    chain = prompt | llm | StrOutputParser()
    resp = chain.invoke({"question": question, "context": ctx})
    # List the sources
    sources = []
    for d in docs:
        src = d.metadata.get("source_name", d.metadata.get("source", ""))
        page = d.metadata.get("page")
        sources.append(f"{src}: page {page+1 if isinstance(page, int) else page}")
    print(resp)
    print("\nSources:")
    for s in dict.fromkeys(sources):  # Remove duplicates
        print("-", s)


if __name__ == "__main__":
    # Example: Question: 第2章の主張を要約して。
    while True:
        try:
            q = input("\nQuestion> ").strip()
            if not q: continue
            if q.lower() in {"exit","quit"}: break
            answer(q)
        except KeyboardInterrupt:
            break
