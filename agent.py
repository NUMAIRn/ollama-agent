"""
Ollama agentic chatbot
  question -> router -> resume RAG  (if about the resume owner)
                     -> web search  (Parallel Search API) otherwise
"""

import os
import re
import sys
from pathlib import Path

import ollama
from dotenv import load_dotenv
from parallel import Parallel

from rag import CHAT_MODEL, build_index, retrieve

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", CHAT_MODEL)
MIN_SCORE = 0.6
MAX_WEB_RESULTS = 5
MAX_EXCERPT_CHARS = 800   # keep context small for local models
HISTORY_TURNS = 4         # how many past messages to use for follow-up questions

client = Parallel(api_key=os.environ["PARALLEL_API_KEY"])


# ---------------------------------------------------------------------------
# 1. Resume RAG (uses your rag_chatbot.py)
# ---------------------------------------------------------------------------
CHUNKS: list[dict] = []
EMBEDDINGS = None


def load_resume(pdf_path: str) -> None:
    global CHUNKS, EMBEDDINGS
    CHUNKS, EMBEDDINGS = build_index(pdf_path)


def retrieve_resume(query: str) -> list[dict]:
    """Returns [{"page", "text", "score"}, ...] with cosine similarity scores."""
    return retrieve(query, CHUNKS, EMBEDDINGS)


# ---------------------------------------------------------------------------
# 2. Web search (Parallel)
# ---------------------------------------------------------------------------
def web_search(objective: str, queries: list[str] | None = None) -> list[dict]:
    try:
        search = client.search(
            objective=objective,
            search_queries=queries or [objective],
            mode="fast",  # "turbo" = fastest, "advanced" = best quality (~3s)
        )
        return [
            {
                "title": r.title,
                "url": r.url,
                "body": "\n".join(r.excerpts)[:MAX_EXCERPT_CHARS],
            }
            for r in search.results[:MAX_WEB_RESULTS]
        ]
    except Exception as e:
        print(f"[search error] {e}")
        return []


# ---------------------------------------------------------------------------
# 3. LLM helpers
# ---------------------------------------------------------------------------
def llm(prompt: str, temperature: float = 0.2) -> str:
    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return r["message"]["content"].strip()


def format_history(history: list[dict]) -> str:
    recent = history[-HISTORY_TURNS:]
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)


def rewrite_question(question: str, history: list[dict]) -> str:
    """Turn follow-ups like 'and where did he study?' into standalone questions."""
    if not history:
        return question
    prompt = f"""Rewrite the final question so it is fully standalone, using the
conversation for context. Output ONLY the rewritten question.

Conversation:
{format_history(history)}

Final question: {question}
Standalone question:"""
    return llm(prompt, temperature=0) or question


SMALL_TALK = re.compile(
    r"^\s*(hi|hii+|hey+|hello+|yo|hola|sup|howdy|good\s+(morning|afternoon|evening)|"
    r"thanks?( you)?|thank u|ty|ok(ay)?|cool|nice|great|bye|goodbye|see you|"
    r"how are you|how\'?s it going|what\'?s up|who are you|what can you do|help)"
    r"[\s!.?,]*$",
    re.IGNORECASE,
)


def route(question: str) -> str:
    """Return 'CHAT', 'RESUME' or 'WEB'."""
    # Fast path: greetings and small talk never need retrieval or search
    if SMALL_TALK.match(question):
        return "CHAT"

    prompt = f"""Classify the message into exactly one category. Reply with ONE word.

CHAT = greetings, thanks, small talk, or questions about the assistant itself
       (e.g. "hello", "thanks!", "who are you?", "what can you do?")
RESUME = asks about the resume owner personally (their skills, experience,
       education, projects, certifications, contact info, background, facts about him)
WEB = a factual or general-knowledge question that needs looking up
       (news, definitions, how-to, facts about other people/places/things)

Message: {question}
Category:"""
    out = llm(prompt, temperature=0).upper()
    for label in ("CHAT", "RESUME", "WEB"):
        if label in out:
            return label
    return "WEB"


def make_search_objective(question: str) -> tuple[str, list[str]]:
    """Ask the LLM for a clear objective + short keyword queries."""
    prompt = f"""Turn the question into a web search plan.
Line 1: one sentence objective describing what to find.
Lines 2-3: short keyword search queries (2-6 words each).
Output only those lines, nothing else.

Question: {question}"""
    lines = [l.strip("-• ").strip() for l in llm(prompt, 0).splitlines() if l.strip()]
    if len(lines) < 2:
        return question, [question]
    return lines[0], lines[1:3]


# ---------------------------------------------------------------------------
# 4. Answer paths
# ---------------------------------------------------------------------------
def answer_chat(question: str, history: list[dict]) -> str:
    prompt = f"""You are a friendly assistant that can answer questions about the
resume owner and look things up on the web. Reply briefly and naturally to the
message below. Do not invent facts.

Conversation so far:
{format_history(history)}

Message: {question}
Reply:"""
    return llm(prompt, temperature=0.5)


def answer_from_resume(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks)
    prompt = f"""You are answering questions about a person using their resume.
Use ONLY the resume excerpts below. If the answer isn't there, say you don't
see that in the resume. Speak about the person in the third person.

Resume excerpts:
{context}

Question: {question}
Answer:"""
    return llm(prompt)


def answer_from_web(question: str) -> str:
    objective, queries = make_search_objective(question)
    results = web_search(objective, queries)
    if not results:
        return "I couldn't find anything online for that."

    context = "\n\n".join(
        f"[{i}] {r['title']}\n{r['body']}\nSource: {r['url']}"
        for i, r in enumerate(results, 1)
    )
    prompt = f"""Answer the question using ONLY the search results below.
Cite sources with their numbers like [1]. If the results don't answer the
question, say so.

Search results:
{context}

Question: {question}
Answer:"""
    answer = llm(prompt)
    sources = "\n".join(f"[{i}] {r['url']}" for i, r in enumerate(results, 1))
    return f"{answer}\n\nSources:\n{sources}"


# ---------------------------------------------------------------------------
# 5. The agent
# ---------------------------------------------------------------------------
def ask(question: str, history: list[dict]) -> tuple[str, str]:
    """Returns (answer, source_label)."""
    # Greetings / small talk: skip rewriting, retrieval and search entirely
    if SMALL_TALK.match(question):
        return answer_chat(question, history), "chat"

    standalone = rewrite_question(question, history)
    kind = route(standalone)

    if kind == "CHAT":
        return answer_chat(question, history), "chat"

    if kind == "RESUME":
        chunks = retrieve_resume(standalone)
        good = [c for c in chunks if c["score"] >= MIN_SCORE]
        if good:
            pages = sorted({c["page"] for c in good})
            label = "resume, pages " + ", ".join(map(str, pages))
            return answer_from_resume(standalone, good), label
        # router said resume, but retrieval is weak -> fall back to web

    return answer_from_web(standalone), "web"


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python agent.py path/to/resume.pdf")
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        sys.exit(f"File not found: {pdf_path}")
    load_resume(pdf_path)

    print(f"Agent ready (model: {MODEL}). Type 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        answer, source = ask(q, history)
        print(f"\nBot [{source}]: {answer}\n")
        history += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": answer},
        ]


if __name__ == "__main__":
    main()
