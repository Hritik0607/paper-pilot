import json
import logging
from typing import AsyncIterator

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_MODEL = "llama3.2:1b"
MODEL_CHOICES = ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"]

# A small local model on CPU can take minutes (guardrail + grading + rewrite + answer).
# 10 minutes for the whole request, 10 seconds to connect.
BASIC_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
AGENTIC_TIMEOUT = httpx.Timeout(900.0, connect=10.0)


def _format_search_info(sources: list, chunks_used: int, search_mode: str) -> str:
    """Build the 'Search Info' block shown under a basic RAG answer."""
    text = "\n\n**Search Info:**\n"
    text += f"- Mode: {search_mode}\n"
    text += f"- Chunks used: {chunks_used}\n"
    if sources:
        text += f"- Sources: {len(sources)} papers\n"
        for i, source in enumerate(sources[:3], 1):
            text += f"  {i}. [{source.split('/')[-1]}]({source})\n"
        if len(sources) > 3:
            text += f"  ... and {len(sources) - 3} more\n"
    return text


async def stream_response(
    query: str, top_k: int = 3, use_hybrid: bool = True, model: str = DEFAULT_MODEL, categories: str = ""
) -> AsyncIterator[str]:
    """BASIC RAG: stream the answer from POST /api/v1/stream."""
    if not query.strip():
        yield "Please enter a question."
        return

    category_list = [c.strip() for c in categories.split(",") if c.strip()] if categories else None
    payload = {"query": query, "top_k": int(top_k), "use_hybrid": use_hybrid, "model": model, "categories": category_list}
    sources: list = []
    chunks_used = 0
    search_mode = ""
    current_answer = ""

    yield "⏳ Searching papers and generating the answer..."

    try:
        async with httpx.AsyncClient(timeout=BASIC_TIMEOUT) as client:
            async with client.stream(
                "POST", f"{API_BASE_URL}/stream", json=payload, headers={"Accept": "text/plain"}
            ) as response:
                if response.status_code != 200:
                    yield f"Error: API returned status {response.status_code}"
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue  # skip malformed lines

                    if "error" in data:
                        yield f"Error: {data['error']}"
                        return

                    if "sources" in data:
                        sources = data["sources"]
                        chunks_used = data.get("chunks_used", 0)
                        search_mode = data.get("search_mode", "unknown")
                        continue

                    if "chunk" in data:
                        current_answer += data["chunk"]
                        yield current_answer + (
                            _format_search_info(sources, chunks_used, search_mode) if (sources or chunks_used) else ""
                        )

                    if data.get("done", False):
                        current_answer = data.get("answer", current_answer)
                        yield current_answer + (
                            _format_search_info(sources, chunks_used, search_mode) if (sources or chunks_used) else ""
                        )
                        break

    except httpx.RequestError as e:
        yield f"Connection error: {e!r}\nMake sure the API server is running at {API_BASE_URL}"
    except Exception as e:
        yield f"Unexpected error: {e}"


async def agentic_response(query: str, model: str = DEFAULT_MODEL) -> AsyncIterator[str]:
    """AGENTIC RAG: one call to POST /api/v1/ask-agentic, then show answer + reasoning."""
    if not query.strip():
        yield "Please enter a question."
        return

    yield (
        "⏳ The agent is working: checking scope → searching → grading → (maybe rewriting) → answering.\n\n"
        "On a small local model this can take **several minutes**. Please wait."
    )

    payload = {"query": query, "model": model}

    try:
        async with httpx.AsyncClient(timeout=AGENTIC_TIMEOUT) as client:
            response = await client.post(f"{API_BASE_URL}/ask-agentic", json=payload)

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            yield f"Error {response.status_code}: {detail}"
            return

        data = response.json()
    except httpx.RequestError as e:
        yield f"Connection error: {e!r}\nMake sure the API server is running at {API_BASE_URL}"
        return
    except Exception as e:
        yield f"Unexpected error: {e}"
        return

    # ---- Build the display text ----
    out = data.get("answer", "") or "_(empty answer)_"

    status = data.get("status", "answered")
    if status == "out_of_scope":
        out += "\n\n---\n\n🚫 **Out of scope** - the agent stopped before searching any papers.\n"
    else:
        if status == "no_relevant_papers":
            out += "\n\n---\n\n🔎 **No relevant papers were found** after the allowed attempts.\n"
        sources = data.get("sources", []) or []
        out += "\n\n---\n\n**Sources:**\n"
        if sources:
            for i, src in enumerate(sources, 1):
                out += f"{i}. [{str(src).split('/')[-1]}]({src})\n"
        else:
            out += "_none (the agent did not keep any document as relevant)_\n"

        out += f"\n**Retrieval attempts:** {data.get('retrieval_attempts', 0)}\n"

    if data.get("rewritten_query"):
        out += f"\n**Question was rewritten to:** {data['rewritten_query']}\n"

    steps = data.get("reasoning_steps", []) or []
    if steps:
        out += "\n**Reasoning steps:**\n"
        for i, step in enumerate(steps, 1):
            out += f"{i}. {step}\n"

    if data.get("trace_id"):
        out += f"\n_Langfuse trace id: `{data['trace_id']}`_\n"

    yield out


def create_gradio_interface():
    """Create the two-tab Gradio interface: Agentic RAG and Basic RAG."""

    with gr.Blocks(title="Paper Pilot - RAG Chat", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🔬 Paper Pilot - RAG Chat

            Ask questions about machine learning and AI research papers from arXiv.
            Choose between basic RAG or agentic RAG with intelligent query refinement.
            """
        )

        with gr.Tabs():
            # ---------------- Agentic tab ----------------
            with gr.Tab("🤖 Agentic RAG"):
                gr.Markdown(
                    """
                    ### Intelligent RAG with Query Refinement

                    The agentic system will:
                    - Validate your query is within scope
                    - Retrieve relevant papers
                    - Grade document relevance
                    - Rewrite the question if needed for better results
                    - Show its reasoning steps

                    ⚠️ Slow on a small local model: several minutes per question.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        agentic_query = gr.Textbox(
                            label="Your Question",
                            placeholder="What are the latest advances in transformer architectures?",
                            lines=2,
                            max_lines=5,
                        )
                    with gr.Column(scale=1):
                        agentic_btn = gr.Button("Ask with Intelligence", variant="primary", size="lg")

                with gr.Accordion("Model Options", open=False):
                    agentic_model = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL,
                        label="LLM Model",
                        info="Larger models may give better answers but are slower",
                    )

                agentic_output = gr.Markdown(value="Ask a question to see the agentic RAG in action!")

                agentic_btn.click(
                    fn=agentic_response,
                    inputs=[agentic_query, agentic_model],
                    outputs=[agentic_output],
                    show_progress="minimal",
                )
                agentic_query.submit(
                    fn=agentic_response,
                    inputs=[agentic_query, agentic_model],
                    outputs=[agentic_output],
                    show_progress="minimal",
                )

            # ---------------- Basic tab ----------------
            with gr.Tab("⚡ Basic RAG"):
                gr.Markdown(
                    """
                    ### Basic RAG with streaming

                    Search the papers, then stream the answer. No guardrail, grading or rewriting.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        basic_query = gr.Textbox(
                            label="Your Question",
                            placeholder="What are transformers in machine learning?",
                            lines=2,
                            max_lines=5,
                        )
                    with gr.Column(scale=1):
                        basic_btn = gr.Button("Ask Question", variant="primary", size="lg")

                with gr.Accordion("Advanced Options", open=False):
                    basic_top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of chunks to retrieve",
                        info="More chunks = more context but slower generation",
                    )
                    basic_hybrid = gr.Checkbox(
                        value=True,
                        label="Use hybrid search (BM25 + vector embeddings)",
                        info="Usually better results than keyword-only search",
                    )
                    basic_model = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL,
                        label="LLM Model",
                        info="Larger models may give better answers but are slower",
                    )
                    basic_categories = gr.Textbox(
                        label="arXiv Categories (optional)",
                        placeholder="cs.AI, cs.LG",
                        info="Comma-separated. Leave empty for all categories",
                    )

                basic_output = gr.Markdown(value="Ask a question to get started!")

                basic_btn.click(
                    fn=stream_response,
                    inputs=[basic_query, basic_top_k, basic_hybrid, basic_model, basic_categories],
                    outputs=[basic_output],
                    show_progress="minimal",
                )
                basic_query.submit(
                    fn=stream_response,
                    inputs=[basic_query, basic_top_k, basic_hybrid, basic_model, basic_categories],
                    outputs=[basic_output],
                    show_progress="minimal",
                )

        gr.Markdown(
            """
            ---
            **Note**: The RAG API must be running at `http://localhost:8000`.
            Each question is answered on its own: this chat has no memory of earlier questions.
            """
        )

    return interface


def main():
    """Main entry point for the Gradio app"""
    print("🚀 Starting Paper Pilot Gradio Interface...")
    print(f"📡 API Base URL: {API_BASE_URL}")

    interface = create_gradio_interface()

    # Long requests: allow a few to queue instead of failing.
    interface.queue()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Changed to avoid port conflict
        share=False,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()