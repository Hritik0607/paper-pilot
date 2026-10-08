"""Tests for the Week 7 fix: sources are built from the search tool's structured result."""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage

from src.services.agents.nodes.utils import extract_sources_from_tool_messages
from src.services.agents.tools import create_retriever_tool, format_documents_for_llm


class FakeEmbeddings:
    async def embed_query(self, query):
        return [0.0, 0.0]


class FakeOpenSearch:
    def __init__(self, hits):
        self.hits = hits

    def search_unified(self, **kwargs):
        return {"total": len(self.hits), "hits": self.hits}


HITS = [
    {"chunk_text": "Transformers use self-attention.", "arxiv_id": "1706.03762", "title": "Attention Is All You Need", "authors": "A, B", "score": 2.5},
    {"chunk_text": "More about attention.", "arxiv_id": "1706.03762", "title": "Attention Is All You Need", "authors": "A, B", "score": 1.5},
    {"chunk_text": "BERT pre-training.", "arxiv_id": "1810.04805", "title": "BERT", "authors": ["C"], "score": 1.0},
]


@pytest.mark.asyncio
async def test_tool_returns_text_and_documents():
    tool = create_retriever_tool(FakeOpenSearch(HITS), FakeEmbeddings(), top_k=3)
    message = await tool.ainvoke({"name": "retrieve_papers", "args": {"query": "transformers"}, "id": "retrieve_1", "type": "tool_call"})
    assert isinstance(message, ToolMessage)
    assert "[1] Attention Is All You Need (arXiv:1706.03762)" in message.content
    assert len(message.artifact) == 3


@pytest.mark.asyncio
async def test_empty_search_gives_empty_text():
    tool = create_retriever_tool(FakeOpenSearch([]), FakeEmbeddings(), top_k=3)
    message = await tool.ainvoke({"name": "retrieve_papers", "args": {"query": "x"}, "id": "retrieve_1", "type": "tool_call"})
    assert message.content == ""
    assert message.artifact == []


@pytest.mark.asyncio
async def test_sources_are_unique_papers_with_urls():
    tool = create_retriever_tool(FakeOpenSearch(HITS), FakeEmbeddings(), top_k=3)
    message = await tool.ainvoke({"name": "retrieve_papers", "args": {"query": "transformers"}, "id": "retrieve_1", "type": "tool_call"})
    sources = extract_sources_from_tool_messages([HumanMessage(content="q"), message])
    assert [s.arxiv_id for s in sources] == ["1706.03762", "1810.04805"]
    assert sources[0].url == "https://arxiv.org/pdf/1706.03762.pdf"
    assert sources[0].authors == ["A", "B"]


def test_no_tool_message_means_no_sources():
    assert extract_sources_from_tool_messages([HumanMessage(content="q")]) == []


def test_format_documents_empty():
    assert format_documents_for_llm([]) == ""
    assert "Title" in format_documents_for_llm([Document(page_content="text", metadata={"title": "Title", "arxiv_id": "1"})])
