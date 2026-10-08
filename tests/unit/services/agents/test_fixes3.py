"""Tests for Week 7 fix script 3: status, honest reasoning steps, fail-open guardrail, raising generate_answer."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from src.schemas.api.ask import AgenticAskResponse
from src.services.agents.agentic_rag import AgenticRAGService, determine_status
from src.services.agents.nodes.generate_answer_node import ainvoke_generate_answer_step
from src.services.agents.nodes.guardrail_node import ainvoke_guardrail_step, continue_after_guardrail

GUARD_OK = SimpleNamespace(score=75)
GUARD_LOW = SimpleNamespace(score=40)
GRADE_YES = SimpleNamespace(is_relevant=True)
GRADE_NO = SimpleNamespace(is_relevant=False)


def test_status_out_of_scope():
    assert determine_status({"guardrail_result": GUARD_LOW, "grading_results": []}, 60) == "out_of_scope"


def test_status_answered():
    assert determine_status({"guardrail_result": GUARD_OK, "grading_results": [GRADE_YES]}, 60) == "answered"


def test_status_answered_after_a_no_then_a_yes():
    result = {"guardrail_result": GUARD_OK, "grading_results": [GRADE_YES]}  # only the last verdict is kept
    assert determine_status(result, 60) == "answered"


def test_status_no_relevant_papers():
    assert determine_status({"guardrail_result": GUARD_OK, "grading_results": [GRADE_NO]}, 60) == "no_relevant_papers"
    assert determine_status({"guardrail_result": GUARD_OK, "grading_results": []}, 60) == "no_relevant_papers"


def test_status_when_guardrail_failed_is_not_out_of_scope():
    assert determine_status({"guardrail_result": None, "grading_results": [GRADE_YES]}, 60) == "answered"


def _steps(result, status):
    return AgenticRAGService._extract_reasoning_steps(None, result, status)


def test_last_step_only_when_answered():
    base = {"guardrail_result": GUARD_OK, "retrieval_attempts": 1, "grading_results": [GRADE_YES]}
    assert _steps(base, "answered")[-1] == "Generated answer from context"
    assert "Generated answer from context" not in _steps(base, "no_relevant_papers")
    assert _steps({"guardrail_result": GUARD_LOW}, "out_of_scope") == ["Validated query scope (score: 40/100)"]


def _runtime(ollama_client):
    ctx = SimpleNamespace(
        langfuse_enabled=False,
        trace=None,
        guardrail_threshold=60,
        model_name="test-model",
        temperature=0.0,
        ollama_client=ollama_client,
    )
    return SimpleNamespace(context=ctx)


class BrokenOllama:
    def get_langchain_model(self, **kwargs):
        raise RuntimeError("ollama is down")


async def test_guardrail_failure_lets_the_question_through():
    runtime = _runtime(BrokenOllama())
    state = {"messages": [HumanMessage(content="What are self-evolving multi-agent systems")]}

    update = await ainvoke_guardrail_step(state, runtime)

    assert update == {"guardrail_result": None}
    assert continue_after_guardrail({"guardrail_result": None}, runtime) == "continue"


async def test_generate_answer_failure_is_raised_not_returned_as_an_answer():
    class BrokenLLM:
        async def ainvoke(self, prompt):
            raise RuntimeError("model timed out")

    class OllamaWithBrokenLLM:
        def get_langchain_model(self, **kwargs):
            return BrokenLLM()

    runtime = _runtime(OllamaWithBrokenLLM())
    state = {"messages": [HumanMessage(content="q")], "relevant_sources": []}

    with pytest.raises(RuntimeError, match="model timed out"):
        await ainvoke_generate_answer_step(state, runtime)


def test_response_status_must_be_one_of_the_three_values():
    fields = dict(
        query="q", answer="a", sources=[], chunks_used=3, search_mode="hybrid", reasoning_steps=[], retrieval_attempts=0
    )
    assert AgenticAskResponse(**fields).status == "answered"
    assert AgenticAskResponse(status="out_of_scope", **fields).status == "out_of_scope"
    with pytest.raises(ValidationError):
        AgenticAskResponse(status="bogus", **fields)
