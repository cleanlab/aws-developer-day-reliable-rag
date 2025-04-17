import os
from typing import TypedDict, cast

from cleanlab_codex.validator import BadResponseThresholds, Validator
from cleanlab_tlm import TLM
from cleanlab_tlm.tlm import TLMResponse
from cleanlab_tlm.utils.rag import Eval as TrustworthyRAGEval
from cleanlab_tlm.utils.rag import get_default_evals

from constants import (
    EVAL_THRESHOLD_OVERRIDES,
    PROMPT_TEMPLATE,
    QUERY_POLITENESS_CRITERIA,
    QUERY_POLITENESS_QUERY_IDENTIFIER,
    RELATED_TO_COMPETITOR_CRITERIA,
    RELATED_TO_COMPETITOR_QUERY_IDENTIFIER,
    RETRIEVAL_RESULTS,
    SIMILARITY_SCORE_THRESHOLD,
    TLM_PROMPT_QUALITY_PRESET,
)
from knowledge_base import KnowledgeBase


class Eval(TypedDict):
    name: str
    score: float
    is_bad: bool


class Response(TypedDict):
    response: str
    is_bad_response: bool
    is_expert_answer: bool
    evals: list[Eval]


# Custom Eval
related_to_competitor_eval = TrustworthyRAGEval(
    name="related_to_competitor",
    criteria=RELATED_TO_COMPETITOR_CRITERIA.format(company_name="Cursor", competitors="code editors or integrated development environments such as VSCode, JetBrains, and Codeium Windsurf"),  # noqa: E501
    query_identifier=RELATED_TO_COMPETITOR_QUERY_IDENTIFIER,
)

politeness_eval = TrustworthyRAGEval(
    name="politeness",
    criteria=QUERY_POLITENESS_CRITERIA,
    query_identifier=QUERY_POLITENESS_QUERY_IDENTIFIER,
)


class RAG:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._retriever = knowledge_base.as_retriever(similarity_top_k=RETRIEVAL_RESULTS)
        self._llm = TLM(quality_preset=TLM_PROMPT_QUALITY_PRESET)

        evals = get_default_evals() + [related_to_competitor_eval, politeness_eval]
        thresholds = {**EVAL_THRESHOLD_OVERRIDES, "related_to_competitor": 0, "politeness": 0.7}
        self._validator = Validator(
            codex_access_key=os.environ["CLEANLAB_CODEX_ACCESS_KEY"],
            tlm_api_key=os.environ["CLEANLAB_TLM_API_KEY"],
            trustworthy_rag_config={"evals": evals},
            bad_response_thresholds=BadResponseThresholds.model_validate(thresholds).model_dump(),
        )

    def _retrieve(self, question: str) -> list[str]:
        context_nodes = self._retriever.retrieve(question)
        return [
            node.text
            for node in context_nodes
            if node.score is not None and node.score >= SIMILARITY_SCORE_THRESHOLD
        ]

    def _format_contexts(self, contexts: list[str]) -> str:
        return "\n\n".join(f"Context Chunk {index}:\n{context}" for index, context in enumerate(contexts, 1))

    def _format_prompt(self, question: str, context: str) -> str:
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def _generate(self, question: str, context: str) -> str:
        prompt = self._format_prompt(question, context)
        response = cast(TLMResponse, self._llm.prompt(prompt))["response"]
        assert isinstance(response, str)
        return response

    def query(self, question: str) -> Response:
        contexts = self._retrieve(question)
        context = self._format_contexts(contexts)
        initial_response = self._generate(question, context)

        validation_results = self._validator.validate(
            query=question, context=context, response=initial_response, form_prompt=self._format_prompt
        )
        initial_response_is_bad = validation_results.pop("is_bad_response")
        expert_answer = validation_results.pop("expert_answer")
        eval_results = [
            Eval(name=eval, score=result["score"], is_bad=result["is_bad"])
            for eval, result in validation_results.items()
        ]

        if expert_answer is not None:
            return {
                "response": expert_answer,
                "is_bad_response": False,
                "is_expert_answer": True,
                "evals": eval_results,
            }

        return {
            "response": initial_response,
            "is_bad_response": initial_response_is_bad,
            "is_expert_answer": False,
            "evals": eval_results,
        }
