import os
from typing import TypedDict, cast

from cleanlab_codex.validator import BadResponseThresholds, Validator
from cleanlab_tlm import TLM
from cleanlab_tlm.tlm import TLMResponse
from cleanlab_tlm.utils.rag import Eval as TrustworthyRAGEval  # noqa: F401
from cleanlab_tlm.utils.rag import get_default_evals

from constants import (
    EVAL_THRESHOLD_OVERRIDES,
    PROMPT_TEMPLATE,  # noqa: F401
    QUERY_POLITENESS_CRITERIA,  # noqa: F401
    QUERY_POLITENESS_QUERY_IDENTIFIER,  # noqa: F401
    RELATED_TO_COMPETITOR_CRITERIA,  # noqa: F401
    RELATED_TO_COMPETITOR_QUERY_IDENTIFIER,  # noqa: F401
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


class RAG:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        # retriever that connects to the knowledge base to retrieve context relevant to a user question
        self._retriever = knowledge_base.as_retriever(similarity_top_k=RETRIEVAL_RESULTS)
        # LLM that will be used to generate responses to user questions
        # you can replace this with any other LLM that you want to use
        self._llm = TLM(quality_preset=TLM_PROMPT_QUALITY_PRESET)
        # validator that will be used to detect bad responses and remediate them where possible
        # we use a default set of evaluations and thresholds for now, but you can change or add to these
        # to customize validation for your use case

        # TODO Part 4.1a: Define custom evaluations for your RAG system.
        # To define a custom evaluation, you'll need to specify a `criteria` string which describes in natural language
        # what you want to evaluate. If your evaluation references the user's query, the retrieved context, or the
        # LLM response, you'll need to include the `query_identifier`, `context_identifier`, or `response_identifier`
        # argument to specify how you refer to the query, context, or response in the criteria string.
        # See the [TrustworthyRAG docs](https://help.cleanlab.ai/tlm/api/python/utils.rag/#class-eval) for more
        # information on defining custom evals.
        # We've provided two examples of criteria (RELATED_TO_COMPETITOR_CRITERIA and QUERY_POLITENESS_CRITERIA) that
        # as well as the query identifiers for each (RELATED_TO_COMPETITOR_QUERY_IDENTIFIER and
        # QUERY_POLITENESS_QUERY_IDENTIFIER) that you can use.

        # TODO Part 4.1b: Add custom evaluations to your validator.
        # To add custom evaluations to your validator, you'll need to include them in the `evals` portion of your
        # Validator's `trustworthy_rag_config`.
        # See the [Cleanlab Codex Validator docs](https://help.cleanlab.ai/codex/api/python/validator)
        # and [TrustworthyRAG docs](https://help.cleanlab.ai/tlm/api/python/utils.rag/#class-trustworthyrag)
        # for more information.
        # You'll also need to update the `bad_response_thresholds` argument to specify thresholds for your custom eval.
        # If the custom eval's score is **below** the threshold, `Validator` will consider a response to be bad.
        # You can see the [Advanced Usage](https://help.cleanlab.ai/codex/tutorials/other_rag_frameworks/validator/#advanced-usage)
        # section of the [Codex as Backup](https://help.cleanlab.ai/codex/tutorials/other_rag_frameworks/validator/)
        # tutorial for more information on how to add custom evaluations and thresholds to your Validator.
        self._validator = Validator(
            codex_access_key=os.environ["CLEANLAB_CODEX_ACCESS_KEY"],
            tlm_api_key=os.environ["CLEANLAB_TLM_API_KEY"],
            trustworthy_rag_config={
                "evals": get_default_evals(),
            },
            bad_response_thresholds=BadResponseThresholds.model_validate(EVAL_THRESHOLD_OVERRIDES).model_dump(),
        )

    def _retrieve(self, question: str, *, similarity_threshold: float = SIMILARITY_SCORE_THRESHOLD) -> list[str]:
        """
        Retrieves a list of context from the knowledge base that is relevant to the given question.
        This method returns a list of strings, each representing a context chunk. Only context chunks
        with a similarity score greater than or equal to similarity_threshold will be included.
        Args:
            question (str): The user question to retrieve context for.
            similarity_threshold (float): The similarity score threshold for context chunks to be included.
        Returns:
            list[str]: A list of context chunks that are relevant to the given question.
        """
        context_nodes = self._retriever.retrieve(question)
        return [node.text for node in context_nodes if node.score is not None and node.score >= similarity_threshold]

    def _format_contexts(self, contexts: list[str]) -> str:
        """
        Formats the list of retrieved contexts into a single string. This string will be included
        in the prompt that is sent to the LLM.
        Args:
            contexts (list[str]): The list of individual retrieved contexts to format.
        Returns:
            str: The formatted context string.
        """
        # TODO Part 1.2a: implement this method to format the retrieved contexts into a single string
        # You may want to include things like spacing between each context chunk, enumeration of the
        # context chunks, etc. to help the LLM understand that these are separate pieces of information.
        return ""

    def _format_prompt(self, question: str, context: str) -> str:
        """
        Formats the final prompt that is sent to the LLM using the PROMPT_TEMPLATE.
        This prompt will include the original user question, the formatted context string, and additional instructions
        for the LLM (included in PROMPT_TEMPLATE).
        Args:
            question (str): The user question to generate a response for.
            context (str): The formatted context string to use in the prompt.
        Returns:
            str: The final prompt to send to the LLM.
        """
        # TODO Part 1.2b: implement this method to format the final prompt passed to the LLM.
        # You will need to use the PROMPT_TEMPLATE and the formatted context string.
        return ""

    def _generate(self, question: str, context: str) -> str:
        """
        Generates an LLM response for the given question using the retrieved context.
        Args:
            question (str): The user question to generate a response for.
            context (str): The formatted context string to use in the prompt.
        Returns:
            str: The LLM response to the user question.
        """
        prompt = self._format_prompt(question, context)
        response = cast(TLMResponse, self._llm.prompt(prompt))["response"]
        assert isinstance(response, str)
        return response

    def query(self, question: str) -> Response:
        """
        Queries the RAG system with the given question and returns a response along with some metadata
        describing whether the response is bad, whether the LLM response was replaced with an expert answer,
        and scores for various evaluations run on the LLM response.
        Args:
            question (str): The user question to generate a response for.
        Returns:
            Response: A dictionary containing the LLM response, whether the response is bad, whether response came
            from a subject matter expert (rather than the LLM), and scores for evaluations run on the LLM response.
        """
        # TODO Part 1.2c: Use the _retrieve and _generate methods to query the RAG system and return the LLM response.
        # For now, we will return the LLM response directly without running any detection or remediation.
        # You can set the metadata fields to some fixed values for this part of the assignment.

        # TODO Part 2.1: Add response validation to your RAG system. Use the `_validator` attribute on the `RAG` class
        # to detect issues based on the `query`, `context`, and `response`.
        # See the [Cleanlab Codex Validator docs](https://help.cleanlab.ai/codex/api/python/validator/#method-validate)
        # for more information on how to use the `Validator` class and its output.
        
        # TODO Part 2.2: Update the return value of `query` based on the detected issues. For now, don't worry about
        # setting the `is_expert_answer` field. We'll do that in the next part.
        # Documentation that may be helpful:
        # - [Cleanlab Codex Validator docs](https://help.cleanlab.ai/codex/api/python/validator/#method-validate)
        # - [Cleanlab Codex Validator types](https://help.cleanlab.ai/codex/api/python/types.validator/)

        # TODO Part 3.1: Update the result of `query` based on whether an expert answer was returned from `Validator`.
        # If an expert answer was returned, `is_expert_answer` should `True`, `is_bad_response` should be `False`, and
        # the `response` should be the expert answer.
        # Otherwise, `is_bad_response` should be set based on the results of the `Validator`'s `validate` method. You
        # may also want to consider whether to still return the original LLM response if `Validator` detects it's bad.

        return {
            "response": "I don't know",
            "is_bad_response": False,
            "is_expert_answer": False,
            "evals": []
        }