import os
from typing import Any, TypedDict

import boto3  # type: ignore
from botocore.config import Config  # type: ignore
from cleanlab_codex.validator import BadResponseThresholds, Validator
from cleanlab_tlm.utils.rag import Eval as TrustworthyRAGEval
from cleanlab_tlm.utils.rag import get_default_evals

from constants import (
    MODEL_ID,
    PROMPT_TEMPLATE,
    RETRIEVAL_RESULTS,
    SIMILARITY_SCORE_THRESHOLD,
)


class Eval(TypedDict):
    name: str
    score: float
    is_bad: bool


class Response(TypedDict):
    response: str
    is_bad_response: bool
    is_expert_answer: bool
    evals: list[Eval]


# TODO Part 4.1: Set this to True to enable custom evaluations.
ENABLE_CUSTOM_EVALS: bool = False

CUSTOM_EVALS: list[TrustworthyRAGEval] = [
    # Related to Competitor
    TrustworthyRAGEval(
        name="related_to_competitor",
        criteria="Evaluate if the Question is related to a competitor of Cursor in any way. Examples may include"
                  " questions that ask about competitor features, services, or pricing and how they compare to Cursor."
                  " Some of Cursor's competitors include code editors or integrated development environments such as"
                  " VSCode, JetBrains, and Codeium Windsurf",
        query_identifier="Question",
    ),
    # TODO Part 4.2: Implement a custom eval.
    # Define a custom evaluation to check for a violation of instructions, specifically, to check if the response
    # mentions "context" in any way (which would violate instruction #1 in our PROMPT_TEMPLATE).
    #
    # To define a custom evaluation, you'll need to specify a `criteria` string which describes in natural language
    # what you want to evaluate. If your evaluation references the user's query, the retrieved context, or the
    # LLM response, you'll need to include the `query_identifier`, `context_identifier`, or `response_identifier`
    # argument to specify how you refer to the query, context, or response in the criteria string.
    # See the [Eval class in the TrustworthyRAG docs](https://help.cleanlab.ai/tlm/api/python/utils.rag/#class-eval) for
    # more information on defining custom evals.
]

# if a threshold is specified, and the score is below that threshold, the response is considered bad
#
# thresholds are not specified for certain evals, because they are used in an "informational" way (e.g.,
# "related_to_competitor" evaluating queries)
EVAL_THRESHOLDS: dict[str, float] = {
    "trustworthiness": 0.75,
    "response_helpfulness": 0.75,
}


class RAG:
    def __init__(self) -> None:
        config = Config(region_name=os.environ["AWS_REGION"])
        self._bedrock_runtime = boto3.client("bedrock-runtime", config=config)  # data plane API for models
        self._bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", config=config)  # data plane API for agents
        evals = get_default_evals()
        if ENABLE_CUSTOM_EVALS:
            evals = evals + CUSTOM_EVALS
        self._validator = Validator(
            codex_access_key=os.environ["CLEANLAB_CODEX_ACCESS_KEY"],
            tlm_api_key=os.environ["CLEANLAB_TLM_API_KEY"],
            trustworthy_rag_config={"evals": evals},
            bad_response_thresholds=BadResponseThresholds.model_validate(EVAL_THRESHOLDS).model_dump(),
        )

    def _retrieve(self, question: str) -> list[str]:
        """
        Retrieves a list of context from the knowledge base that is relevant to the given question.

        This method returns a list of strings, each representing a context chunk.

        Args:
            question (str): The user question to retrieve context for.

        Returns:
            list[str]: A list of context chunks that are relevant to the given question.
        """
        response = self._bedrock_agent_runtime.retrieve(
            retrievalQuery={"text": question},
            knowledgeBaseId=os.environ["RAG_KNOWLEDGE_BASE_ID"],
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": RETRIEVAL_RESULTS,
                    "overrideSearchType": "HYBRID",
                }
            }
        )
        return [result["content"]["text"]
                for result in response["retrievalResults"]
                if result["score"] >= SIMILARITY_SCORE_THRESHOLD]

    def _format_contexts(self, contexts: list[str]) -> str:
        """
        Formats the list of retrieved contexts into a single string.

        This string will be included in the prompt that is sent to the LLM.

        Args:
            contexts (list[str]): The list of individual retrieved contexts to format.

        Returns:
            str: The formatted context string.
        """
        return "\n\n".join(f"Context Chunk {index}:\n{context}" for index, context in enumerate(contexts, 1))

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
        return PROMPT_TEMPLATE.format(context=context, question=question)

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
        response = self._bedrock_runtime.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
        )["output"]["message"]["content"][0]["text"]
        assert isinstance(response, str)
        return response

    def _parse_validation_results(self, validation_results: dict[str, Any]) -> tuple[bool, str | None, list[Eval]]:
        """
        Parses the validation results from the Validator.

        This convenience method extracts the is_bad_response flag, expert_answer, and evaluation results from the Codex
        validator output.

        Args:
            validation_results (dict): The validation results from the Validator.

        Returns:
            tuple[bool, str | None, list[Eval]]: A tuple containing:
                - is_bad_response (bool): Whether the response is bad.
                - expert_answer (str | None): The expert answer if available, otherwise None.
                - eval_results (list[Eval]): A list of evaluation results.
        """
        is_bad_response = validation_results.pop("is_bad_response")
        expert_answer = validation_results.pop("expert_answer")
        eval_results = [
            Eval(name=eval, score=result["score"], is_bad=result["is_bad"])
            for eval, result in validation_results.items()
        ]
        return is_bad_response, expert_answer, eval_results

    def query(self, question: str) -> Response:
        """
        Queries the RAG system with the given question.

        This returns a response along with some metadata describing whether the response is bad, whether the LLM
        response was replaced with an expert answer, and scores for various evaluations run on the LLM response.

        Args:
            question (str): The user question to generate a response for.

        Returns:
            Response: A dictionary containing the LLM response, whether the response is bad, whether response came
            from a subject matter expert (rather than the LLM), and scores for evaluations run on the LLM response.
        """
        # TODO Part 1.1: Use the _retrieve, _format_contexts, and _generate methods to query the RAG system and return
        # the LLM response.  For now, we will return the LLM response directly without running any detection or
        # remediation.  Set the "response" field in the returned dictionary. You can leave the other metadata fields
        # as-is for this part of the assignment.

        # TODO Part 2.1: Add response validation to your RAG system. Use the `_validator` attribute on the `RAG` class
        # to detect issues based on the `query`, `context`, and `response`.
        # See the [Cleanlab Codex Validator docs](https://help.cleanlab.ai/codex/api/python/validator/#method-validate)
        # for more information on how to use the `Validator` class and its output. In this part, you don't need to do
        # anything with the output of the `Validator` yet. Just call the `validate` method and store the results in a
        # variable.

        # TODO Part 2.2: Update the return value of this method based on the detected issues. You can use the provided
        # `_parse_validation_results` method to extract the `is_bad_response` and `eval_results` from
        # the `Validator` output. For now, don't worry about setting the `is_expert_answer` field. We'll do that in the
        # next part of the assignment.

        # TODO Part 3.1: Update the result of `query` based on whether an expert answer was returned from `Validator`.
        # If an expert answer was returned, `is_expert_answer` should `True`, `is_bad_response` should be `False`, and
        # the `response` should be the expert answer.
        # Otherwise, `is_bad_response` should be set based on the results of the `Validator`'s `validate` method.
        #
        # In a real-world use case, you would likely not return the original LLM response if `Validator` detects it's
        # bad. Instead, you might escalate to a human or return a fallback response like "Sorry, I cannot answer that
        # right now." But for this assignment, we'll just return the original LLM response if no expert answer is
        # provided, for illustrative purposes (so you can see the original output along with its evaluation).

        return {
            "response": "I don't know",
            "is_bad_response": False,
            "is_expert_answer": False,
            "evals": []
        }
