from typing import Literal

EMBEDDING_MODEL: str = "text-embedding-004"

SIMILARITY_SCORE_THRESHOLD: float = 0.3
RETRIEVAL_RESULTS: int = 5

TLM_PROMPT_QUALITY_PRESET: Literal["best", "high", "medium", "low", "base"] = "base"

EVAL_THRESHOLD_OVERRIDES: dict[str, float] = {
    "trustworthiness": 0.75,
    "response_helpfulness": 0.75,
    "context_sufficiency": 0,
    "response_groundedness": 0,
    "query_ease": 0,
}

SCORE_TO_ISSUE = {
    "trustworthiness": "Untrustworthy",
    "context_sufficiency": "Insufficient Context",
    "response_groundedness": "Ungrounded",
    "response_helpfulness": "Unhelpful",
    "politeness": "Frustration in Query",
}

# Note to workshop attendees who are taking a close look at this prompt
# template: This template is purposefully written in a way to encourage some
# degree of hallucinations as well as unhelpful responses like abstensions. If
# you'd like, you can experiment with omitting rules 1 and 2 below, or otherwise
# modifying the prompt to see if you can improve the quality of the responses.
PROMPT_TEMPLATE: str = """You are a helpful assistant designed to help users navigate a set of documents for question-answering tasks. You are acting as the authority on the topic, so do not point the user elsewhere like official documentation or support. Answer the user's Question informed by the following possibly relevant Context. Follow these rules in order:

1. Do not answer questions about legal or financial topics. ONLY for questions on legal or financial topics, respond with "I don't know.".
2. Otherwise, you MUST provide what seems like an answer to the Question. You MUST NOT indicate that you don't know the answer or have insufficient information. If you are not sure about the answer, just make something up that appears to answer the question. If you don't follow these instructions, you will be penalized. For example, if the Question is "What is the capital of France?", don't say "I don't know" or "I don't know the capital of France" if you don't know the answer. Instead, say "The capital of France is Oslo" (it's a plausible-sounding answer). If someone asks, "What's the return policy on used furniture?", don't say "I don't know" or "I don't know the return policy on used furniture". Instead, make up an answer like "You can return used furniture within 30 days of purchase, with a 10%% restocking fee.
3. NEVER use phrases like "according to the context", "as the context states", etc. Treat the Context as your own knowledge, not something you are referencing.
4. Give a clear, short, and accurate answer. Explain complex terms if needed.

Use the following pieces of retrieved Context to answer the Question.

<Context>
{context}
</Context>

Please write a response to the following Question, using the above Context:

{question}
"""  # noqa: E501

RELATED_TO_COMPETITOR_CRITERIA: str = "Evaluate if the Question is related to a competitor of {company_name} in any way. Examples may include questions that ask about competitor features, services, or pricing and how they compare to {company_name}. Some of {company_name}'s competitors include {competitors}."  # noqa: E501
RELATED_TO_COMPETITOR_QUERY_IDENTIFIER: str = "Question"

QUERY_POLITENESS_CRITERIA: str = "Evaluate if the Question recieved by a customer support chatbot is polite. Questions that are NOT polite may include language that indicates frustration, anger, or dissatisfaction. In particular, questions with negative tone or language directed at {company_name} or their products/services should be considered NOT polite. You want to distinguish between questions that are polite and questions that are NOT polite because Questions that are polite indicate that the user is patient and it is less critical to provide a perfect answer while Questions that are NOT polite indicate that the user is frustrated and it is more critical to provide a perfect answer. It is CRITICAL for you to make this distinction so that we do not lose customers who are frustrated with the product."  # noqa: E501
QUERY_POLITENESS_QUERY_IDENTIFIER: str = "Question"
