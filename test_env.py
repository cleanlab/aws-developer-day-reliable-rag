import os

import boto3  # type: ignore
from botocore.config import Config  # type: ignore
from cleanlab_codex import Project as CodexProject
from cleanlab_tlm import TLM
from dotenv import load_dotenv

import patch_aiohttp  # noqa: F401
from constants import MODEL_ID

load_dotenv()

QUESTION = "Test question from test_env.py"

def main() -> None:
    # check that we can query TLM
    tlm_api_key = os.getenv("CLEANLAB_TLM_API_KEY")
    if not tlm_api_key:
        raise ValueError("CLEANLAB_TLM_API_KEY is not set")
    tlm = TLM(api_key=tlm_api_key)
    tlm_response = tlm.prompt(QUESTION)
    assert "trustworthiness_score" in tlm_response

    # check that we can query Codex
    codex_access_key = os.getenv("CLEANLAB_CODEX_ACCESS_KEY")
    if not codex_access_key:
        raise ValueError("CLEANLAB_CODEX_ACCESS_KEY is not set")
    project = CodexProject.from_access_key(access_key=codex_access_key)
    project.query(QUESTION)  # Just verify we can query without error

    # check that we can retrieve from the Bedrock knowledge base
    config = Config(region_name=os.environ["AWS_REGION"])
    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", config=config)  # data plane API for agents
    retrieval_response = bedrock_agent_runtime.retrieve(
        retrievalQuery={"text": "What models does Cursor support?"},
        knowledgeBaseId=os.environ["RAG_KNOWLEDGE_BASE_ID"],
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 5,
                "overrideSearchType": "HYBRID",
            }
        }
    )
    if len(retrieval_response["retrievalResults"]) == 0:
        raise ValueError("Knowledge base query returned no results: did you forget to sync the knowledge base (step 0.3)?")  # noqa: E501

    # check that we can generate text with the Bedrock model
    bedrock_runtime = boto3.client("bedrock-runtime", config=config)  # data plane API for models
    bedrock_runtime.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": QUESTION}]}],
    )

    print("all ok")

if __name__ == "__main__":
    main()
