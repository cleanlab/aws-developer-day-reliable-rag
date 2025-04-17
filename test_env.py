import os

from cleanlab_codex import Project
from cleanlab_tlm import TLM
from dotenv import load_dotenv
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding  # type: ignore

load_dotenv()

QUESTION = "How many syllables are in the phrase 'AI User Conference'?"

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
    project = Project.from_access_key(access_key=codex_access_key)
    project.query(QUESTION)  # Just verify we can query without error

    # check that we can query Gemini
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY is not set")
    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
        embed_batch_size=100,
        api_key=gemini_api_key,
    )
    embedding = embed_model.get_text_embedding(QUESTION)
    assert len(embedding) > 0
    print("all ok")

if __name__ == "__main__":
    main()
